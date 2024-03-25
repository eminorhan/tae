# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import os
import sys
import math
import argparse
import json
from pathlib import Path
import torch
print(torch.__version__)
from numpy import mean as npmean
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import save_image

import dae
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler


def get_args_parser():
    parser = argparse.ArgumentParser('DAE training', add_help=False)
    parser.add_argument('--batch_size_per_gpu', default=256, type=int, help='Batch size per GPU (effective batch size is batch_size_per_gpu * accum_iter * # gpus')
    parser.add_argument('--epochs', default=999999, type=int)
    parser.add_argument('--accum_iter', default=1, type=int, help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument("--save_prefix", default="", type=str, help="""prefix for saving checkpoint and log files""")

    # Model parameters
    parser.add_argument('--model', default='dae_huge_patch14', type=str, help='Name of model to train')
    parser.add_argument('--resume', default='', help='resume from a checkpoint')
    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    parser.add_argument('--compile', action='store_true', help='whether to compile the model for improved efficiency (default: false)')
    parser.add_argument('--display', action='store_true', help='whether to display reconstruction at the end of each epoch.')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (absolute lr)')
    parser.add_argument('--min_lr', type=float, default=0.0001, help='lower lr bound for cyclic schedulers that hit 0')

    # Dataset parameters
    parser.add_argument('--train_data_path', default='', type=str)
    parser.add_argument('--val_data_path', default='', type=str)
    parser.add_argument('--output_dir', default='./output_dir', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training/testing')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument("--jitter_scale", default=[0.2, 1.0], type=float, nargs="+")
    parser.add_argument("--jitter_ratio", default=[3.0/4.0, 4.0/3.0], type=float, nargs="+")

    # distributed training parameters
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser


def main(args):
    misc.init_distributed_mode(args)
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    cudnn.benchmark = True

    # validation transforms
    val_transform = transforms.Compose([
        transforms.Resize(args.input_size + 32, interpolation=3),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # training transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(args.input_size, scale=args.jitter_scale, ratio=args.jitter_ratio, interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # train and val datasets and loaders
    val_dataset = ImageFolder(args.val_data_path, transform=val_transform)
    val_sampler = SequentialSampler(val_dataset)
    val_loader = DataLoader(val_dataset, sampler=val_sampler, batch_size=16*args.batch_size_per_gpu, num_workers=args.num_workers, pin_memory=True, drop_last=False)  # note we use a larger batch size for eval

    train_dataset = ImageFolder(args.train_data_path, transform=train_transform)
    train_sampler = DistributedSampler(train_dataset, num_replicas=misc.get_world_size(), rank=misc.get_rank(), shuffle=True)
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size_per_gpu, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    print(f"Data loaded with {len(train_dataset)} train and {len(val_dataset)} val imgs.")
    print(f"{len(train_loader)} train and {len(val_loader)} val iterations per epoch.")

    # define the model
    model = dae.__dict__[args.model]()
    model.to(device)
    model_without_ddp = model

    # optionally compile model
    if args.compile:
        model = torch.compile(model)

    model = DDP(model, device_ids=[args.gpu])  # TODO: try FSDP
    
    print(f"Model: {model_without_ddp}")
    print(f"Number of params (M): {(sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad) / 1.e6)}")

    # set wd as 0 for bias and norm layers
    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay, bias_wd=False)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95), fused=True)  # setting fused True for faster updates (hopefully)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    optimizer.zero_grad()

    print("Starting DAE training!")
    for epoch in range(args.start_epoch, args.epochs):

        train_loader.sampler.set_epoch(epoch)
        header = 'Epoch: [{}]'.format(epoch)

        for it, (samples, _) in enumerate(metric_logger.log_every(train_loader, len(train_loader) // 1, header)):

            # TODO: push this shit to eval
            if epoch == 0 and it == 0:
                samples_for_display_and_softmax = samples[:8, ...]  # pick 8 examples for display and softmax estimation at each epoch end

            samples = samples.to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                loss, _ = model(samples)

            loss_value = loss.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            loss = loss / args.accum_iter
            loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=(it + 1) % args.accum_iter == 0)
            if (it + 1) % args.accum_iter == 0:
                optimizer.zero_grad()

            torch.cuda.synchronize()

            metric_logger.update(loss=loss_value)

        # estimate eval loss at the end of epoch
        eval_loss = evaluate(val_loader, model, device)

        # ============ writing logs + saving checkpoint at the end of epoch ============ #
        save_dict = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'args': args,
            'epoch': epoch,
            'scaler': loss_scaler.state_dict(),
        }

        misc.save_on_master(save_dict, os.path.join(args.output_dir, args.save_prefix + '_checkpoint.pth'))

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'eval_loss': eval_loss, 'epoch': epoch}

        if misc.is_main_process():
            with (Path(args.output_dir) / (args.save_prefix + "_log.txt")).open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        if args.display:
            with torch.no_grad():
                samples_for_display_and_softmax = samples_for_display_and_softmax.to(device, non_blocking=True)
                with torch.cuda.amp.autocast():
                    _, pred = model(samples_for_display_and_softmax)
                    pred = model_without_ddp.unpatchify(pred)
                    
                combined = torch.cat((samples_for_display_and_softmax, pred), 0)

                # save original images and their reconstructions
                save_image(combined, f"{args.save_prefix}_reconstructions_epoch_{epoch}.jpg", nrow=8, padding=1, normalize=True, scale_each=True)

        # start a fresh logger to wipe off old stats
        metric_logger = misc.MetricLogger(delimiter="  ")

@torch.no_grad()
def evaluate(data_loader, model, device):

    # switch to eval mode
    model.eval()

    eval_loss = []

    for _, (samples, _) in enumerate(data_loader):
        samples = samples.to(device, non_blocking=True)

        # compute loss
        with torch.cuda.amp.autocast():
            loss, _ = model(samples)

        loss_value = loss.item()
        eval_loss.append(loss_value)

    eval_loss = npmean(eval_loss)
    print(f"Eval loss at the end of epoch: {eval_loss}")

    return eval_loss

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)