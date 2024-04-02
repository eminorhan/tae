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
import argparse
from pathlib import Path
import torch
print(torch.__version__)
from numpy import mean as npmean
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, SequentialSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import save_image

import tae
import util.misc as misc


def get_args_parser():
    parser = argparse.ArgumentParser('TAE evaluation', add_help=False)
    parser.add_argument('--batch_size_per_gpu', default=8192, type=int, help='Batch size per GPU (effective batch size is batch_size_per_gpu * accum_iter * # gpus')
    parser.add_argument("--save_prefix", default="", type=str, help="""prefix for saving checkpoint and log files""")

    # Model parameters
    parser.add_argument('--model', default='', type=str, help='Name of model to train')
    parser.add_argument('--resume', default='', help='resume from a checkpoint')
    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    parser.add_argument('--compile', action='store_true', help='whether to compile the model for improved efficiency (default: false)')
    parser.add_argument('--display', action='store_true', help='whether to display reconstruction at the end of each epoch.')

    # Dataset parameters
    parser.add_argument('--val_data_path', default='', type=str)
    parser.add_argument('--output_dir', default='./output_dir', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training/testing')
    parser.add_argument('--num_workers', default=16, type=int)
    
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

    # val dataset and loader
    val_dataset = ImageFolder(args.val_data_path, transform=val_transform)
    val_sampler = SequentialSampler(val_dataset)
    val_loader = DataLoader(val_dataset, sampler=val_sampler, batch_size=args.batch_size_per_gpu, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    print(f"Data loaded with {len(val_dataset)} val imgs; {len(val_loader)} val iterations total.")

    # define the model
    model = tae.__dict__[args.model]()
    model.to(device)
    model_without_ddp = model

    # optionally compile model
    if args.compile:
        model = torch.compile(model)

    model = DDP(model, device_ids=[args.gpu])
    
    print(f"Model: {model_without_ddp}")
    print(f"Number of params (M): {(sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad) / 1.e6)}")

    misc.load_model(args=args, model_without_ddp=model_without_ddp)
    
    print("Starting TAE evaluation!")

    eval_loss = []

    with torch.no_grad():
        # switch to eval mode
        model.eval()

        for it, (samples, _) in enumerate(val_loader):
            samples = samples.to(device, non_blocking=True)

            if it == 0:
                samples_for_display = samples[:8, ...]  # pick 8 examples for display

            # compute loss
            with torch.cuda.amp.autocast():
                loss, _ = model(samples)

            loss_value = loss.item()
            eval_loss.append(loss_value)

        eval_loss = npmean(eval_loss)
        print(f"Eval loss: {eval_loss}")

    if args.display:
        with torch.no_grad():
            samples_for_display = samples_for_display.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                _, pred = model(samples_for_display)
                pred = model_without_ddp.unpatchify(pred)
                
            combined = torch.cat((samples_for_display, pred), 0)

            # save original images and their reconstructions
            save_image(combined, f"{args.save_prefix}_sample_reconstructions.jpg", nrow=8, padding=1, normalize=True, scale_each=True)

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)