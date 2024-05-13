import os
import argparse
import json
from pathlib import Path
import torch
print(torch.__version__)
import torch.backends.cudnn as cudnn

import tae
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from streaming import StreamingDataset
from torch.utils.data import DataLoader
from engine import train_one_epoch, evaluate


def get_args_parser():
    parser = argparse.ArgumentParser('Training on a downstream recognition task', add_help=False)
    parser.add_argument('--batch_size', default=256, type=int, help='Total batch size')
    parser.add_argument('--accum_iter', default=1, type=int, help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--save_prefix', default="", type=str, help="""prefix for saving checkpoint and log files""")

    # Model parameters
    parser.add_argument('--model', default='', type=str, help='Name of model to train')
    parser.add_argument('--model_ckpt', default='', type=str, help='Name of model to train')
    parser.add_argument('--num_classes', default=None, type=int, help='number of classes')
    parser.add_argument('--compile', action='store_true', help='whether to compile the model for improved efficiency (default: false)')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (absolute lr)')

    # Dataset parameters
    parser.add_argument('--data_path', default='', type=str)

    # Misc
    parser.add_argument('--output_dir', default='./output_dir', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training/testing')

    return parser


def main(args):
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    cudnn.benchmark = True

    device = torch.device(args.device)

    # train and val datasets
    train_dataset = StreamingDataset(local=args.data_path, batch_size=args.batch_size, split='train', shuffle=True)
    val_dataset = StreamingDataset(local=args.data_path, batch_size=args.batch_size, split='val', shuffle=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # define the model
    model = tae.__dict__[args.model](num_classes=args.num_classes)
    model.to(device)
    model_without_ddp = model

    # optionally compile model
    if args.compile:
        model = torch.compile(model)
    
    print(f"Model: {model_without_ddp}")
    print(f"Number of params (M): {(sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad) / 1.e6)}")

    # set wd as 0 for bias and norm layers
    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay, bias_wd=False)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95), fused=True)  # setting fused True for faster updates (hopefully)
    criterion = torch.nn.CrossEntropyLoss()
    loss_scaler = NativeScaler()

    misc.load_model(args.model_ckpt, model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    max_accuracy_1 = 0.0
    for epoch in range(args.start_epoch, args.epochs):

        train_stats = train_one_epoch(model, criterion, train_loader, optimizer, device, epoch, loss_scaler)
        test_stats = evaluate(val_loader, model, device)

        if args.output_dir and test_stats["acc1"] > max_accuracy_1:
            print('Improvement in max test accuracy. Saving model!')
            save_dict = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'args': args,
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
            }
            misc.save_on_master(save_dict, os.path.join(args.output_dir, f"{args.save_prefix}_checkpoint.pth"))

        max_accuracy_1 = max(max_accuracy_1, test_stats["acc1"])

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, **{f'test_{k}': v for k, v in test_stats.items()}, 'epoch': epoch}

        if args.output_dir and misc.is_main_process():
            with open(os.path.join(args.output_dir, args.save_prefix + "_{}_log.txt".format(args.frac_retained)), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)