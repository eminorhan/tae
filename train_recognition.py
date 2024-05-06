import os
import sys
import math
import argparse
import json
from pathlib import Path
import torch
print(torch.__version__)
import torch.backends.cudnn as cudnn

import tae
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler


def get_args_parser():
    parser = argparse.ArgumentParser('Training on a downstream recognition task', add_help=False)
    parser.add_argument('--batch_size', default=256, type=int, help='Total batch size')
    parser.add_argument('--accum_iter', default=1, type=int, help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--save_prefix', default="", type=str, help="""prefix for saving checkpoint and log files""")
    parser.add_argument('--save_freq', default=10000, type=int, help='Save checkpoint every this many iterations.')

    # Model parameters
    parser.add_argument('--model', default='', type=str, help='Name of model to train')
    parser.add_argument('--model_ckpt', default='', type=str, help='Name of model to train')
    parser.add_argument('--num_classes', default=None, type=int, help='number of classes')
    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    parser.add_argument('--compile', action='store_true', help='whether to compile the model for improved efficiency (default: false)')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (absolute lr)')

    # Dataset parameters
    parser.add_argument('--train_data_path', default='', type=str)
    parser.add_argument('--val_data_path', default='', type=str)

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
    train_data = torch.load(args.train_data_path, map_location='cpu')
    val_data = torch.load(args.val_data_path, map_location='cpu')

    n_train = train_data['targets'].shape[0]

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
    
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    optimizer.zero_grad()

    best_eval_acc1 = 0.0
    it = 0

    print("Starting TAE training!")
    # infinite stream of training data
    while True:
        # randomly sample indices
        indx = torch.randint(0, n_train, size=(args.batch_size, ))

        # take the corresponding slices of data
        samples = train_data['latents'][indx]
        targets = train_data['targets'][indx]

        # move to gpu
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)
        
        loss_value = loss.item()
        print(outputs.shape, targets.shape, loss_value)

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss = loss / args.accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=(it + 1) % args.accum_iter == 0)
        if (it + 1) % args.accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        if it != 0 and it % args.save_freq == 0:
            # estimate eval loss
            print(f"Iteration {it}, evaluating ...")
            test_stats = evaluate(val_data, args.batch_size, model_without_ddp, device)
            
            # save checkpoint only if eval_loss decreases
            if test_stats['acc1'] > best_eval_acc1:
                print("Best eval accuracy improved! Saving checkpoint.")
                save_dict = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'args': args,
                    'iteration': it,
                    'scaler': loss_scaler.state_dict(),
                }

                misc.save_on_master(save_dict, os.path.join(args.output_dir, f"{args.save_prefix}_checkpoint.pth"))
                best_eval_acc1 = test_stats['acc1']

            # gather the stats from all processes
            metric_logger.synchronize_between_processes()
            train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'eval_loss': test_stats['loss'], 'iteration': it}

            # write log
            if misc.is_main_process():
                with (Path(args.output_dir) / (args.save_prefix + "_log.txt")).open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

            # start a fresh logger to wipe off old stats
            metric_logger = misc.MetricLogger(delimiter="  ")

            # switch back to train mode, not 100% sure if this is strictly necessary since we're passing the unwrapped model to eval now
            model.train()

        it += 1    

@torch.no_grad()
def evaluate(val_data, batch_size, model, device):

    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = misc.MetricLogger(delimiter="  ")

    n_val = val_data['targets'].shape[0]    

    # switch to eval mode
    model.eval()

    for i in range(0, n_val, batch_size):

        samples = val_data['latents'][i:(i+batch_size)]
        targets = val_data['targets'][i:(i+batch_size)]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # compute loss
        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)

        acc1, acc5 = misc.accuracy(outputs, targets, topk=(1, 5))

        bsize = samples.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=bsize)
        metric_logger.meters['acc5'].update(acc5.item(), n=bsize)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'.format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)