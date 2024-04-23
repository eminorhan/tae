import os
import sys
import argparse
import torch
print(torch.__version__)
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, SequentialSampler

sys.path.insert(0, os.path.abspath('..'))
import tae
from  util import misc as misc


def get_args_parser():
    parser = argparse.ArgumentParser('TAE evaluation', add_help=False)
    parser.add_argument('--batch_size_per_gpu', default=8192, type=int, help='Batch size per GPU (effective batch size is batch_size_per_gpu * accum_iter * # gpus')

    # Model parameters
    parser.add_argument('--model', default='', type=str, help='Name of model to train')
    parser.add_argument('--resume', default='', help='resume from a checkpoint')
    parser.add_argument('--input_size', default=224, type=int, help='images input size')

    # Dataset parameters
    parser.add_argument('--val_data_path', default='', type=str)
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
    model_without_ddp = tae.__dict__[args.model]()
    model_without_ddp.to(device)
    
    print(f"Model: {model_without_ddp}")
    print(f"Number of params (M): {(sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad) / 1.e6)}")

    misc.load_model(args=args, model_without_ddp=model_without_ddp)
    
    print("Starting TAE evaluation!")
    with torch.no_grad():
        # switch to eval mode
        model_without_ddp.eval()

        for it, (samples, _) in enumerate(val_loader):
            samples = samples.to(device, non_blocking=True)

            # pass thru encoder
            with torch.cuda.amp.autocast():
                latents = model_without_ddp.forward_encoder(samples)
                print(f"Latents shape: {latents.shape}, dtype: {latents.dtype}")

            if it == 0:
                break

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)