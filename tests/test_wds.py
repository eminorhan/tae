import os
import sys
import argparse
import torch
print(torch.__version__)
import torchvision.transforms as transforms
import webdataset as wds

sys.path.insert(0, os.path.abspath('..'))
from util import misc as misc


def get_args_parser():
    parser = argparse.ArgumentParser('TAE training', add_help=False)
    parser.add_argument('--batch_size_per_gpu', default=256, type=int, help='Batch size per GPU (effective batch size is batch_size_per_gpu * accum_iter * # gpus')
    parser.add_argument('--input_size', default=224, type=int, help='images input size')

    # Dataset parameters
    parser.add_argument('--train_data_path', default='', type=str)
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

    # training transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(args.input_size, scale=args.jitter_scale, ratio=args.jitter_ratio, interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # train dataset and loader
    train_dataset = wds.WebDataset(args.train_data_path, resampled=True).shuffle(10000, initial=10000).decode("pil").to_tuple("jpg", "cls").map_tuple(train_transform, lambda x: x)
    train_loader = wds.WebLoader(train_dataset, shuffle=False, batch_size=args.batch_size_per_gpu, num_workers=args.num_workers)

    for it, (samples, targets) in enumerate(train_loader):
        print('Iter, samples shape, targets:', it+1, samples.shape, targets)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)