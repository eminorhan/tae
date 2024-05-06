import os
import sys
import argparse
import torch
print(torch.__version__)

import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import webdataset as wds

sys.path.insert(0, os.path.abspath('..'))
import tae
from  util import misc as misc
from pathlib import Path


def get_args_parser():
    parser = argparse.ArgumentParser('Encode a dataset with a TAE', add_help=False)
    parser.add_argument('--batch_size_per_gpu', default=256, type=int, help='Batch size per GPU (effective batch size is batch_size_per_gpu * accum_iter * # gpus')
    parser.add_argument('--maxcount', default=32, type=int, help='Max record count per shard')
    parser.add_argument('--data_len', default=13151276, type=int, help='Total length of dataset')

    # Model parameters
    parser.add_argument('--model', default='', type=str, help='Name of model to train')
    parser.add_argument('--model_ckpt', default='', help='Load a pretrained checkpoint')
    parser.add_argument('--input_size', default=224, type=int, help='images input size')

    # Dataset parameters
    parser.add_argument('--data_path', default='', type=str)
    parser.add_argument('--device', default='cuda', help='device to use for training/testing')
    parser.add_argument('--num_workers', default=16, type=int)
    
    # distributed training parameters
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # Misc
    parser.add_argument('--output_dir', default='', help='path where to save processed dataset')
    parser.add_argument('--save_prefix', default="", type=str, help='prefix for saving dataset')

    return parser


def main(args):
    misc.init_distributed_mode(args)
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    cudnn.benchmark = True

    # data transforms
    transform = transforms.Compose([
        transforms.Resize(args.input_size + 32, interpolation=3),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    num_iters = args.data_len // args.batch_size_per_gpu + 1

    # dataset and loader
    dataset = wds.WebDataset(args.data_path, resampled=False).decode("pil").to_tuple("jpg", "cls").map_tuple(transform, lambda x: x)
    loader = wds.WebLoader(dataset, batch_size=args.batch_size_per_gpu, num_workers=args.num_workers).with_epoch(num_iters)

    # define the model
    model_without_ddp = tae.__dict__[args.model]()
    model_without_ddp.to(device)

    # load pretrained autoencoder
    misc.load_model(args.model_ckpt, model_without_ddp)

    latents_list = []
    targets_list = []

    with torch.no_grad():
        # switch to eval mode
        model_without_ddp.eval()

        for it, (samples, targets) in enumerate(loader):
            samples = samples.to(device, non_blocking=True)

            # pass thru encoder
            with torch.cuda.amp.autocast():
                latents = model_without_ddp.forward_encoder(samples)

            latents_list.append(latents.cpu())
            targets_list.append(targets)

            if it % 100 == 0:
                print(f"Iteration {it} of {num_iters}")

        latents_list = torch.cat(latents_list, 0)
        targets_list = torch.cat(targets_list, 0)
        print(f"Final latents shape-dytpe: {latents_list.shape}-{latents_list.dtype}")
        print(f"Final targets shape-dytpe: {targets_list.shape}-{targets_list.dtype}")

        # Write the sample to the sharded pth archives.
        save_dict = {"latents": latents_list, "targets": targets_list}
        torch.save(save_dict, os.path.join(args.output_dir, f"{args.save_prefix}_{args.model}.pth"))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)