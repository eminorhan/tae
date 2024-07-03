import datetime
import os
import sys
import time
import argparse
import presets
import torch
import torchvision
import utils

from torch import nn
from coco_utils import get_coco
from pathlib import Path
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath('..'))
import tae
from util import misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler


def get_dataset(args, is_train):
    def sbd(*args, **kwargs):
        return torchvision.datasets.SBDataset(*args, mode="segmentation", **kwargs)

    def voc(*args, **kwargs):
        return torchvision.datasets.VOCSegmentation(*args, **kwargs)

    paths = {
        "voc": (args.data_path, voc, 21),
        "voc_aug": (args.data_path, sbd, 21),
        "coco": (args.data_path, get_coco, 21),
    }
    p, ds_fn, num_classes = paths[args.dataset]

    image_set = "train" if is_train else "val"
    ds = ds_fn(p, image_set=image_set, transforms=get_transform(is_train))
    return ds, num_classes


def get_transform(is_train):
    if is_train:
        return presets.SegmentationPresetTrain(base_size=288, crop_size=256)
    else:
        return presets.SegmentationPresetEval(base_size=256)


def criterion(inputs, target):
    losses = {}
    for name, x in inputs.items():
        losses[name] = nn.functional.cross_entropy(x, target, ignore_index=255)

    if len(losses) == 1:
        return losses["out"]

    return losses["out"] + 0.5 * losses["aux"]


@torch.no_grad()
def evaluate(model, encoder, data_loader, device_model, device_encoder, num_classes):

    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Val:"

    for image, target in metric_logger.log_every(data_loader, 100, header):

        with torch.cuda.amp.autocast():
            image = image.to(device_encoder)
            image = encoder.forward_encoder(image)

        image = image.to(device_model)   # move to other gpu
        target = target.to(device_model)

        with torch.cuda.amp.autocast():
            output = model(image)
            output = output["out"]

        confmat.update(target.flatten(), output.argmax(1).flatten())

    confmat.reduce_from_all_processes()

    return confmat


def train_one_epoch(model, encoder, optimizer, data_loader, lr_scheduler, device_model, device_encoder, epoch, print_freq, loss_scaler):

    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    header = f"Epoch: [{epoch}]"

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                image = image.to(device_encoder)
                image = encoder.forward_encoder(image)

        image = image.to(device_model)   # move to other gpu
        target = target.to(device_model)

        with torch.cuda.amp.autocast():
            output = model(image)  # output["out"] and output["aux"] both have shape: (B, C, H, W) where B is batch size, C is the number of classes, and H, W are the spatial dimensions of image 
            loss = criterion(output, target)

        optimizer.zero_grad()
        loss_scaler(loss, optimizer, parameters=model.parameters())

        lr_scheduler.step()

        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])


def main(args):
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    torch.backends.cudnn.benchmark = True

    # TODO: this implementation uses two gpus, add an assert here to make sure there are at least two gpus
    device_encoder = torch.device('cuda:0')
    device_model = torch.device('cuda:1')

    train_dataset, num_classes = get_dataset(args, is_train=True)
    train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size_per_gpu, sampler=train_sampler, num_workers=args.workers, collate_fn=utils.collate_fn, drop_last=True)

    val_dataset, _ = get_dataset(args, is_train=False)
    val_sampler = SequentialSampler(val_dataset)
    val_loader = DataLoader(val_dataset, batch_size=1, sampler=val_sampler, num_workers=args.workers, collate_fn=utils.collate_fn)

    # define the model (a bit ugly and hacky atm)
    if args.model_ckpt:
        model = tae.__dict__[args.model](num_classes=1000)  # load imagenet-1k pretrained checkpoint
    else:
        model = tae.__dict__[args.model](num_classes=num_classes)
    model.to(device_model)
    print(f"Model: {model}")
    print(f"Number of params (M): {(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1.e6)}")

    # define the encoder
    encoder = tae.__dict__[args.encoder]()
    encoder.to(device_encoder)
    encoder.eval()
    print(f"Model: {encoder}")
    print(f"Number of params (M): {(sum(p.numel() for p in encoder.parameters() if p.requires_grad) / 1.e6)}")

    # set wd as 0 for bias and norm layers
    param_groups = misc.add_weight_decay(model, args.weight_decay, bias_wd=False)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95), fused=True)  # setting fused True for faster updates (hopefully)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=90, gamma=0.1)
    loss_scaler = NativeScaler()

    # optionally load model and encoder (a bit ugly and hacky atm)
    misc.load_model(args.model_ckpt, model, optimizer=optimizer, loss_scaler=loss_scaler)
    if args.model_ckpt:
        model.head = torch.nn.Linear(model.head.weight.shape[-1], num_classes, bias=True).to(device_model)
    misc.load_model(args.encoder_ckpt, encoder)

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):

        train_one_epoch(model, encoder, optimizer, train_loader, scheduler, device_model, device_encoder, epoch, args.print_freq, loss_scaler)
        confmat = evaluate(model, encoder, val_loader, device_model, device_encoder, num_classes)

        print(confmat)
        
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "args": args,
            "scaler": loss_scaler.state_dict()
        }

        utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description="PyTorch Segmentation Training", add_help=add_help)
    parser.add_argument("--data_path", default="", type=str, help="dataset path")
    parser.add_argument("--dataset", default="coco", type=str, help="dataset name")
    parser.add_argument('--model', default='', type=str, help='Name of model to train')
    parser.add_argument('--model_ckpt', default='', type=str, help='Model checkpoint to resume from')
    parser.add_argument('--encoder', default='', type=str, help='Name of encoder')
    parser.add_argument('--encoder_ckpt', default='', type=str, help='Encoder checkpoint to resume from')
    parser.add_argument("--aux_loss", action="store_true", help="auxiliary loss")
    parser.add_argument("--batch_size_per_gpu", default=8, type=int, help="batch size per gpu, the total batch size is $NGPU x batch_size")
    parser.add_argument("--epochs", default=30, type=int, help="number of total epochs to run")
    parser.add_argument("--workers", default=16, type=int, help="number of data loading workers (default: 16)")
    parser.add_argument("--lr", default=0.01, type=float, help="initial learning rate")
    parser.add_argument("--weight_decay", default=1e-4, type=float, help="weight decay (default: 1e-4)", dest="weight_decay")
    parser.add_argument("--print_freq", default=10, type=int, help="print frequency")
    parser.add_argument("--output_dir", default=".", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start_epoch", default=0, type=int, help="start epoch")
    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist_url", default="env://", type=str, help="url used to set up distributed training")
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
