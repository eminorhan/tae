import math
import sys
from typing import Iterable

import torch

import util.misc as misc


def train_one_epoch(
        model: torch.nn.Module, 
        criterion: torch.nn.Module, 
        data_loader: Iterable, 
        optimizer: torch.optim.Optimizer, 
        device: torch.device, 
        epoch: int, 
        loss_scaler
        ):

    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)
    optimizer.zero_grad()

    for _, (samples, targets) in enumerate(metric_logger.log_every(data_loader, len(data_loader) // 1, header)):

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss_scaler(loss, optimizer, clip_grad=None, parameters=model.parameters(), create_graph=False, update_grad=True)
        optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for inp, target in metric_logger.log_every(data_loader, len(data_loader) // 1, header):
        inp = inp.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(inp)
            loss = criterion(output, target)

        acc1, acc5 = misc.accuracy(output, target, topk=(1, 5))

        batch_size = inp.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'.format(
        top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss)
        )
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}