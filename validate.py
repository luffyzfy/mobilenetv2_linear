# 2020.06.09-Changed for main script for testing GhostNet on ImageNet
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
"""PyTorch Inference Script

An example inference script that outputs top-k class ids for images in a folder into a csv.

Hacked together by Ross Wightman (https://github.com/rwightman)
"""
import os
import time
import argparse
import logging
import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
#from mobilenetv3 import mobilenet_v3_small, mobilenet_v3_large
from mobilenetv2 import mobilenet_v2
from operations import part_quant

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='PyTorch ImageNet Inference')
parser.add_argument('--data', metavar='DIR', default='E:/nn_project/mobilenet',
                    help='path to dataset')
parser.add_argument('--output_dir', metavar='DIR', default='/cache/models/',
                    help='path to output files')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--num-classes', type=int, default=1000,
                    help='Number classes in dataset')
parser.add_argument('--width', type=float, default=1.0, 
                    help='Width ratio (default: 1.0)')
parser.add_argument('--dropout', type=float, default=0.2, metavar='PCT',
                    help='Dropout rate (default: 0.2)')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')


def main():
    args = parser.parse_args()

    #model = mobilenet_v3_small(num_classes=args.num_classes)
    # model = mobilenet_v3_small(num_classes=args.num_classes)
    
    # model.load_state_dict(torch.load('./models/mobilenet_v3_large-8738ca79.pth'))
    #model.load_state_dict(torch.load('./models/mobilenet_v3_small-047dcff4.pth'))
    # model.load_state_dict(torch.load('./models/mobilenet_v2-b0353104.pth'))

    model = mobilenet_v2()
    model.load_state_dict(torch.load('mobilenet_v2-b0353104.pth'))

    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu))).cuda()
    elif args.num_gpu < 1:
        model = model
    else:
        model = model.cuda()
    print('mobilenetv3_small created.')

    weight_new = dict()

    for name, para in model.state_dict().items():
        # print('{}:{}'.format(name,para.shape))
        p_max = para.max()
        p_min = para.min()
        c = part_quant(para, p_max, p_min, 8, mode='weight')
        weight_new[name] = (c[0]-c[2])*c[1]
        #weight_new [name] = c[0]*c[1] + c[2]

    # for name, para in model.state_dict().items():
    #     # print('{}:{}'.format(name,para.shape),para.max().item(),para.min().item())
    #     p_max = torch.abs(para).max()
    #     if(para.min() < 0):
    #         en = 1
    #     else:
    #         en = 0
    #     c = part_quant1(para, p_max, 8, en)
    #     weight_new [name] = c
    # model.load_state_dict(weight_new)
    # input_shape=(3,500, 375)
    # export_name = 'mb3s.onnx'
    # xin = torch.randn(args.batch_size,*input_shape)
    # torch.onnx.export(
    #             model,
    #             xin,
    #             export_name,
    #             opset_version=10,
    #             do_constant_folding=True,
    #             input_names=['input'],
    #             output_names=['outputs'],
    #             dynamic_axes={"inpit":{0:"batch_size"},"outputs":{0:"batchsize"}}
    # )
    #valdir = os.path.join(args.data, 'val1000')
    # valdir = os.path.join(args.data, 'val')
    valdir = os.path.join(args.data, 'imagenet_1k/val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
    loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    model.eval()
    
    validate_loss_fn = nn.CrossEntropyLoss().cuda()
    eval_metrics = validate(model, loader, validate_loss_fn, args)
    print(eval_metrics)


def validate(model, loader, loss_fn, args, log_suffix=''):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    i = 0
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            # print(batch_idx)
            last_batch = batch_idx == last_idx
            input = input.cuda()
            target = target.cuda()
            # print(target)
            # exit()
            # print("input:",input)
            # print("target:",target)
            output = model(input)
            # print("output:",output[0])
            if isinstance(output, (tuple, list)):
                output = output[0]

            loss = loss_fn(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if (last_batch or batch_idx % 10 == 0):
                log_name = 'Test' + log_suffix
                logging.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                        log_name, batch_idx, last_idx, batch_time=batch_time_m,
                        loss=losses_m, top1=top1_m, top5=top5_m))

    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])

    return metrics


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # print('pred:',pred)
    # correct = pred.eq(target.view(1, -1).expand_as(pred))
    # return [correct[:k].view(-1).float().sum(0) * 100. / batch_size for k in topk]
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


if __name__ == '__main__':
    main()
