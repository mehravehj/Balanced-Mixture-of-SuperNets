import argparse
import os
import random
import shutil
import time
from enum import Enum
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models

from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, RandomHorizontalFlip, ToTorchImage, \
    ModuleWrapper, RandomTranslate
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, RandomResizedCropRGBImageDecoder, SimpleRGBImageDecoder
from ffcv.fields.basics import IntDecoder

from torchvision import datasets,transforms

# from ffcv.loader import Loader, OrderOption
# from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, RandomHorizontalFlip, ToTorchImage
# from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, RandomResizedCropRGBImageDecoder
# from ffcv.fields.basics import IntDecoder


# from create_model_baseline import ResNet, BasicBlock, Bottleneck, create_search_space
from create_model_baseline import ResNet, BasicBlock, Bottleneck, create_search_space

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training with FFCV')
parser.add_argument('-t', '--train-file', default='/self/scr-sync/nlp/imagenet_ffcv/train_all_256_1.0_90.ffcv',
                    help='path to FFCV train dataset')
parser.add_argument('-v', '--val-file', default='/self/scr-sync/nlp/imagenet_ffcv/train_all_256_1.0_90.ffcv',
                    help='path to FFCV val dataset')
parser.add_argument('-ds', '--dataset', default='CIFAR10',
                    help='path to FFCV val dataset')
parser.add_argument('-d', '--data-dir', metavar='DIR', default='/self/scr-sync/nlp/imagenet',
                    help='path to dataset (default: /self/scr-sync/nlp/imagenet)')
parser.add_argument('-j', '--workers', default=6, type=int, metavar='N',
                    help='number of data loading processes per gpu')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50', choices=model_names, 
                    help='model architecture: ' + ' | '.join(model_names))
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--fp32', action='store_true',
                    help='train in full precision (instead of fp16)')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N',
                    help='batch size on each gpu')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--epochs', default=250, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-p', '--print-freq', default=195, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-s', '--save-dir', default='./',
                    help='directory to save checkpoints')
parser.add_argument('--dist-url', default=f'tcp://127.0.0.1:{random.randint(1, 9999)+30000}', type=str,
                    help='url used to set up distributed training')


parser.add_argument('--learning_rate', '-lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--min_learning_rate', '-mlr', default=0.001, type=float, help='min learning rate')
parser.add_argument('--weight_momentum', '-wm', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', '-wd', default=0.0005, type=float, help='weight decay')


parser.add_argument('-pn', '--path_number', type=int, default=0, help='index of the path')

best_acc1 = 0

def main():
    args = parser.parse_args()
    args.ngpus = torch.cuda.device_count()
    mp.spawn(main_worker, nprocs=args.ngpus, args=(args,))


def main_worker(gpu, args):
    print(args)
    args.gpu = gpu
    print("Use GPU: {} for training".format(args.gpu))
    dist.init_process_group('nccl', init_method=args.dist_url, rank=args.gpu, world_size=args.ngpus)
    torch.cuda.set_device(args.gpu)
    seed_np = int(np.random.randint(low=0, high=9999, size=None, dtype=int))
    print('randomseed is:', seed_np)
    torch.manual_seed(seed_np)
    np.random.seed(seed_np)
    global best_acc1
    cudnn.benchmark = True

    paths, num_paths = create_search_space()


    n_class = 100
        # mean = [0.485, 0.456, 0.406]
        # std = [0.229, 0.224, 0.225]

    # create model
    # create model
    model = ResNet(Bottleneck, [3, 4, 6, 3])

    model.avgpool = nn.AdaptiveAvgPool2d(1)
    model.fc = nn.Linear(model.fc.in_features, n_class)  
    path_test = paths[args.path_number]
    print('path ', args.path_number, 'blocks: ', path_test)
    model.set_path(path_test)
    print(model)
    model = model.to(memory_format=torch.channels_last).cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss().cuda()
    # linear scale learning rate with 256 base batch size
    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate,
                                momentum=args.weight_momentum,
                                weight_decay=args.weight_decay)
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60,120,160], gamma=0.2, last_epoch=- 1, verbose=False)


    # fp16 loss scaler
    scaler = torch.cuda.amp.GradScaler(enabled=not args.fp32)
    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=f'cuda:{args.gpu}')
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            scaler.load_state_dict(checkpoint['scaler'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))



    cifar_mean = np.array([0.49139968, 0.48215827, 0.44653124])* 255
    cifar_std = np.array([0.2023, 0.1994, 0.2010])* 255

    train_image_pipeline = [
        SimpleRGBImageDecoder(),
        RandomHorizontalFlip(),
        RandomTranslate(padding=2),
        ToTensor(),
        ToDevice(f'cuda:{args.gpu}', non_blocking=True),
        ToTorchImage(),
        NormalizeImage(cifar_mean, cifar_std, np.float16 if not args.fp32 else np.float32),
    ]

    val_image_pipeline = [
        SimpleRGBImageDecoder(),
        ToTensor(),
        ToDevice(f'cuda:{args.gpu}', non_blocking=True),
        ToTorchImage(),
        NormalizeImage(cifar_mean, cifar_std, np.float16 if not args.fp32 else np.float32)
    ]

    label_pipeline = [IntDecoder(), ToTensor(), Squeeze(),
                      ToDevice(f'cuda:{args.gpu}', non_blocking=True)]


    train_loader = Loader(args.train_file, batch_size=args.batch_size, num_workers=args.workers,
                          order=OrderOption.RANDOM, os_cache=True, drop_last=True,
                          pipelines={'image': train_image_pipeline, 'label': label_pipeline},
                          distributed=True, seed=seed_np)

    val_loader = Loader(args.val_file, batch_size=args.batch_size, num_workers=args.workers,
                        order=OrderOption.SEQUENTIAL, os_cache=True, drop_last=True,
                        pipelines={'image': val_image_pipeline, 'label': label_pipeline},
                        distributed=True)

    for epoch in range(args.start_epoch, args.epochs):
        train(train_loader, model, criterion, optimizer, scaler, epoch, args)
        with torch.no_grad():
            acc1 = validate(val_loader, model, criterion, args)
        scheduler.step()

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        if args.gpu == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict(),
                'scaler': scaler.state_dict(),
            }, is_best, args.save_dir)
    print('best accuracy', best_acc1)


def train(train_loader, model, criterion, optimizer, scaler, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(len(train_loader), [batch_time, losses, top1], 
                              prefix="Epoch: [{}]".format(epoch), is_master=args.gpu==0)

    model.train()
    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        images = images.cuda()
        target = target.cuda()
        #print(target)

        # compute loss in fp16 (unless disabled)
        with torch.cuda.amp.autocast(enabled=not args.fp32):
            output = model(images)
            loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, = accuracy(output, target)
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))

        # compute gradient and do SGD step
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i + 1)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(len(val_loader), [batch_time, losses, top1, top5], prefix='Test: ',
                             is_master=args.gpu == 0)

    model.eval()
    end = time.time()
    for i, (images, target) in enumerate(val_loader):
        images = images.cuda()
        target = target.cuda()

        # compute output in fp16 (unless disabled)
        with torch.cuda.amp.autocast(enabled=not args.fp32):
            output = model(images)
            loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, = accuracy(output, target)
        acc5, = accuracy(output, target, topk=(5,))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i + 1)

    top1.all_reduce()
    top5.all_reduce()
    progress.display_summary()

    return top1.avg


def save_checkpoint(state, is_best, filedir):
    torch.save(state, os.path.join(filedir, 'checkpoint.pth'))
    if is_best:
        shutil.copyfile(os.path.join(filedir, 'checkpoint.pth'), 
                        os.path.join(filedir, 'checkpoint_best.pth'))

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
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

    def all_reduce(self):
        total = torch.FloatTensor([self.sum, self.count]).cuda()
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", is_master=True):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.is_master = is_master

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        if self.is_master:
            print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        if self.is_master:
            print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
