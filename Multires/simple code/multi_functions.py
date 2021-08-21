import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os
import torchvision.datasets as datasets
from random import shuffle
from copy import deepcopy as dcp

#import torch
from torch.utils.data import Dataset, DataLoader

def _data_transforms_cifar10():
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, test_transform


def data_loader(dataset, valid_percent, batch_size, indices):
    print('Dataset: ', dataset)
    if dataset == 'CIFAR10':
        train_transform_CIFAR, test_transform_CIFAR = _data_transforms_cifar10()
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform_CIFAR)
        valset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=test_transform_CIFAR)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform_CIFAR)
        num_class = 10

    if valid_percent:  # create validation set
        num_train = len(trainset)
        train_size = num_train - int(valid_percent * num_train)
        val_size = num_train - train_size
        print('training set size:', train_size)
        print('validation set size:', val_size)
        if not indices:
            indices = list(range(num_train))
            shuffle(indices)
            split = int(np.floor((1 - valid_percent) * num_train))
            train_indc = indices[:split]
            val_indc = indices[split:]
            indices = [train_indc, val_indc]
        else:
            train_indc = indices[0]
            val_indc = indices[1]
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indc), num_workers=6, pin_memory=False, drop_last=True)
        valloader = torch.utils.data.DataLoader(
            valset, batch_size=batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(val_indc), num_workers=6, pin_memory=False, drop_last=True)
    else:  # validation set is the same as trainset
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=False, drop_last=True)
        valloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=False, drop_last=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=6, pin_memory=False, drop_last=True)

    return trainloader, valloader, testloader, indices, num_class


def downsample(x, nres):
    lx = [x]
    for idx in range(nres - 1):
        xx = (F.max_pool2d(lx[idx], kernel_size=2))
        lx.append(xx)
    return lx


class sConv2d(nn.Module):
    def __init__(self, in_, out_, kernel_size, max_scales=4, usf=True, factor=1, alinit=0, athr=0, lf=0, prun=0,
                 alpha=0, pooling=False):
        super(sConv2d, self).__init__()
        self.interp = F.interpolate
        self.max_scales = max_scales
        self.athr = athr
        self.prun = prun
        self.usf = usf
        self.prun = prun
        if lf:  # learn factor or constant factor
            self.factor = nn.Parameter(torch.ones(1) * factor)
        else:
            self.factor = factor

        if 0: #alinit
            self.alpha = nn.Parameter(alpha)
        else:
            self.alpha = nn.Parameter(torch.ones(self.max_scales))
        print('initial alpha:', self.alpha.data)

        if self.athr:
            print('threshold:', athr)
            nalpha = F.softmax(self.alpha * self.factor, 0).view(-1, 1)
            self.nalpha = torch.zeros(nalpha.size())
            print('nalpha: ', nalpha.data)
            self.corr = [r for r in range(self.max_scales) if nalpha[r] > self.athr]
            self.alpha = nn.Parameter(alpha[self.corr])
            #            self.nalpha =
            #            print(self.corr)
            if self.corr == []:
                print('Threshold is too high!')
        else:
            self.corr = [r for r in range(self.max_scales)]
            self.nalpha = F.softmax(self.alpha * self.factor, 0).view(-1, 1)

        if self.usf:
            self.conv = nn.Conv2d(in_, out_, kernel_size, stride=1, padding=1)
            self.relu = nn.ModuleList([nn.ReLU() for _ in self.corr])
            self.bn = nn.ModuleList([nn.BatchNorm2d(out_, affine=False) for _ in self.corr])

        else:
            self.conv = nn.ModuleList([nn.Conv2d(in_, out_, kernel_size, stride=1, padding=1) for _ in self.corr])
            self.relu = nn.ModuleList([nn.ReLU() for _ in self.corr])
            self.bn = nn.ModuleList([nn.BatchNorm2d(out_, affine=False) for _ in self.corr])

    def forward(self, x):
        if self.max_scales > np.floor(np.log2(x.shape[3])):
            print('number of resolutions is too high')
        sel = F.softmax(self.alpha * self.factor, 0).view(-1, 1)
        lx = downsample(x, self.max_scales)
        ly = []
        for r in self.corr:
            idx = self.corr.index(r)
            self.nalpha[r] = sel[idx]
            if self.usf:
                y = self.conv(lx[r])
                y = self.relu[idx](y)
                y = self.bn[idx](y)
            else:
                y = self.conv[idx](lx[r])
                y = self.relu[idx](y)
                y = self.bn[idx](y)
            ly.append(self.interp(y, scale_factor=2 ** r, mode='nearest'))
        y = (torch.stack(ly, 0))
        ys = y.shape
        out = (y * sel.view(ys[0], 1, 1, 1, 1)).sum(0)
        return out

class sRes2d(nn.Module):
    def __init__(self, in_, out_, kernel_size, max_scales=4, usf=True):
        super(sConv2d, self).__init__()
        self.interp = F.interpolate
        self.max_scales = max_scales
        self.usf = usf
        self.alpha = nn.Parameter(torch.ones(self.max_scales))
        print('initial alpha:', self.alpha.data)
        self.nalpha = F.softmax(self.alpha, 0).view(-1, 1)

        if self.usf:
            self.conv1 = nn.Conv2d(in_, out_, kernel_size, stride=1, padding=1, bias=False)
            self.relu = nn.ReLU()
            self.bn1 = nn.ModuleList([nn.BatchNorm2d(out_, affine=False) for _ in range(self.max_scales)])
            self.conv2 = nn.Conv2d(out_, out_, kernel_size, stride=1, padding=1, bias=False)
            self.bn2 = nn.ModuleList([nn.BatchNorm2d(out_, affine=False) for _ in range(self.max_scales)])
            if in_ != out_:
                self.conv3 = nn.Conv2d(in_, out_, kernel_size=1, stride=1, padding=0, bias=False)
                self.bn3 = nn.ModuleList([nn.BatchNorm2d(out_, affine=False) for _ in range(self.max_scales)])

        else:
            self.conv1 = nn.ModuleList([nn.Conv2d(in_, out_, kernel_size, stride=1, padding=1, bias=False) for _ in range(self.max_scales)])
            self.relu = nn.ReLU()
            self.bn1 = nn.ModuleList([nn.BatchNorm2d(out_, affine=False) for _ in range(self.max_scales)])
            self.conv2 = nn.ModuleList([nn.Conv2d(out_, out_, kernel_size, stride=1, padding=1, bias=False) for _ in range(self.max_scales)])
            self.bn2 = nn.ModuleList([nn.BatchNorm2d(out_, affine=False) for _ in range(self.max_scales)])
            if in_ != out_:
                self.conv3 = nn.ModuleList([nn.Conv2d(in_, out_, kernel_size=1, stride=1, padding=0, bias=False) for _ in range(self.max_scales)])
                self.bn3 = nn.ModuleList([nn.BatchNorm2d(out_, affine=False) for _ in range(self.max_scales)])

    def forward(self, x):
        # sel = F.softmax(self.alpha, 0).view(-1, 1)
        lx = downsample(x, self.max_scales)
        ly = []
        for r in range(self.max_scales):
            if self.usf:
                y = self.conv1(lx[r])
                y = self.bn1[r](y)
                y = self.relu(y)
                y = self.conv2(y)
                y = self.bn2[r](y)

                if self.in_ != self.out_:
                    resid = self.conv3(lx[r])
                    resid = self.bn3[r](resid)
                else:
                    resid = lx[r]
                y += resid
                y = self.relu(y)
            else:
                y = self.conv1[r](lx[r])
                y = self.bn1[r](y)
                y = self.relu(y)
                y = self.conv2[r](y)
                y = self.bn2[r](y)

                if self.in_ != self.out_:
                    resid = self.conv3[r](lx[r])
                    resid = self.bn3[r](resid)
                else:
                    resid = lx[r]
                y += resid
                y = self.relu(y)

            ly.append(self.interp(y, scale_factor=2 ** r, mode='nearest'))
        y = (torch.stack(ly, 0))
        ys = y.shape
        # out = (y * sel.view(ys[0], 1, 1, 1, 1)).sum(0)
        out = (y * self.nalpha.view(ys[0], 1, 1, 1, 1)).sum(0)
        return out


class normal_net(nn.Module):
    def __init__(self, in_, out_, kernel_size, pooling=False):
        super(normal_net, self).__init__()
        self.pool = pooling
        self.conv = nn.Conv2d(in_, out_, kernel_size, padding=1, stride=1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_, affine=True)

    def forward(self, x):
        if self.pool:
            x = F.max_pool2d(x, kernel_size=2)
        y = self.conv(x)
        y = self.relu(y)
        y = self.bn(y)
        return y

