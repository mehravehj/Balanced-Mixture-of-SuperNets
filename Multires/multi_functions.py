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

global image_names
image_names = {}


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        return pil_loader(path)


def my_loader(path):
    if path in image_names.keys():
        return image_names[path]
    else:
        from torchvision import get_image_backend
        if get_image_backend() == 'accimage':
            image_names[path] = accimage_loader(path)
        else:
            image_names[path] = pil_loader(path)
        return image_names[path]

from multiprocessing import Manager

#import torch
from torch.utils.data import Dataset, DataLoader

class Cache(Dataset):
    def __init__(self, dataset, shared_dict):
        self.shared_dict = shared_dict
        self.dataset = dataset

    def __getitem__(self, index):
        if index not in self.shared_dict:
             print('Adding {} to shared_dict'.format(index))
             # self.shared_dict[index] = torch.tensor(index)
             self.shared_dict[index] = self.dataset[index]
        return self.shared_dict[index]

    def __len__(self):
        return len(self.dataset)


# Init
#manager = Manager()
#shared_dict = manager.dict()


# this goes right after creating your dataset


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
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform_CIFAR)
        num_class = 10
    elif dataset == 'STL10':
        trainset = torchvision.datasets.STL10(
            root='./data', split='train', download=True,
            transform=transforms.Compose([
                transforms.Pad(4),
                transforms.RandomCrop(96),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))
        testset = torchvision.datasets.STL10(
            root='./data', split='test', download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))
        num_class = 10
    elif dataset == 'TIN':
        data_transforms = {
            'train': transforms.Compose([
                transforms.Pad(4),
                transforms.RandomCrop(64),
                transforms.RandomRotation(20),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomResizedCrop(size=64, scale=(0.9, 1.08), ratio=(0.99, 1)),
                # (size=(64,64), scale=(0.9, 1.08), ratio=(1,1), interpolation=2),
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ]),
            'test': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ])
        }
        data_dir = './data/tiny-imagenet-200/'
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])#, loader=my_loader)
                          for x in ['train', 'test']}
        trainset = image_datasets['train']
        testset = image_datasets['test']
        num_class = 200
        #trainset = Cache(trainset, shared_dict)
#        testset = Cache(testset, shared_dict)

    if valid_percent:  # create validation set
        train_size = len(trainset) - int(valid_percent * len(trainset))
        num_train = len(trainset)
        val_size = num_train - train_size
        print('training set size:', train_size)
        print('validation set size:', val_size)
        if dataset == 'TIN':
            smp_per_clss = int(num_train / num_class)
            val_per_class = int(val_size / num_class)
            if not indices:
                val_indc = []
                train_indc = []
                for i in range(smp_per_clss, num_train + smp_per_clss, smp_per_clss):
                    jj = list(range(i - smp_per_clss, i))
                    shuffle(jj)
                    val_indc += jj[:val_per_class]
                    train_indc += jj[val_per_class:]
                    shuffle(val_indc)
                    shuffle(train_indc)
                    indices = [train_indc, val_indc]
            else:
                train_indc = indices[0]
                val_indc = indices[1]
        else:
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
            sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indc), num_workers=4, pin_memory=True)
        valloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(val_indc), num_workers=4, pin_memory=True)
    else:  # validation set is the same as trainset
        #        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        #        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)
        valloader = dcp(trainloader)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=6, pin_memory=True)

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

