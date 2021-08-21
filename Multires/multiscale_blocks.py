from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


def downsample(x, num_res): # eliminate append
    '''
    downasample input num_res times using maxpooling
    :param x: input feature map
    :param num_res: how many times to downsample
    :return: list for downsampled features
    '''
    multi_scale_input = [x]
    for idx in range(num_res - 1):
        xx = F.max_pool2d(multi_scale_input[idx], kernel_size=2)
        multi_scale_input.append(xx)
    return multi_scale_input

def define_alpha(max_scales, ini_alpha=0, factor=1):
    '''
    define parameter alpha either as uniform ones or supplied initilization
    :param max_scales:
    :param ini_alpha:
    :param factor:
    :return:
    '''
    if ini_alpha:
        alpha = torch.eye(max_scales)[ini_alpha-1] * factor
    else:
        alpha = torch.ones(max_scales) * factor
    return alpha

class conv_block_same_filter(nn.Module):
    def __init__(self, in_, out_, kernel_size, max_scales=4):
        super(conv_block_same_filter, self).__init__()
        self.interp = F.interpolate
        self.max_scales = max_scales
        self.conv = nn.Conv2d(in_, out_, kernel_size, stride=1, padding=1)
        self.relu = nn.ReLU()
        # separate BN for each resolution
        # self.bn = nn.ModuleList([nn.BatchNorm2d(out_, affine=False)
        #                          for _ in range(self.max_scales)])
        self.bn = nn.BatchNorm2d(out_, affine=False)

    def forward(self, x): # eliminate append
        lx = downsample(x, self.max_scales)
        ly = []
        for r in range(self.max_scales):
            y = self.conv(lx[r])
            y = self.relu(y)
            # separate BN for each resolution
            # y = self.bn[r](y)
            y = self.bn(y)
            ly.append(self.interp(y, scale_factor=2 ** r, mode='nearest'))
        out = (torch.stack(ly, 0))
        return out


class conv_block_normal(nn.Module):
    def __init__(self, in_, out_, kernel_size):
        super(conv_block_normal, self).__init__()
        self.conv = nn.Conv2d(in_, out_, kernel_size, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_, affine=False)

    def forward(self, x):
        y = self.conv(x)
        y = self.relu(y)
        out = self.bn(y)
        return out


class ResBlock_same_filters(nn.Module):
    def __init__(self, in_, out_, kernel_size, max_scales=4):
        super(ResBlock_same_filters, self).__init__()
        self.interp = F.interpolate
        self.max_scales = max_scales
        self.in_ = in_
        self.out_ = out_

        self.conv1 = nn.Conv2d(in_, out_, kernel_size, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU()
        # self.bn1 = nn.BatchNorm2d(out_, affine=False)
        self.bn1 = nn.ModuleList([nn.BatchNorm2d(out_, affine=False) for _ in range(self.max_scales)])
        self.conv2 = nn.Conv2d(out_, out_, kernel_size, stride=1, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(out_, affine=False)
        # separate BN for each resolution
        self.bn2 = nn.ModuleList([nn.BatchNorm2d(out_, affine=False) for _ in range(self.max_scales)])
        # self.bn = nn.ModuleList([nn.BatchNorm2d(out_, affine=False) for _ in range(self.max_scales)])  # batchnorm not in default resnet block
        ########################## xtfsaxtstax
        if in_ != out_:
            self.conv3 = nn.Conv2d(in_, out_, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn3 = nn.BatchNorm2d(out_, affine=False)

    def forward(self, x): # eliminate append
        lx = downsample(x, self.max_scales)
        ly = []
        for r in range(self.max_scales):
            y = self.conv1(lx[r])
            y = self.bn1[r](y)
            y = self.relu(y)
            y = self.conv2(y)
            y = self.bn2[r](y)

            if self.in_ != self.out_:
                resid = self.conv3(lx[r])
                resid = self.bn3(resid)
            else:
                resid = lx[r]
            y += resid
            y = self.relu(y)
            # separate BN for each resolution
            # y = self.bn[r](y)# batchnorm not in default resnet block
            ly.append(self.interp(y, scale_factor=2 ** r, mode='nearest'))
        out = (torch.stack(ly, 0))
        # print(out.size())
        return out


class ResBlock_normal(nn.Module):
    def __init__(self, in_, out_, kernel_size):
        super(ResBlock_normal, self).__init__()
        self.in_ = in_
        self.out_ = out_

        self.conv1 = nn.Conv2d(in_, out_, kernel_size, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(out_, affine=False)
        self.conv2 = nn.Conv2d(out_, out_, kernel_size, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_, affine=False)
        if in_ != out_:
            self.conv3 = nn.Conv2d(in_, out_, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn3 = nn.BatchNorm2d(out_, affine=False)

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)

        if self.in_ != self.out_:
            resid = self.conv3(x)
            resid = self.bn3(resid)
        else:
            resid = x
        y += resid
        out = self.relu(y)
        return out