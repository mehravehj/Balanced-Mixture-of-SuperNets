from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F


class conv_block_same_filter(nn.Module):
    def __init__(self, in_, out_, kernel_size):
        super(conv_block_same_filter, self).__init__()
        self.conv = nn.Conv2d(in_, out_, kernel_size, stride=1, padding=1)
        self.relu = nn.ReLU()
        # self.bn = nn.ModuleList([nn.BatchNorm2d(out_, affine=False) for _ in range(self.max_scales)])
        self.bn = nn.BatchNorm2d(out_, affine=False)

    def forward(self, x):
        y = self.conv(x)
        y = self.relu(y)
        out = self.bn(y)
        return out

class ResBlock_same_filters(nn.Module):
    def __init__(self, in_, out_, kernel_size):
        super(ResBlock_same_filters, self).__init__()
        self.in_ = in_
        self.out_ = out_

        self.conv1 = nn.Conv2d(in_, out_, kernel_size, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(out_, affine=False)#, momentum=1)#, track_running_stats=False)
        self.conv2 = nn.Conv2d(out_, out_, kernel_size, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_, affine=False)#, momentum=1)#, track_running_stats=False)
        # self.bn = nn.ModuleList([nn.BatchNorm2d(out_, affine=False) for _ in range(self.max_scales)])  # batchnorm not in default resnet block
        # self.bn = nn.BatchNorm2d(out_, affine=False)  # batchnorm not in default resnet block
        ########################## xtfsaxtstax
        if in_ != out_:
            self.conv3 = nn.Conv2d(in_, out_, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn3 = nn.BatchNorm2d(out_, affine=False)#, momentum=1)#, track_running_stats=False)

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
        y = self.relu(y)
        # y = self.bn(y)# batchnorm not in default resnet block
        return y

class network_layer(nn.Module): # pooling is perfomed on the input of layer
    def __init__(self, block_type='sconv', channels_in=32, channels_out=32, kernel_size=3):
        super(network_layer, self).__init__()
        if block_type == 'sconv':
            self.block = conv_block_same_filter(channels_in, channels_out, kernel_size)
        elif block_type == 'sres':
            self.block = ResBlock_same_filters(channels_in, channels_out, kernel_size)

    def forward(self, x):
        out = self.block(x)
        return out
