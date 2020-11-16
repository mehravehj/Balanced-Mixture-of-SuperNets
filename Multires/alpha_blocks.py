from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from multiscale_blocks import *


def pooling(x,pool):
    if pool:
        for p in range(pool):
            x = (F.max_pool2d(x, kernel_size=2))
    return x

class multi_res(nn.Module):
    def __init__(self, block_type='sconv', channels_in=32, channels_out=32, kernel_size=3, max_scales=4, initial_alpha=0, factor=1):
        super(multi_res, self).__init__()
        self.alpha = nn.Parameter(define_alpha(max_scales, ini_alpha=initial_alpha, factor=factor))
        self.factor = factor
        if block_type == 'sconv':
            self.block = conv_block_same_filter(channels_in, channels_out, kernel_size, max_scales)
        elif block_type == 'sres':
            self.block = ResBlock_same_filters(channels_in, channels_out, kernel_size, max_scales)

    def forward(self, x):
        abs_alpha = torch.abs(self.alpha)
        nalpha = F.softmax(abs_alpha * self.factor, 0).view(-1, 1, 1, 1, 1)
        y = self.block(x)
        out = (y * nalpha).sum(0)
        return out

class normal_net(nn.Module): # pooling is perfomed on the input of layer
    def __init__(self, block_type='sconv', channels_in=32, channels_out=32, kernel_size=3, pool=0):
        super(normal_net, self).__init__()
        self.pool = pool
        if block_type == 'sconv':
            self.block = conv_block_normal(channels_in, channels_out, kernel_size)
        elif block_type == 'sres':
            self.block = ResBlock_normal(channels_in, channels_out, kernel_size)

    def forward(self, x):
        y = pooling(x, self.pool)
        out = self.block(y)
        return out



class network_layer(nn.Module): # pooling is perfomed on the input of layer
    def __init__(self, net_type, block_type='sconv', channels_in=32, channels_out=32, kernel_size=3, max_scales=4, initial_alpha=0, factor=1 , pool=0):
        super(network_layer, self).__init__()
        if net_type == 'multires':
            self.block = multi_res(block_type, channels_in, channels_out, kernel_size, max_scales, initial_alpha, factor)
        elif net_type == 'normal':
            self.block = normal_net(block_type, channels_in, channels_out, kernel_size, pool)

    def forward(self, x):
        out = self.block(x)
        return out
