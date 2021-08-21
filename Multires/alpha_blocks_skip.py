from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from multiscale_blocks import *


def pooling(x,pool):
    if pool > 0:
        for p in range(pool):
            x = (F.max_pool2d(x, kernel_size=2))
    elif pool < 0:
        x = F.interpolate(x, scale_factor=2 ** abs(pool), mode='nearest')
    return x

class multi_res(nn.Module):
    def __init__(self, block_type='sconv', channels_in=32, channels_out=32, kernel_size=3, max_scales=4, initial_alpha=0, factor=1, skip=1):
        super(multi_res, self).__init__()
        self.alpha = nn.Parameter(define_alpha(max_scales+skip, ini_alpha=initial_alpha, factor=factor))
        self.factor = factor
        self.max_scales = max_scales
        self.skip = skip
        print(self.skip)
        if block_type == 'sconv':
            self.block = conv_block_same_filter(channels_in, channels_out, kernel_size, max_scales)
        elif block_type == 'sres':
            self.block = ResBlock_same_filters(channels_in, channels_out, kernel_size, max_scales)
        # self.gamma = 1
        # a = [(i+1)**self.gamma for i in range(max_scales)]
        # self.sia_multiplier = torch.FloatTensor(a).view(-1, 1, 1, 1, 1).cuda()

    def forward(self, x):
        nalpha = F.softmax(self.alpha * self.factor, 0).view(-1, 1, 1, 1, 1)
        y = self.block(x)
        if self.skip:
            y = torch.cat((y, torch.unsqueeze(x,0)), 0)
        print(y.size())
        # y = y[:self.max_scales+self.skip,...]
        # y = y[:self.max_scales,...]
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
    def __init__(self, net_type, block_type='sconv', channels_in=32, channels_out=32, kernel_size=3, max_scales=4, initial_alpha=0, factor=1 , pool=0, skip=0):
        super(network_layer, self).__init__()
        if net_type == 'multires':
            self.block = multi_res(block_type, channels_in, channels_out, kernel_size, max_scales, initial_alpha, factor, skip)
        elif net_type == 'normal':
            self.block = normal_net(block_type, channels_in, channels_out, kernel_size, pool)

    def forward(self, x):
        out = self.block(x)
        return out
