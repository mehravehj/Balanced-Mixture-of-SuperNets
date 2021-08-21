from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from multiscale_blocks import *
import numpy as np


def pooling(x,pool):
    if pool:
        for p in range(pool):
            x = (F.max_pool2d(x, kernel_size=2))
    return x

def alpha_dropout(alpha, max_scales, dropout_rate):
    indices = np.random.randint(0,max_scales-1,dropout_rate)



class multi_res(nn.Module):
    def __init__(self, block_type='sconv', channels_in=32, channels_out=32, kernel_size=3, max_scales=4, initial_alpha=0, factor=1):
        super(multi_res, self).__init__()
        self.alpha = nn.Parameter(define_alpha(max_scales, ini_alpha=initial_alpha, factor=factor))
        self.factor = factor
        self.drop_res = 0
        self.max_scales = max_scales
        if block_type == 'sconv':
            self.block = conv_block_same_filter(channels_in, channels_out, kernel_size, max_scales)
        elif block_type == 'sres':
            self.block = ResBlock_same_filters(channels_in, channels_out, kernel_size, max_scales)


    def forward(self, x):
        if self.drop_res:
            indices = []
            while indices==[]:
                indices = torch.Tensor([i for i in range(1, self.max_scales + 1)])
                indices = F.dropout(indices.type(torch.FloatTensor), p=0.5)
                # print(indices)
                indices = np.sort(indices)
                indices = [int(j / 2) - 1 for j in indices if j != 0]
            nalpha = F.softmax(self.alpha[indices], 0).view(-1, 1, 1, 1, 1)
            y = self.block(x)[indices]
        else:
            nalpha = F.softmax(self.alpha * self.factor, 0).view(-1, 1, 1, 1, 1)
            y = self.block(x)
        out = (y * nalpha).sum(0)
        # print(out.size())




        # blah = [[0, 0], [0, 2]]
        # val = (self.alpha.view(1,-1))[blah]
        # print(val)
        # print(self.alpha)
        # s = torch.sparse_coo_tensor(list(zip(*blah)), val, (1, self.max_scales))
        # print(s)
        # nalpha = F.softmax(self.alpha * self.factor, 0).view(-1, 1, 1, 1, 1)
        # print(nalpha)
        # ss = F.softmax(s._values()).view(-1, 1, 1, 1, 1)
        # print(ss)
        # y = self.block(x)
        # print('y size', y.size())
        # print('nalpha size', nalpha.size())
        # out = (y * nalpha).sum(0)
        # xx =
        # y = self.block(x)[i for i in blah[j,i]]
        # print(y.size())
        # print(ss.size())
        # out = (y * ss).sum(0)
        # print('out size', out.size())
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
