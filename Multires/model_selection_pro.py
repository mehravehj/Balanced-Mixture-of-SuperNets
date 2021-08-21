from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from alpha_blocks import network_layer


def string_to_list(x):
    x = x.split(',')
    res = [int(i) for i in x]
    return res


def relative_pooling(x):
    pooling = [max(x[0]-1, 0)]
    for i in range(len(x)-1):
        pooling.append(x[i+1] - x[i])
    return pooling


def pooling_alpha(x, leng):
    if ',' in x:
        x = string_to_list(x)
    else:
        x = [int(x) for _ in range(leng)]
    return x


class multires_model(nn.Module):
    def __init__(self, ncat=10, net_type='multires', mscale='sconv', channels='32', leng=3, max_scales=2, factor=1, initial_alpha='0', pool=[1], current_layer=0):
        super(multires_model, self).__init__()
        self.net_type = net_type
        self.mscale = mscale
        self.leng = leng
        self.max_scales=max_scales
        self.channels = pooling_alpha(channels, leng-1)
        self.initial_alpha = pooling_alpha(initial_alpha, leng)
        # pool = pooling_alpha(pool, leng)
        print('pool', pool)
        self.abs_pool = pool[current_layer]
        self.pool = relative_pooling(pool)
        # print('pool', self.pool)
        # print(self.pool)
        # print(self.initial_alpha)
        # print(self.channels)
        self.current_layer = current_layer

        list_layer = [network_layer('normal', mscale, 3, self.channels[0], 3, self.max_scales,
                                    self.initial_alpha[0], factor, self.pool[0])]
        if self.current_layer != self.leng-1:

            list_layer += [network_layer('normal', mscale, self.channels[i-1], self.channels[i], 3, self.max_scales,
                                         self.initial_alpha[i], factor, self.pool[i]) for i in range(1, self.current_layer+1)]

            list_layer += [network_layer(net_type, mscale, self.channels[i-1], self.channels[i], 3, self.max_scales,
                                         self.initial_alpha[i], factor, 0) for i in range(self.current_layer+1, leng - 1)]

            list_layer += [network_layer(net_type, mscale, self.channels[-1], ncat, 3, self.max_scales,
                                         self.initial_alpha[-1], factor, 0)]
        else:
            list_layer += [network_layer('normal', mscale, self.channels[i - 1], self.channels[i], 3, self.max_scales,
                                         self.initial_alpha[i], factor, self.pool[i]) for i in
                           range(1, self.leng-1)]

            list_layer += [network_layer('normal', mscale, self.channels[-1], ncat, 3, self.max_scales,
                                         self.initial_alpha[-1], factor, self.pool[-1])]

        # self.bn = nn.BatchNorm2d(ncat, affine=False)
        self.layer = nn.ModuleList(list_layer)

    def forward(self, x):
        out = x
        for l in range(self.current_layer+1):
            out = self.layer[l](out)
            # print('layer ', l)
            # print(out.size())
        if out.size() != x.size():
            out = F.interpolate(out, scale_factor=2 ** (self.abs_pool-1), mode='nearest')
        # print('layer ', l)
        # print(out.size())
        for l in range(self.current_layer+1, self.leng):
            out = self.layer[l](out)
        # out = self.bn(out)
        out = F.avg_pool2d(out, kernel_size=out.size()[2:])
        out = out.view(out.size(0), -1)
        return out