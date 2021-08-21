from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F
from TD_blocks import network_layer

def string_to_list(x):
    x = x.split(',')
    res = [int(i) for i in x]
    return res

def relative_path(x):
    """
    number of max pooling relative to previous layer
    :param x: list of resolution per layer
    :return: list of relative pooling(+)/upsampling(-)
    """
    pooling = [max(x[0], 0)]
    for i in range(len(x)-1):
        pooling.append(x[i+1] - x[i])
    return pooling


def downsample(x, p):
    for i in range(p):
        x = F.max_pool2d(x, kernel_size=2)
    return x


def upsample(x,p):
    x = F.interpolate(x, scale_factor=2 ** p, mode='nearest')
    return x


def identity(x):
    return x


def skip(x):
    pass

def sampling_func(p):
    if p == 0:
        smp_func = identity()
    elif p > 0:
        smp_func = downsample()
    elif p < 0:
        smp_func = upsample()
    return smp_func

def list_layer_wise(x, leng): # a list for a layer-wise property
    if ',' in x:
        x = string_to_list(x)
    else:
        x = [int(x) for _ in range(leng)]
    return x


class path_model(nn.Module):
    """
    init: create cnn model (supernet)
    sample: sample a path using a distribution
    forward: forward

    """
    def __init__(self, ncat=10, net_type='multires', mscale='sconv', channels='32', leng=3, max_scales=2):
        super(path_model, self).__init__()
        self.net_type = net_type
        self.mscale = mscale
        self.leng = leng
        self.max_scales=max_scales
        self.channels = list_layer_wise(channels, leng-2) # channels per layer, 1st and last layers are 3 and ncat

        list_layer = [network_layer(net_type, mscale, 3, self.channels[0], 3, self.max_scales)]

        list_layer += [network_layer(net_type, mscale, self.channels[i-1], self.channels[i], 3) for i in range(1, leng - 1)]

        list_layer += [network_layer(net_type, mscale, self.channels[-1], ncat, 3, self.max_scales)]
        # self.bn = nn.BatchNorm2d(ncat, affine=False)
        self.layer = nn.ModuleList(list_layer)

    def set_path(self, path):
        self.path = path
        self.relative_path = relative_path(path)
        self.sample_functions = [sampling_func(i) for i in self.relative_path]

    def forward(self, x):
        out = x
        for l in range(self.leng):
            out = self.sample_functions[l]
            out = self.layer[l](out)
        # out = self.bn(out)
        out = F.avg_pool2d(out, kernel_size=out.size()[2:])
        out = out.view(out.size(0), -1)
        return out
