from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from multiscale_blocks import *

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

class multi_res(nn.Module):
    def __init__(self, block_type='sconv', channels_in=32, channels_out=32, kernel_size=3, max_scales=4, initial_alpha=0, factor=1):
        super(multi_res, self).__init__()
        self.alpha = nn.Parameter(define_alpha(max_scales, ini_alpha=initial_alpha, factor=factor))
        self.factor = factor
        if block_type == 'sconv':
            self.block = conv_block_same_filter(channels_in, channels_out, kernel_size, max_scales)
        elif block_type == 'sres':
            self.block = ResBlock_same_filters(channels_in, channels_out, kernel_size, max_scales)
        self.bn = nn.BatchNorm2d(channels_out, affine=False, momentum=0.01)

        # self.gamma = 1
        # a = [(i+1)**self.gamma for i in range(max_scales)]
        # self.sia_multiplier = torch.FloatTensor(a).view(-1, 1, 1, 1, 1).cuda()

    def forward(self, x):
        # abs_alpha = torch.abs(self.alpha)
        nalpha = F.softmax(self.alpha * self.factor, 0).view(-1, 1, 1, 1, 1)
        y = self.block(x)
        # out = (y * nalpha * self.sia_multiplier).sum(0)
        out = (y * nalpha).sum(0)
        out = self.bn(out)
        return out


class network_layer(nn.Module): # pooling is perfomed on the input of layer
    def __init__(self, net_type, block_type='sconv', channels_in=32, channels_out=32, kernel_size=3, max_scales=4, initial_alpha=0, factor=1 , pool=0):
        super(network_layer, self).__init__()
        if net_type == 'multires':
            self.block = multi_res(block_type, channels_in, channels_out, kernel_size, max_scales, initial_alpha, factor)

    def forward(self, x):
        out = self.block(x)
        return out
