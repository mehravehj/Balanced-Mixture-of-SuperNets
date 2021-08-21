from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
# from multiscale_blocks import *

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
    #check
    # print('downsample size check')
    # for i in range(num_res):
    #     print(multi_scale_input[i].size())
    return multi_scale_input

def string_to_list(x):
    x = x.split(',')
    res = [int(i) for i in x]
    return res

class normal_res(nn.Module):
    def __init__(self, in_, out_, kernel_size, max_scales=4, usf=True, skip=True):
        super(normal_res, self).__init__()
        self.interp = F.interpolate
        self.in_ = in_
        self.out_ = out_
        self.alpha = nn.Parameter(torch.ones(max_scales))


        self.conv1 = nn.Conv2d(in_, out_, kernel_size, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(out_, affine=False)
        self.conv2 = nn.Conv2d(out_, out_, kernel_size, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_, affine=False)
        # self.bn3 = nn.ModuleList([nn.BatchNorm2d(out_, affine=False) for _ in range(self.max_scales)])
        if in_ != out_:
            self.conv3 = nn.Conv2d(in_, out_, kernel_size=1, stride=1, padding=0, bias=False)



    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)

        if self.in_ != self.out_:
            resid = self.conv3(x)
        else:
            resid = x
        y1 = y + resid
        y1 = self.relu(y1)
        # y1 = self.bn3[r](y1)

        ys = y1.shape
        out = y1
        return out


class multires_model(nn.Module):
    def __init__(self, ncat=10, channels=32, leng=10, max_scales=4, usf=True):
        super(multires_model, self).__init__()
        self.leng = leng
        channels = int(channels)
        listc = [normal_res(3, channels, 3, max_scales, usf, False)]
        listc += [normal_res(channels, channels, 3, max_scales, usf, True) for i in range(leng - 2)]
        listc += [normal_res(channels, ncat, 3, max_scales, usf, False)]

        self.conv = nn.ModuleList(listc)

    def forward(self, x):
        out = x
        # for c in range(self.leng):
        out = self.conv[0](out)
        out = F.max_pool2d(out, kernel_size=2)
        out = self.conv[1](out)
        out = F.max_pool2d(out, kernel_size=2)
        out = self.conv[2](out)
        out = self.conv[3](out)
        out = F.max_pool2d(out, kernel_size=2)
        out = self.conv[4](out)
        out = self.conv[5](out)
        out = F.avg_pool2d(out, kernel_size=out.size()[2:])
        out = out.view(out.size(0), -1)
        return out