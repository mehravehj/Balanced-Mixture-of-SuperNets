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
    return multi_scale_input

def string_to_list(x):
    x = x.split(',')
    res = [int(i) for i in x]
    return res

def pooling_alpha(x, leng):
    if ',' in x:
        x = string_to_list(x)
    else:
        x = [int(x) for _ in range(leng)]
    return x

class sRes2d(nn.Module):
    def __init__(self, in_, out_, kernel_size, max_scales=4, usf=True):
        super(sRes2d, self).__init__()
        self.interp = F.interpolate
        self.max_scales = max_scales
        self.usf = usf
        self.in_ = in_
        self.out_ = out_
        self.alpha = nn.Parameter(torch.ones(self.max_scales))
        print('initial alpha:', self.alpha.data)
        self.nalpha = F.softmax(self.alpha, 0).view(-1, 1)

        self.conv1 = nn.Conv2d(in_, out_, kernel_size, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU()
        self.bn1 = nn.ModuleList([nn.BatchNorm2d(out_, affine=False) for _ in range(self.max_scales)])
        self.conv2 = nn.Conv2d(out_, out_, kernel_size, stride=1, padding=1, bias=False)
        self.bn2 = nn.ModuleList([nn.BatchNorm2d(out_, affine=False) for _ in range(self.max_scales)])
        self.bn = nn.ModuleList([nn.BatchNorm2d(out_, affine=False) for _ in range(self.max_scales)])
        if in_ != out_:
            self.conv3 = nn.Conv2d(in_, out_, kernel_size=1, stride=1, padding=0, bias=False)
            #self.bn3 = nn.ModuleList([nn.BatchNorm2d(out_, affine=False) for _ in range(self.max_scales)])


    def forward(self, x):
        self.nalpha = F.softmax(self.alpha, 0).view(-1, 1)
        # sel = F.softmax(self.alpha, 0).view(-1, 1)
        lx = downsample(x, self.max_scales)
        ly = []
        inputs = lx
        middle = []
        residual = []
        for r in range(self.max_scales):
            # print(lx[r].size())
            y = self.conv1(lx[r])
            y = self.bn1[r](y)
            y = self.relu(y)
            y = self.conv2(y)
            y = self.bn2[r](y)

            if self.in_ != self.out_:
                resid = self.conv3(lx[r])
                #resid = self.bn3[r](resid)
            else:
                resid = lx[r]
            middle.append(self.interp(y, scale_factor=2 ** r, mode='nearest').detach().data)
            y += resid
            residual.append(self.interp(y, scale_factor=2 ** r, mode='nearest').detach().data)
            y = self.relu(y)
            y = self.bn[r](y)
            ly.append(self.interp(y, scale_factor=2 ** r, mode='nearest').detach().data)
        # print(self.conv1[0].weight.data[0, 0, :, :])
        # print(self.conv1[3].weight.data[0, 0, :, :])
        output = ly
        y = (torch.stack(ly, 0))
        ys = y.shape
        # out = (y * sel.view(ys[0], 1, 1, 1, 1)).sum(0)
        out = (y * self.nalpha.view(ys[0], 1, 1, 1, 1)).sum(0)

        torch.save(inputs, './weights/8338/inputs.t7')
        torch.save(residual, './weights/8338/residual.t7')
        torch.save(middle, './weights/8338/middle.t7')
        torch.save(output, './weights/8338/output.t7')

        return out


class multires_model(nn.Module):
    def __init__(self, ncat=10, mscale='sres', channels=32, leng=10, max_scales=4, usf=True, pooling=0):
        super(multires_model, self).__init__()
        self.leng = leng
        channels = int(channels)
        listc = [sRes2d(3, channels, 3, max_scales, usf)]
        listc += [sRes2d(channels, channels, 3, max_scales, usf) for i in range(leng - 2)]
        listc += [sRes2d(channels, ncat, 3, max_scales, usf)]

        self.conv = nn.ModuleList(listc)

    def forward(self, x):
        out = x
        # for c in range(self.leng):
        #     out = self.conv[c](out)
        for c in range(2):
            out = self.conv[c](out)
        # out = F.avg_pool2d(out, kernel_size=out.size()[2:])
        # out = out.view(out.size(0), -1)
        out = F.sigmoid(out)
        # out = F.tanh(out)
        # feature_maps.append(out.detach().data)
        # save_dir = './weights/8338/fmap.t7'
        # torch.save(feature_maps, save_dir)
        return out