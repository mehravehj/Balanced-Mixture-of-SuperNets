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

        if self.usf:
            self.conv1 = nn.Conv2d(in_, out_, kernel_size, stride=1, padding=1, bias=False)
            self.relu = nn.ReLU()
            self.bn1 = nn.ModuleList([nn.BatchNorm2d(out_, affine=False) for _ in range(self.max_scales)])
            self.conv2 = nn.Conv2d(out_, out_, kernel_size, stride=1, padding=1, bias=False)
            self.bn2 = nn.ModuleList([nn.BatchNorm2d(out_, affine=False) for _ in range(self.max_scales)])
            self.bn3 = nn.ModuleList([nn.BatchNorm2d(out_, affine=False) for _ in range(self.max_scales)])
            if in_ != out_:
                self.conv3 = nn.Conv2d(in_, out_, kernel_size=1, stride=1, padding=0, bias=False)

        else:
            self.conv1 = nn.ModuleList([nn.Conv2d(in_, out_, kernel_size, stride=1, padding=1, bias=False) for _ in range(self.max_scales)])
            self.relu = nn.ReLU()
            self.bn1 = nn.ModuleList([nn.BatchNorm2d(out_, affine=False) for _ in range(self.max_scales)])
            self.conv2 = nn.ModuleList([nn.Conv2d(out_, out_, kernel_size, stride=1, padding=1, bias=False) for _ in range(self.max_scales)])
            self.bn2 = nn.ModuleList([nn.BatchNorm2d(out_, affine=False) for _ in range(self.max_scales)])
            self.bn3 = nn.ModuleList([nn.BatchNorm2d(out_, affine=False) for _ in range(self.max_scales)])
            if in_ != out_:
                self.conv3 = nn.ModuleList([nn.Conv2d(in_, out_, kernel_size=1, stride=1, padding=0, bias=False) for _ in range(self.max_scales)])
            # copy weights from 1st res
            for r in range(1,self.max_scales):
                self.conv1[r].weight.data = self.conv1[0].weight.detach().clone().data
                self.conv2[r].weight.data = self.conv2[0].weight.detach().clone().data
                if in_ != out_:
                    self.conv3[r].weight.data = self.conv3[0].weight.detach().clone().data

    def forward(self, x):
        self.nalpha = F.softmax(self.alpha, 0).view(-1, 1)
        feature_maps = {}
        lx = downsample(x, self.max_scales)
        feature_maps['input'] = lx
        feature_maps['conv1'] = []
        feature_maps['bn1'] = []
        feature_maps['relu1'] = []
        feature_maps['conv2'] = []
        feature_maps['bn2'] = []
        feature_maps['resid'] = []
        feature_maps['relu2'] = []
        feature_maps['bn3'] = []

        ly = []
        for r in range(self.max_scales):
            if self.usf:
                y = self.conv1(lx[r])
                feature_maps['conv1'].append(y.detach().data)
                y = self.bn1[r](y)
                feature_maps['bn1'].append(y.detach().data)
                y = self.relu(y)
                feature_maps['relu1'].append(y.detach().data)
                y = self.conv2(y)
                feature_maps['conv2'].append(y.detach().data)
                y = self.bn2[r](y)
                feature_maps['bn2'].append(y.detach().data)

                if self.in_ != self.out_:
                    resid = self.conv3(lx[r])
                else:
                    resid = lx[r]
                y1 = y + resid
                feature_maps['resid'].append(y1.detach().data)
                y1 = self.relu(y1)
                feature_maps['relu2'].append(y1.detach().data)
                y1 = self.bn3[r](y1)
                feature_maps['bn3'].append(y1.detach().data)
            else:
                y = self.conv1[r](lx[r])
                y = self.bn1[r](y)
                y = self.relu(y)
                y = self.conv2[r](y)
                y = self.bn2[r](y)

                if self.in_ != self.out_:
                    resid = self.conv3[r](lx[r])
                else:
                    resid = lx[r]
                y1 = y + resid
                y1 = self.relu(y1)
                y1 = self.bn3[r](y1)

            ly.append(self.interp(y1, scale_factor=2 ** r, mode='nearest'))

        y2 = (torch.stack(ly, 0))
        feature_maps['res_outputs']= y2.detach().data
        ys = y2.shape
        out = (y2 * self.nalpha.view(ys[0], 1, 1, 1, 1)).sum(0)
        feature_maps['outputs'] = out.detach().data
        save_dir = './weights/8141/weights_10.t7'
        torch.save(feature_maps, save_dir)
        return out


class multires_model(nn.Module):
    def __init__(self, ncat=10, channels=32, leng=10, max_scales=4, usf=True):
        super(multires_model, self).__init__()
        self.leng = leng
        channels = int(channels)
        listc = [sRes2d(3, channels, 3, max_scales, usf)]
        listc += [sRes2d(channels, channels, 3, max_scales, usf) for i in range(leng - 2)]
        listc += [sRes2d(channels, ncat, 3, max_scales, usf)]

        self.conv = nn.ModuleList(listc)

    def forward(self, x):
        out = x
        # out = self.conv[0](out)
        for c in range(10):
            out = self.conv[c](out)
        out = F.avg_pool2d(out, kernel_size=out.size()[2:])
        out = out.view(out.size(0), -1)
        return out