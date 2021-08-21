from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

def identity_filter():
    kernel = [[0, 0, 0],
              [0, 1, 0],
              [0, 0, 0]]
    kernel = torch.tensor(kernel)
    kernel = kernel.float()
    return kernel.cuda()

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


class sconv_id(nn.Module):
    def __init__(self, in_, out_, kernel_size, max_scales=4):
        super(sconv_id, self).__init__()
        self.interp = F.interpolate
        self.max_scales = max_scales
        self.in_ = in_
        self.out_ = out_
        self.alpha = nn.Parameter(torch.ones(self.max_scales))
        print('initial alpha:', self.alpha.data)
        self.nalpha = F.softmax(self.alpha, 0).view(-1, 1)
        self.kernel = identity_filter().view(1, 1, 3, 3).repeat(out_, 1, 1, 1)
        self.relu = nn.ReLU()
        self.bn = nn.ModuleList([nn.BatchNorm2d(out_, affine=False) for _ in range(self.max_scales)])

    def forward(self, x):
        self.nalpha = F.softmax(self.alpha, 0).view(-1, 1)
        lx = downsample(x, self.max_scales)
        ly = []
        for r in range(self.max_scales):
            y = F.conv2d(lx[r], self.kernel, stride=1, padding=1, groups=self.out_)
            y = self.relu(y)
            y = self.interp(y, scale_factor=2 ** r, mode='nearest')
            y = self.bn[r](y)
            ly.append(y)
            # ly.append(self.interp(y, scale_factor=2 ** r, mode='nearest'))
        y = (torch.stack(ly, 0))
        ys = y.shape
        out = (y * self.nalpha.view(ys[0], 1, 1, 1, 1)).sum(0)
        return out

class sRes2d(nn.Module):
    def __init__(self, in_, out_, kernel_size, max_scales=4):
        super(sRes2d, self).__init__()
        self.interp = F.interpolate
        self.max_scales = max_scales
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
            self.bn3 = nn.ModuleList([nn.BatchNorm2d(out_, affine=False) for _ in range(self.max_scales)])


    def forward(self, x):
        self.nalpha = F.softmax(self.alpha, 0).view(-1, 1)
        lx = downsample(x, self.max_scales)
        ly = []
        for r in range(self.max_scales):
            y = self.conv1(lx[r])
            y = self.bn1[r](y)
            y = self.relu(y)
            y = self.conv2(y)
            y = self.bn2[r](y)
            if self.in_ != self.out_:
                resid = self.conv3(lx[r])
                resid = self.bn3[r](resid)
            else:
                resid = lx[r]
            y += resid
            y = self.relu(y)
            y = self.interp(y, scale_factor=2 ** r, mode='nearest')
            y = self.bn[r](y)
            ly.append(y)
            # ly.append(self.interp(y, scale_factor=2 ** r, mode='nearest'))
        y = (torch.stack(ly, 0))
        ys = y.shape
        out = (y * self.nalpha.view(ys[0], 1, 1, 1, 1)).sum(0)
        return out


class multires_model(nn.Module):
    def __init__(self, channels=32, leng=10, max_scales=4):
        super(multires_model, self).__init__()
        self.leng = leng
        channels = int(channels)
        listc = [sRes2d(3, channels, 3, max_scales)]
        listc += [sRes2d(channels, channels, 3, max_scales) for i in range(3)]
        listc += [sRes2d(channels, 4, 3, max_scales)]
        listc += [sRes2d(4, channels, 3, max_scales)]
        listc += [sRes2d(channels, channels, 3, max_scales) for i in range(6,9)]
        listc += [sRes2d(channels, 3, 3, max_scales)]
        # listc = [sRes2d(3, channels, 3, max_scales)]
        # listc += [sRes2d(channels, channels, 3, max_scales) for i in range(leng - 2)]
        # listc += [sRes2d(channels, 3, 3, max_scales)]
        # listc = [sconv_id(3, channels, 3, max_scales)]
        # listc += [sconv_id(channels, channels, 3, max_scales) for i in range(leng - 2)]
        # listc += [sconv_id(channels, 3, 3, max_scales)]

        self.conv = nn.ModuleList(listc)

    def forward(self, x):
        out = x
        # feature_maps = [x]
        for c in range(5):
            out = self.conv[c](out)
        out = F.max_pool2d(out, kernel_size=2)
        out = F.max_pool2d(out, kernel_size=2)
        out = F.max_pool2d(out, kernel_size=2)
        out = F.interpolate(out, scale_factor=2, mode='nearest')
        out = F.interpolate(out, scale_factor=2, mode='nearest')
        out = F.interpolate(out, scale_factor=2, mode='nearest')
        for c in range(5,self.leng):
            out = self.conv[c](out)
        out = F.sigmoid(out)
        #out = F.tanh(out)
        # feature_maps.append(out.detach().data)
        # save_dir = './weights/8315/weights.t7'
        # torch.save(feature_maps, save_dir)
        return out
