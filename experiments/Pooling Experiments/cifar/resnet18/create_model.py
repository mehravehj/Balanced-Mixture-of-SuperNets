from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor


'''
pre-activation resnet
resnet18
'''


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicConvBlock(nn.Module):
  '''
  Creates a basic conv-bn-relu block
  :param in_: number of input channels
  :param out_: number of output channels
  :param kernel_size: convolution filter size
  :param max_scales: number of resolutions to use
  '''
  def __init__(self, in_, out_, kernel_size=3, stride=1, padding=1, bias=False):
      super(BasicConvBlock, self).__init__()
      self.conv = nn.Conv2d(in_, out_, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
      self.bn = nn.BatchNorm2d(out_, affine=False, track_running_stats=False) # batchnorm
      self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
      # print(x.size())
      out = self.conv(x)
      out = self.bn(out)
      out = self.relu(out)

      return out



class ResBasicBlockPreAct(nn.Module):
    def __init__(self, in_, out_, kernel_size=3, stride=1, downsample: Optional[nn.Module] = None):
        super(ResBasicBlockPreAct, self).__init__()
        self.bn1 = nn.BatchNorm2d(out_, affine=False, track_running_stats=False)  # batchnorm
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_, out_, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_, affine=False, track_running_stats=False)  # batchnorm
        self.conv2 = nn.Conv2d(out_, out_, kernel_size=3, stride=1, padding=1, bias=False)
        self.downsample = downsample
        self.stride = stride
        self.in_plane = in_
        self.out_plane = out_

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        elif self.conv1.stride == (2, 2):  # if only spatial dim changes, downsample input
            identity = nn.functional.conv2d(x, torch.ones((self.in_plane, 1, 1, 1), device=torch.device('cuda')),
                                            bias=None, stride=2, groups=self.in_plane)

        out += identity

        return out

class ResBasicBlock(nn.Module):
    def __init__(self, in_, out_, kernel_size=3, stride=1, downsample: Optional[nn.Module] = None):
        super(ResBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_, out_, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_, affine=False, track_running_stats=False)  # batchnorm
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_, out_, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_, affine=False, track_running_stats=False)  # batchnorm
        self.downsample = downsample
        self.stride = stride
        self.in_plane = in_
        self.out_plane = out_

 
    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)


        if self.downsample is not None:
          identity = self.downsample(x)
        elif self.conv1.stride==(2,2): # if only spatial dim changes, downsample input
          identity = nn.functional.conv2d(x,torch.ones((self.out_plane,1,1,1),device=torch.device('cuda')), bias=None, stride=2, groups=self.in_plane)


        out += identity
        out = self.relu(out)

        return out

class ResNet18(nn.Module):
    def __init__(self, num_classes=5):
      print('ResNet18 init')
      # first block
      listc = []
      channels = (64,64,64,128,128,256,256,512,512)
      super(ResNet18, self).__init__()
      self.num_layers = 9
      self.inplanes = channels[0]
      self.pool = None
      block = ResBasicBlock
      # first layer of Resnet is conv3
      self.convb = BasicConvBlock(3, 64, kernel_size=3, stride=1, padding=1, bias=False) #for cifar
      # a maxpooling layer
      listc += [self._make_layer(block, channels[i], 1) for i in range(1, 9)] # would be 8 layers

      self.res = nn.ModuleList(listc)
      self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
      self.fc = nn.Linear(self.inplanes, num_classes) # last layer

    def set_path(self, binary_new_block):
        for layer in range(0,8):
            lay = layer
            stride = binary_new_block[lay] + 1
            if lay < 2:
                self.res[lay].conv1.stride = (stride, stride)
            elif lay == 2:
                self.res[lay].conv1.stride = (stride, stride)
                self.res[lay].downsample[0].stride = (stride, stride)
            elif 2 < lay < 4:
                self.res[lay].conv1.stride = (stride, stride)
            elif lay == 4:
                self.res[lay].conv1.stride = (stride, stride)
                self.res[lay].downsample[0].stride = (stride, stride)
            elif 4 < lay < 6:
                self.res[lay].conv1.stride = (stride, stride)
            elif lay == 6:
                self.res[lay].conv1.stride = (stride, stride)
                self.res[lay].downsample[0].stride = (stride, stride)
            elif lay > 6:
                l = lay - 6
                self.res[lay].conv1.stride = (stride, stride)



    def forward(self, x):
      out = x
      out = self.convb(out)
      for c in range(8):
        out = self.res[c](out)
      out = self.avgpool(out)
      out = self.avgpool(out)
      out = out.view(out.size(0), -1)
      out = self.fc(out)
      return out



    def _make_layer(self, block, planes, stride=1):
      downsample = None
      if self.inplanes != planes:
          downsample = nn.Sequential(
              nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
              nn.BatchNorm2d(planes,affine=False, track_running_stats=False),
          )

      layer = block(self.inplanes, planes,3, stride, downsample)
      self.inplanes = planes
      return layer
