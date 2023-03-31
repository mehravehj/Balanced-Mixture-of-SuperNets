from typing import Optional

import torch.nn as nn
import torch.nn.functional as F


def pooling_func(x):
    out = F.max_pool2d(x, kernel_size=2)
    return out


def relative_pooling(x):
    """
    function to determine pooling
    """
    pooling = [max(x[0]-1, 0)]
    for i in range(len(x)-1):
        pooling.append(x[i+1] - x[i])
    return pooling

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ResBasicBlock(nn.Module):
  '''
  Creates a resnet block
  :param in_: number of input channels
  :param out_: number of output channels
  :param kernel_size: convolution filter size
  :param max_scales: number of resolutions to use
  '''
  def __init__(self, in_, out_, kernel_size=3, stride=1, downsample: Optional[nn.Module] = None):
    super(ResBasicBlock, self).__init__()
    self.conv1 = nn.Conv2d(in_, out_, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(out_, affine=False, track_running_stats=False) # batchnorm
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = nn.Conv2d(out_, out_, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(out_) # batchnorm
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    if self.downsample is not None:
        identity = self.downsample(x)
        # print(identity)

    out += identity #here
    out = self.relu(out)
    return out

class BasicConvBlock(nn.Module):
  '''
  Creates a basic conv-bn-relu block
  :param in_: number of input channels
  :param out_: number of output channels
  :param kernel_size: convolution filter size
  :param max_scales: number of resolutions to use
  '''
  def __init__(self, in_, out_, kernel_size=3):
      super(BasicConvBlock, self).__init__()
      self.conv = nn.Conv2d(in_, out_, kernel_size, stride=1, padding=1, bias=False)
      self.bn = nn.BatchNorm2d(out_) # batchnorm
      self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
      out = self.conv(x)
      out = self.bn(out)
      out = self.relu(out)

      return out

class ResNet20(nn.Module):

  def __init__(self, block, num_layers, channels, num_classes=10):
    print('ResNet20 init')
    # first block
    super(ResNet20, self).__init__()
    self.num_layers = num_layers
    self.inplanes = channels[0]
    self.pool = None
    # first layer of Resnet is conv3
  
    
    listc = [BasicConvBlock(3, channels[0])] # first layer
    listc += [self._make_layer(block, channels[i], 1) for i in range(1, num_layers)] 

    self.res = nn.ModuleList(listc)
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(self.inplanes, num_classes)

  def set_path(self, path):
    # self.pool = relative_pooling(path)
    self.pool = path

  def forward(self, x):
    out = x
    for c in range(self.num_layers):
      if self.pool[c]:
          out = pooling_func(out)
      out = self.res[c](out)
    out = self.avgpool(out)
    out = out.view(out.size(0), -1)
    out = self.fc(out)
#         out = out.view(out.size(0), -1)
    return out
      
      

  def _make_layer(self, block, planes, stride=1):
    downsample = None
    if self.inplanes != planes:
        downsample = nn.Sequential(
            nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes),
        )

    layers = []
    layers.append(block(self.inplanes, planes,3, stride, downsample))
    self.inplanes = planes
    return nn.Sequential(*layers)
