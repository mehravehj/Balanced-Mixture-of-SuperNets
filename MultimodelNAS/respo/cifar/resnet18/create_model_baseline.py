from typing import Optional

import numpy as np
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
        self.bn = nn.BatchNorm2d(out_, affine=False, track_running_stats=True)  # batchnorm
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
        self.bn1 = nn.BatchNorm2d(in_, affine=True, track_running_stats=True)  # batchnorm
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_, out_, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_, affine=True, track_running_stats=True)  # batchnorm
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

        #
        # out = self.conv1(x)
        # out = self.bn1(out, track_running_stats=False, affine=False)
        # out = self.relu(out)
        #
        # out = self.conv2(out)
        # out = self.bn2(out, track_running_stats=False, affine=False)

        if self.downsample is not None:
            identity = self.downsample(x)

        # if self.stride==2:
        #   if self.downsample is not None: # if channels change use conv1x1
        #     print('channel mismatch')
        #     identity = self.downsample(x)
        elif self.conv1.stride == (2, 2):  # if only spatial dim changes, downsample input
            # print('size mismatch')
            identity = nn.functional.conv2d(x, torch.ones((self.in_plane, 1, 1, 1), device=torch.device('cuda')),
                                            bias=None, stride=2, groups=self.in_plane)

        out += identity
        # out = self.relu(out)

        return out


class ResBasicBlock(nn.Module):
    def __init__(self, in_, out_, kernel_size=3, stride=1, downsample: Optional[nn.Module] = None):
        super(ResBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_, out_, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_, affine=True, track_running_stats=True)  # batchnorm
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_, out_, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_, affine=True, track_running_stats=True)  # batchnorm
        self.downsample = downsample
        self.stride = stride
        self.in_plane = in_
        self.out_plane = out_

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out, track_running_stats=True, affine=True)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, track_running_stats=True, affine=True)


        if self.downsample is not None:
            # print('channel mismatch')
            identity = self.downsample(x)
        # if self.stride==2:
        #   if self.downsample is not None: # if channels change use conv1x1
        #     print('channel mismatch')
        #     identity = self.downsample(x)
        elif self.conv1.stride == (2, 2):  # if only spatial dim changes, downsample input
            # print('size mismatch')
            identity = nn.functional.conv2d(x, torch.ones((self.out_plane, 1, 1, 1), device=torch.device('cuda')),
                                            bias=None, stride=2, groups=self.planes * self.expansion)

        out += identity
        out = self.relu(out)

        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes=5):
        print('ResNet18 init')
        # first block
        listc = []
        channels = (64, 64, 64, 128, 128, 256, 256, 512, 512)
        super(ResNet18, self).__init__()
        self.num_layers = 9
        self.inplanes = channels[0]
        self.pool = None
        #block = ResBasicBlockPreAct
        block = ResBasicBlock
        # first layer of Resnet is conv3
        # self.convb = BasicConvBlock(3, 64, kernel_size=7, stride=2, padding=3, bias=False) # for iamgenet
        self.convb = BasicConvBlock(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # for cifar
        # a maxpooling layer
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        listc += [self._make_layer(block, channels[i], 1) for i in range(1, 9)]  # would be 8 layers

        self.res = nn.ModuleList(listc)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.inplanes, num_classes)  # last layer
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            # elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            #     print(m.weight)
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        #if zero_init_residual:
        if 1:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def set_path(self, binary_new_block):
        for layer in range(0, 8):
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
        #print('0')
        #print(out.size())
        # out = self.maxpool(out)
        for c in range(8):
            out = self.res[c](out)
            #print(c + 1)
            #print(out.size())
        out = self.avgpool(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        #print(out.size())
        #         out = out.view(out.size(0), -1)
        return out

    def _make_layer(self, block, planes, stride=1):
        downsample = None
        if self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes, affine=True, track_running_stats=True),
            )

        # layers = []
        # layers.append(block(self.inplanes, planes,3, stride, downsample))
        # self.inplanes = planes
        layer = block(self.inplanes, planes, 3, stride, downsample)
        self.inplanes = planes
        # return nn.Sequential(*layers)
        return layer


def to_sum_k_rec(n, k):
    if n == 1:
        yield [k,]
    else:
        for x in range(1, k):
            for i in to_sum_k_rec(n - 1, k - x):
                yield [x,] + i


def create_search_space():
    # print(list(to_sum_k_rec(3, 5))
    path_blocks = list(to_sum_k_rec(4, 9))
    number_paths = len(path_blocks)
    print('all %d paths created: ' % (number_paths))
    print(path_blocks)
    print(len(list(to_sum_k_rec(4, 9))))
    all_paths = []
    for p in path_blocks:
        print(p)
        p[0] = p[0]-1
        cul_new_block = list(np.cumsum(p))
        cul_new_index = [i + 1 for i in cul_new_block]
        binary_new_block = [1 if i in cul_new_index else 0 for i in range(1, 9)]
        binary_new_block = tuple(binary_new_block)
        all_paths.append(binary_new_block)

    if number_paths != len(all_paths):
        print('bug... please fix')
    print(path_blocks)

    return tuple(all_paths), number_paths


# def _resnet(
#     block: Type[Union[BasicBlock, Bottleneck]],
#     layers: List[int],
#     weights: Optional[WeightsEnum],
#     progress: bool,
#     **kwargs: Any,
# ) -> ResNet:
#     if weights is not None:
#         _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

#     model = ResNet(block, layers, **kwargs)

#     if weights is not None:
#         model.load_state_dict(weights.get_state_dict(progress=progress))

#     return model

# model = resnet50(pretrained=False)



