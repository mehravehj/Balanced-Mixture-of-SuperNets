from __future__ import print_function

import argparse
import os
from datetime import datetime
from os import path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from data_loader import data_loader
from model import multires_model

def initla_alpha(m):
    print(type(m))

net = multires_model(ncat=200, net_type='multires', mscale='sconv', channels='8', leng=3,
                     max_scales=4, factor=1)
net.cuda()
net = nn.DataParallel(net)
# print(net)
# net.apply(initla_alpha)
keys = []
for name, value in net.named_parameters():
    keys.append(name)
# print(keys)
net.module.layer[0].block.factor = 2
print(net.module.layer[0].block.factor)
print(net.module.leng)
# with torch.no_grad():
#     getattr(net, keys[0])#[0].split('.')[0]).weight.fill_(0.)  # split key to only get 'fc1'

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
#
# def define_alpha(max_scales, ini_alpha=0, factor=1):
#     if ini_alpha:
#         alpha = torch.eye(max_scales)[ini_alpha-1] * factor
#     else:
#         alpha = torch.ones(max_scales) * factor
#     return alpha
#
# initial_alpha = 0
# # alpha = nn.Parameter(define_alpha(4, ini_alpha=initial_alpha, factor=1))
# # # alpha = define_alpha(4, ini_alpha=initial_alpha, factor=1)
# # alpha = alpha.type(torch.FloatTensor)
# # print(alpha.type())
# # print('alpha', alpha)
# # alpha_dropout = F.dropout(alpha, p=0.5)
# # alpha_dropout[alpha_dropout == 0] = float('NaN')
# # print('alpha_dropout', alpha_dropout)
# # nalpha = F.softmax(alpha, 0).view(-1, 1, 1, 1, 1)
# # nalpha_dropout = F.softmax(alpha_dropout, 0).view(-1, 1, 1, 1, 1)
# # # print(F.softmax(torch.Tensor([1,0,0,0,0])), 0)
# # print('nalpha', nalpha)
# # print('nalpha dropout', nalpha_dropout)
#
#
# ### MANUAL implemetation
# # p = 0.5
# # xx = F.feature_alpha_dropout(alpha, p=0.5)
# # print(xx)
#
#
# # i = [[0, 2], [1, 0], [1, 2]]
# # v =  [3,      4,      5    ]
# #
# # s = torch.sparse_coo_tensor(list(zip(*i)), v, (2, 3))
# #
# # i = np.array(i)
# # transpose = i.T
# #
# # i = transpose.tolist()
# # y = torch.sparse_coo_tensor(i, v, torch.Size([2,3])).to_dense()
# # print(y)
#
#
#
# i = [[0, 2], [0, 0], [0, 4]]
# v =  [1,      1,      1    ]
# v = nn.Parameter(torch.FloatTensor(v))
# s = torch.sparse_coo_tensor(list(zip(*i)), v, (1, 5))
# i = np.array(i)
# transpose = i.T
#
# i = transpose.tolist()
# y = torch.sparse_coo_tensor(i, v, torch.Size([1,5])).to_dense()
# # s = s.cuda()
# print(s)
# print(y)
# print('values',s._values())
# # torch.sparse_coo_tensor(i.t(), v, torch.Size([2,3])).to_dense()
# print(s.is_coalesced())
# ss = F.softmax(s._values(), -1)#.view(-1, 1, 1, 1, 1)
# print(ss)
# y = torch.sparse_coo_tensor(i, ss, torch.Size([1,5])).to_dense()
# print(y)
# xx = [j[1] for j in i]
# print(xx)