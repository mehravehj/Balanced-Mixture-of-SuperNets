from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
# from multiscale_blocks import *
from data_loader import data_loader
import torch.optim as optim


class sRes2d(nn.Module):
    def __init__(self, in_, out_, kernel_size, max_scales=4, usf=True):
        super(sRes2d, self).__init__()
        self.in_ = in_
        self.out_ = out_
        self.conv1 = nn.Conv2d(in_, out_, kernel_size, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(out_, affine=False)
        self.conv2 = nn.Conv2d(out_, 10, kernel_size, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(10, affine=False)

    def forward(self, x):
        # self.nalpha = F.softmax(self.alpha, 0).view(-1, 1)
        # sel = F.softmax(self.alpha, 0).view(-1, 1)
        lx = x
        y = self.conv1(lx)
        y = self.relu(y)
        y = self.bn1(y)
        y = self.conv2(y)
        y = self.relu(y)
        out = self.bn2(y)
        print(self.bn2.running_mean)
        out = F.avg_pool2d(out, kernel_size=out.size()[2:])
        out = out.view(out.size(0), -1)
        return out

dataset_dir = '~/Desktop/codes/multires/data/'
train_loader, validation_loader, test_loader, indices, num_class = data_loader('CIFAR10', 0.5, 256,0,0, dataset_dir, 4)
model = sRes2d(3, 8, 3)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
model.cuda()
criterion.cuda()

model.train()
train_loss = 0
validation_loss = 0
for batch_idx, (train_inputs, train_targets) in enumerate(train_loader):
    train_inputs, train_targets = train_inputs.cuda(), train_targets.cuda()
    optimizer.zero_grad()
    train_outputs = model(train_inputs)
    train_minibatch_loss = criterion(train_outputs, train_targets)
    train_minibatch_loss.backward()
    optimizer.step()