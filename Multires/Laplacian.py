from __future__ import print_function

import argparse
import os
from datetime import datetime
from os import path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description='PyTorch Laplacian net Training')
parser.add_argument('--model_type', '-mt', default='normal', type=str, help='model type')
parser.add_argument('--learning_rate', '-lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--min_learning_rate', '-mlr', default=0.000001, type=float, help='min learning rate')
parser.add_argument('--weight_decay', '-wd', default=0, type=float, help='weight decay')
parser.add_argument('--test_name', '-tn', type=int, default=666, help='test name for saving model')
parser.add_argument('--data_dir', '-dd', type=str, default='./data/', help='dataset directory')
args = parser.parse_args()

def laplacian():
    kernel = [[0, -1, 0],
              [-1, 4, -1],
              [0, -1, 0]]
    kernel = torch.tensor(kernel)
    kernel = kernel.float()
    # print(kernel.type())
    return kernel.cuda()

class conv_block_normal(nn.Module): #standard net with conv-relu-bn
    def __init__(self, in_, out_, kernel_size=3):
        super(conv_block_normal, self).__init__()
        self.conv = nn.Conv2d(in_, out_, kernel_size, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_, affine=False)

    def forward(self, x):
        y = self.conv(x)
        y = self.relu(y)
        out = self.bn(y)
        return out

class conv_block_laplacian(nn.Module): #conv-laplacian-bn
    def __init__(self, in_, out_, kernel_size=3):
        super(conv_block_laplacian, self).__init__()
        self.conv = nn.Conv2d(in_, out_, kernel_size, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_)
        self.kernel = laplacian().view(1, 1, 3, 3).repeat(out_, 1, 1, 1)
        # self.kernel.type(torch.FloatTensor)
        # self.kernel.double()
        self.out_ = out_

    def forward(self, x):
        y = self.conv(x)
        # print(y.type())
        # print(self.kernel.type())
        y = F.conv2d(y, self.kernel, stride=1, padding=1, groups=self.out_)
        # print((y[0,0,0:2,:2]))
        out = self.bn(y)
        return out

class conv_block_laplacian_channel(nn.Module): #conv-laplacian-bn
    def __init__(self, in_, out_, kernel_size=3):
        super(conv_block_laplacian_channel, self).__init__()
        self.conv = nn.Conv2d(in_, out_, kernel_size, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_)
        self.relu = nn.ReLU()
        self.kernel = laplacian().view(1, 1, 3, 3).repeat(out_, 1, 1, 1)
        self.out_ = out_

    def forward(self, x):
        y = self.conv(x)
        lap = F.conv2d(y, self.kernel, stride=1, padding=1, groups=self.out_)
        lap_norm = torch.norm(lap, dim=1).unsqueeze(1)
        # print((lap_norm[0, 0, 0:2, :2]))
        y = self.relu(y)
        y = self.bn(y)
        out = torch.cat((y,lap_norm), 1)
        return out

class conv_block_laplacian_sum(nn.Module): #conv-laplacian-bn
    def __init__(self, in_, out_, kernel_size=3):
        super(conv_block_laplacian_sum, self).__init__()
        self.conv = nn.Conv2d(in_, out_, kernel_size, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_)
        self.relu = nn.ReLU()
        self.kernel = laplacian().view(1, 1, 3, 3).repeat(out_, 1, 1, 1)
        self.out_ = out_

    def forward(self, x):
        y = self.conv(x)
        lap = F.conv2d(y, self.kernel, stride=1, padding=1, groups=self.out_)
        lap_sum = torch.sum(lap, dim=1).unsqueeze(1)
        # print(abs(lap_sum[0,0,0:2,:2]))
        # print(lap_sum.size())
        y = self.relu(y)
        y = self.bn(y)
        out = torch.cat((y,lap_sum), 1)
        # print('lap', lap_sum[0,0,:2,:2])
        # print('y', y[0,0,:2,:2])
        return out

class conv_block_laplacian_relu(nn.Module): #conv-laplacian-relu-bn
    def __init__(self, in_, out_, kernel_size=3):
        super(conv_block_laplacian_relu, self).__init__()
        self.conv = nn.Conv2d(in_, out_, kernel_size, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_)
        self.kernel = laplacian().view(1, 1, 3, 3).repeat(out_, 1, 1, 1)
        # self.kernel.type(torch.FloatTensor)
        # self.kernel.double()
        self.relu = nn.ReLU()
        self.out_ = out_

    def forward(self, x):
        y = self.conv(x)
        # print(y.type())
        # print(self.kernel.type())
        y = F.conv2d(y, self.kernel, stride=1, padding=1, groups=self.out_)
        lap_sum = torch.sum(y, dim=1)
        # print(abs(lap_sum[0,0:2,:2]))
        y = self.relu(y)
        out = self.bn(y)
        return out

class conv_block_laplacian_weight(nn.Module): #conv-laplacian-relu-bn
    def __init__(self, in_, out_, kernel_size=3):
        super(conv_block_laplacian_weight, self).__init__()
        self.conv = nn.Conv2d(in_, out_, kernel_size, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_)
        self.kernel = laplacian().view(1, 1, 3, 3).repeat(out_, 1, 1, 1)
        # self.kernel.type(torch.FloatTensor)
        # self.kernel.double()
        self.relu = nn.ReLU()
        self.out_ = out_

    def forward(self, x):
        y = self.conv(x)
        # print(y.type())
        # print(self.kernel.type())
        lap = F.conv2d(y, self.kernel, stride=1, padding=1, groups=self.out_)
        print(torch.max(lap))
        lap_= lap / torch.max(abs(lap))
        lap_sum = torch.sum(lap, dim=1).unsqueeze(1)
        lap_sum = lap_sum / torch.max(lap_sum)
        # print(abs(lap_sum[0, 0, 0:2, :2]))
        y = y * lap_sum
        y = self.relu(y)
        out = self.bn(y)
        return out

class layer(nn.Module): #conv-laplacian-bn
    def __init__(self, in_, out_, layer_type):
        super(layer, self).__init__()
        if layer_type == 'normal':
            self.block = conv_block_normal(in_, out_)
        if layer_type == 'laplacian':
            self.block = conv_block_laplacian(in_, out_)
        if layer_type == 'laplacian_relu':
            self.block = conv_block_laplacian_relu(in_, out_)
        if layer_type == 'laplacian_mag':
            self.block = conv_block_laplacian_channel(in_+1, out_)
        if layer_type == 'laplacian_sum':
            self.block = conv_block_laplacian_sum(in_+1, out_)
        if layer_type == 'laplacian_weight':
            self.block = conv_block_laplacian_weight(in_, out_)

    def forward(self, x):
        out = self.block(x)
        return out

def data_loader(dataset_dir):
    cifar_mean = [0.49139968, 0.48215827, 0.44653124]
    cifar_std = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])
    trainset = torchvision.datasets.CIFAR10(root=dataset_dir, train=True,
                                            download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                              shuffle=True, num_workers=6)

    testset = torchvision.datasets.CIFAR10(root=dataset_dir, train=False,
                                           download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                             shuffle=False, num_workers=6)

    return trainloader, testloader


class model_laplacian(nn.Module):
    def __init__(self, model_type):
        super(model_laplacian, self).__init__()
        self.leng = 8
        self.channels = 32
        if model_type in {'laplacian_mag', 'laplacian_sum'}:
            list_layer = [layer(2, self.channels, model_type)]
        else:
            list_layer = [layer(3, self.channels, model_type)]
        list_layer += [layer(self.channels, self.channels, model_type)]
        list_layer += [layer(self.channels, self.channels*2, model_type)]
        list_layer += [layer(self.channels*2, self.channels * 2, model_type)]
        list_layer += [layer(self.channels*2, self.channels * 4, model_type)]
        list_layer += [layer(self.channels*4, self.channels * 4, model_type)]
        list_layer += [layer(self.channels*4, self.channels * 8, model_type)]
        list_layer += [layer(self.channels*8, 10, model_type)]
        self.net_layer = nn.ModuleList(list_layer)

    def forward(self, x):
        out = x
        for i in range(self.leng):
            out = self.net_layer[i](out)
            if i in [1,3,5]:
                out = F.max_pool2d(out, kernel_size=2)
        out = F.avg_pool2d(out, kernel_size=out.size()[2:])
        out = out.view(out.size(0), -1)
        return out

def calculate_accuracy(logits, target, cul_total=0, cul_prediction=0):
    _, test_predicted = logits.max(1)
    test_total = target.size(0)
    correct_prediction = test_predicted.eq(target).sum().item()
    cul_prediction += correct_prediction
    cul_total += test_total
    return cul_prediction, cul_total

def save_checkpoint(save_dir, model, best_model, weight_optimizer, scheduler, epoch, loss_progress, accuracy_progress, best_epoch, best_accuracy):
    state = {
        'test_properties': vars(args),
        'best_net': best_model,
        'best_epoch': best_epoch,
        'best_model': best_model,
        'loss_progress': loss_progress,
        'accuracy_progress': accuracy_progress,
        'best_accuracy': best_accuracy,
        'model': model.state_dict(),
        'epoch': epoch,
        'weight_optimizer': weight_optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
    }
    if not os.path.isdir('checkpoint_laplacian'):
        os.mkdir('checkpoint_laplacian')
    torch.save(state, save_dir)

def load_checkpoint(save_dir, model, weight_optimizer, scheduler):
    epoch = 0
    index = 0
    best_epoch = 0
    best_model = 0
    best_accuracy = 0
    loss_progress = {'train': [], 'test': []}
    accuracy_progress = {'train': [], 'test': []}

    if path.exists(save_dir):
        print('Loading from checkpoint...')
        checkpoint = torch.load(save_dir)
        epoch = checkpoint['epoch']
        loss_progress = checkpoint['loss_progress']
        accuracy_progress = checkpoint['accuracy_progress']
        best_model = checkpoint['best_model']
        best_epoch = checkpoint['best_epoch']
        best_accuracy = checkpoint['best_accuracy']

        model.load_state_dict(checkpoint['model'])
        weight_optimizer.load_state_dict(checkpoint['weight_optimizer'])
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state'])
        except:
            pass

    return best_model, epoch, loss_progress, accuracy_progress, best_epoch, best_accuracy

def main():
    startTime = datetime.now()
    print('Test ', args.test_name)
    print(args)
    save_dir = './checkpoint_laplacian/lap_' + str(args.test_name) + '.t7'
    model = model_laplacian(args.model_type)
    model.cuda()
    model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2,
                                                                     eta_min=args.min_learning_rate)

    if path.exists(args.data_dir):
        dataset_dir = args.data_dir
    else:
        dataset_dir = '~/Desktop/codes/multires/data/'
    train_queue, test_queue = data_loader(dataset_dir)

    loss_progress = {'train': [], 'validation': [], 'test': []}
    accuracy_progress = {'train': [], 'validation': [], 'test': []}
    best_accuracy = 0
    best_model, start_epoch, loss_progress, accuracy_progress, best_epoch, best_accuracy = load_checkpoint(save_dir, model, optimizer, scheduler)

    for epoch in range(start_epoch, 3000):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        print('epoch ', epoch)
        print('net learning rate: ', optimizer.param_groups[0]['lr'])
        for batch_idx, (train_inputs, train_targets) in enumerate(train_queue):
            train_inputs, train_targets = train_inputs.cuda(), train_targets.cuda()
            optimizer.zero_grad()
            train_outputs = model(train_inputs)
            train_minibatch_loss = criterion(train_outputs, train_targets)
            train_minibatch_loss.backward()
            optimizer.step()
            train_loss += train_minibatch_loss.cpu().item()
            train_correct, train_total = calculate_accuracy(train_outputs, train_targets, train_total, train_correct)
        print('training: ', train_correct, ' / ', train_total)
        if epoch % 5 == 0:
            train_accuracy = train_correct / train_total
            model.eval()
            test_loss = 0
            test_correct = 0
            test_total = 0
            with torch.no_grad():
                for batch_idx, (test_inputs, test_targets) in enumerate(test_queue):
                    test_inputs, test_targets = test_inputs.cuda(), test_targets.cuda()
                    test_outputs = model(test_inputs)
                    test_minibatch_loss = criterion(test_outputs, test_targets)
                    test_loss += test_minibatch_loss.cpu().item()
                    test_correct, test_total = calculate_accuracy(test_outputs, test_targets, test_total, test_correct)
            test_accuracy = test_correct / test_total
            if test_accuracy > best_accuracy:
                print('-----------> Best accuracy')
                best_model = model.state_dict()
                best_accuracy = test_accuracy
                best_epoch = epoch
            loss_progress['train'].append(train_loss)
            loss_progress['test'].append(test_loss)
            accuracy_progress['train'].append(train_accuracy)
            accuracy_progress['test'].append(test_accuracy)
            print('train accuracy: ', train_accuracy, ' ....... test accuracy: ', test_accuracy)
            print('best accuracy:', best_accuracy, ' at epoch ', best_epoch)

            print('.....SAVING.....')
            save_checkpoint(save_dir, model, best_model, optimizer, scheduler, epoch, loss_progress,
                            accuracy_progress, best_epoch, best_accuracy)
            print('Training time: ', datetime.now() - startTime)
        scheduler.step()

if __name__ == '__main__':
  main()