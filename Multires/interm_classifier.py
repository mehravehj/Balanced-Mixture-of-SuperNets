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
# from data_loader import data_loader
# from model_drop import multires_model

parser = argparse.ArgumentParser(description='PyTorch Mutiresolution Training')
parser.add_argument('--layer_classifier', '-lc', type=int, default='6', help='layers to add classier to')
parser.add_argument('--test_name', '-tn', type=int, default=666, help='test name for saving model')
parser.add_argument('--load_name', '-ln', type=int, default=7011, help='load test name')
parser.add_argument('--workers', '-wr', type=int, default=0, help='number of workers to load data')
args = parser.parse_args()

# def load_checkpoint(save_dir, model, weight_optimizer, scheduler, alpha_optimizer):
#     epoch = 0
#     index = 0
#     best_epoch = 0
#     best_model = 0
#     best_accuracy = 0
#     loss_progress = {'train': [], 'validation': [], 'test': []}
#     accuracy_progress = {'train': [], 'validation': [], 'test': []}
#     alpha_progress = [get_alpha(args.net_type, model)]
#     best_alpha = get_alpha(args.net_type, model)
#
#     if path.exists(save_dir):
#         print('Loading from checkpoint...')
#         checkpoint = torch.load(save_dir)
#         epoch = checkpoint['epoch']
#         loss_progress = checkpoint['loss_progress']
#         accuracy_progress = checkpoint['accuracy_progress']
#         alpha_progress = checkpoint['alpha_progress']
#         best_model = checkpoint['best_model']
#         best_alpha = checkpoint['best_alpha']
#         best_epoch = checkpoint['best_epoch']
#         best_accuracy = checkpoint['best_accuracy']
#         index = checkpoint['indices']
#
#         model.load_state_dict(checkpoint['model'])
#         weight_optimizer.load_state_dict(checkpoint['weight_optimizer'])
#         scheduler.load_state_dict(checkpoint['scheduler_state'])
#         if args.net_type == 'multires':
#             alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
#
#     return best_model, epoch, loss_progress, accuracy_progress, alpha_progress, best_alpha, best_epoch, best_accuracy, index



save_dir = './checkpoint/ckpt_' + str(args.test_name) + '.t7'
load_dir = './checkpoint/ckpt_' + str(args.load_name) + '.t7'

print('Loading test properties from trained model ' + str(args.load_name)+'...')
test_properties = torch.load(load_dir)['test_properties']
print(test_properties)
mscale = test_properties['mscale']
alpha_train_start = test_properties['alpha_train_start']
seed = test_properties['seed']
net_type = test_properties['net_type']
pooling = test_properties['pooling']
factor = test_properties['factor']
length = test_properties['leng']
dataset = test_properties['dataset']
max_scales = test_properties['max_scales']
validation_percent = test_properties['validation_percent']
number_train = test_properties['number_train']
channels = test_properties['channels']

# Build model


if not os.path.isdir('save_dir'):
else:



#     best_model, epoch, loss_progress, accuracy_progress, alpha_progress, best_alpha, best_epoch, best_accuracy, index = load_checkpoint(
#         save_dir, net, weight_optimizer, scheduler, alpha_optimizer)
# best_model, epoch, _, _, _, best_alpha, _, _, index = load_checkpoint(save_dir, net, weight_optimizer, scheduler, alpha_optimizer)
#
#     if path.exists(args.data_dir):
#         dataset_dir = args.data_dir
#     else:
#         dataset_dir = '~/Desktop/codes/multires/data/'


