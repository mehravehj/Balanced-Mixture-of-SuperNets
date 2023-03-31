import argparse
import copy
import os
from datetime import datetime
from os import path
import torch
import torch.nn as nn
import torch.utils.data

from baseline.base_trainer import create_models, create_optimizers, train, test
from baseline.base_data_loader import data_loader
from utils.utility_functions import string_to_list

parser = argparse.ArgumentParser(description='PyTorch Resnet baseline training')
parser.add_argument('--dataset', '-d', default='CIFAR10', type=str, help='dataset name')
parser.add_argument('--channels_string', '-c', type=str, default='16,16,16,16,32,32,32,64,64,10', help='number of channels per layer')
parser.add_argument('--leng', '-l', type=int, default=10, help='depth of network')
parser.add_argument('--batchsize', '-b', type=int, default=128, help='batch size')
parser.add_argument('--test_name', '-tn', type=int, default=50, help='test name for saving model')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--epochs', '-e', type=int, default=1000, help='epochs to train')

parser.add_argument('--learning_rate', '-lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--min_learning_rate', '-mlr', default=0.001, type=float, help='min learning rate')
parser.add_argument('--weight_momentum', '-wm', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', '-wd', default=0, type=float, help='weight decay')

parser.add_argument('--pooling', '-p', type=str, default='0,0,0,0,1,0,0,1,0,0', help='path of the model')
parser.add_argument('--data_dir', '-dd', type=str, default='./data/', help='dataset directory')
parser.add_argument('--workers', '-wr', type=int, default=0, help='number of workers to load data')

args = parser.parse_args()

def main():
    print('Test: ', args.test_name)
    print('Baseline Path: ', args.pooling)
    print('--------------')
    print(args)
    print('--------------')
    startTime = datetime.now()
    epochs = args.epochs
    num_layers = args.leng
    pooling = tuple(string_to_list(args.pooling, num_layers))
    print(pooling)
    channels = string_to_list(args.channels_string, num_layers)
    lr = args.learning_rate
    mlr = args.min_learning_rate
    moment = args.weight_momentum
    w_decay = args.weight_decay

    save_dir = './checkpoint/res_base_chpt_' + str(args.test_name) + '.t7'  # checkpoint save directory

    # create network
    if args.dataset == 'CIFAR10':
        ncat = 10
    print('creating model....')
    net = create_models(num_layers, channels)
    net.cuda()
    ### set path
    net.set_path(pooling)
    print(net)
    optimizer, scheduler = create_optimizers(net, lr, moment, w_decay, epochs, mlr)

    criterion = nn.CrossEntropyLoss()  # classification loss criterion
    criterion = criterion.cuda()

    current_epoch = 0

    ###loading data
    if path.exists(args.data_dir):
        dataset_dir = args.data_dir
    else:
        dataset_dir = '~/Desktop/codes/multires/data/'

    train_loader, test_loader, num_class = data_loader(args.dataset, args.batchsize, dataset_dir=dataset_dir, workers=args.workers)

    t_loss = []
    t_acc = []
    v_loss = []
    v_acc = []

    for epoch in range(current_epoch, args.epochs + 1):
        print('epoch ', epoch)
        print('net learning rate: ', optimizer.param_groups[0]['lr'])
        # train
        train_loss, t_accuracy= train(net, train_loader, optimizer, criterion=criterion)
        t_loss.append(train_loss)
        t_acc.append(t_accuracy[0]/t_accuracy[1])
        print('train  acc: ', t_accuracy[0]/t_accuracy[1], 'loss: ', train_loss)

        scheduler.step()
        print('Training time: ', datetime.now() - startTime)

        if epoch != 0 and (epoch % 2 == 0 or epoch == args.epochs):  # test and save checkpoint every 5 epochs
            print('Saving models and progress...')
            if epoch != 0 and (epoch % 2 == 0 or epoch == args.epochs):
                print('testing...')
                valid_loss, valid_accuracy = test(net, test_loader, criterion=criterion)
                v_loss.append(valid_loss)
                v_acc.append(valid_accuracy[0] / valid_accuracy[1])
                print('test  acc', valid_accuracy[0] / valid_accuracy[1], 'loss: ', valid_loss)

            save_checkpoint(save_dir, net, optimizer, scheduler, epoch ,t_acc, v_acc, t_loss, v_loss, pooling)

def save_checkpoint(save_dir, models, optimizers, schedulers, epoch, t_acc, v_acc, t_loss, v_loss, pool):
    state = {
        'test_properties': vars(args),
        'seed': args.seed,
        't_loss': t_loss,
        't_acc': t_acc,
        'v_loss': v_loss,
        'v_acc': v_acc,
        'model': models.state_dict(),
        'epoch': epoch,
        'weight_optimizer': optimizers.state_dict(),
        'scheduler_state': schedulers.state_dict(),
        'pool': pool,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, save_dir)


if __name__ == '__main__':
  main()
