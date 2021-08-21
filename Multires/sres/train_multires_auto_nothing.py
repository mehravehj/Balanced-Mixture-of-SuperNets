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
from data_loader_auto_nothing import data_loader
from model_auto_sig import multires_model

torch.cuda.empty_cache()

parser = argparse.ArgumentParser(description='PyTorch Mutiresolution Training')
parser.add_argument('--dataset', '-d', default='CIFAR10', type=str, help='dataset name')
parser.add_argument('--channels', '-c', type=str, default='16', help='number of channels')
parser.add_argument('--leng', '-l', type=int, default=6, help='depth of network')
parser.add_argument('--batchsize', '-b', type=int, default=200, help='batch size')
parser.add_argument('--test_name', '-tn', type=int, default=666, help='test name for saving model')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--epochs', '-e', type=int, default=100, help='epochs to train')
parser.add_argument('--validation_percent', '-vp', type=float, default=0.5, help='percent of train data for validation')
parser.add_argument('--learning_rate', '-lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--min_learning_rate', '-mlr', default=0.001, type=float, help='min learning rate')
parser.add_argument('--weight_momentum', '-wm', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', '-wd', default=0, type=float, help='weight decay')
parser.add_argument('-alr', default=0.01, type=float, help=' alpha learning rate')
parser.add_argument('-am', default=0.9, type=float, help='alpha momentum')
parser.add_argument('-awd', default=0, type=float, help='weight decay')
parser.add_argument('--alpha_train_start', '-ats', type=int, default=0, help='epochs to start training alpha')
parser.add_argument('--max_scales', '-mx', type=int, default=4, help='number of scales to use')
parser.add_argument('--data_dir', '-dd', type=str, default='./data/', help='dataset directory')
parser.add_argument('--workers', '-wr', type=int, default=0, help='number of workers to load data')
args = parser.parse_args()


def main():
    startTime = datetime.now()
    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    print('Test ', args.test_name)
    print('----------------')
    print('Test Parameters')
    print(args)
    print('----------------')
    net = multires_model(channels=args.channels, leng=args.leng, max_scales=args.max_scales)
    net.cuda()
    net = torch.nn.DataParallel(net)###
    #cudnn.benchmark = True###
    print('NETWORK')
    print(net)
    print('----------------')
    criterion = nn.MSELoss() #loss
    criterion = criterion.cuda()

    weight_parameters, alpha_parameters = parameters(net)
    weight_optimizer = optim.SGD(weight_parameters, lr=args.learning_rate, momentum=args.weight_momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(weight_optimizer, args.epochs, eta_min=args.min_learning_rate)
    alpha_optimizer = optim.Adam(alpha_parameters, lr=args.alr, weight_decay=args.awd)

    save_dir = './checkpoint/ckpt_' + str(args.test_name) + '.t7'
    if path.exists(save_dir):
        best_model, epoch, loss_progress, alpha_progress, best_alpha, best_epoch, best_loss, index = load_checkpoint(save_dir, net, weight_optimizer, scheduler, alpha_optimizer)
    else:
        best_model, epoch, loss_progress, alpha_progress, best_alpha, best_epoch, best_loss, index = initilialize_save_parameters(net)

    #load data
    if path.exists(args.data_dir):
        dataset_dir = args.data_dir
    else:
        dataset_dir = '~/Desktop/codes/multires/data/'
    train_loader, validation_loader, test_loader, indices = data_loader(args.dataset, args.validation_percent, args.batchsize,
                                                                                   indices=index,
                                                                                   dataset_dir=dataset_dir,
                                                                                   workers=args.workers)

    for epoch in range(1):
        print('epoch ', epoch)
        print('net learning rate: ', weight_optimizer.param_groups[0]['lr'])
        # train_loss, validation_loss = train_valid(train_loader, validation_loader, net, weight_optimizer, alpha_optimizer, criterion, epoch)
        scheduler.step()
        if epoch % 5 == 0 or epoch == 0:
            print('Testing...')
            test_loss, test_accuracy = test(test_loader, net)
            alpha_progress.append(get_alpha(net))

            if test_loss < best_loss:
                print('-----------> Best Loss')
                best_model = {}
                for key in net.state_dict():
                    best_model[key] = net.state_dict()[key].clone().detach()
                best_loss = test_loss
                best_epoch = epoch
                best_alpha = get_alpha(net)
            print('train loss: ', train_loss, ' ....... validation loss: ', validation_loss, ' ....... test accuracy: ', test_loss)
            print('alpha')
            print(get_alpha(net))
            print('best loss:', best_loss,' at epoch ', best_epoch)
            print('.....SAVING.....')
            # save_checkpoint(save_dir, net, best_model, weight_optimizer, scheduler, alpha_optimizer, epoch, loss_progress,
            #                 alpha_progress, best_alpha, best_epoch, best_loss, index)
            print('Training time: ', datetime.now() - startTime)


def load_checkpoint(save_dir, model, weight_optimizer, scheduler, alpha_optimizer):
    print('Loading from checkpoint...')
    checkpoint = torch.load(save_dir)
    epoch = checkpoint['epoch']
    loss_progress = checkpoint['loss_progress']
    alpha_progress = checkpoint['alpha_progress']
    best_model = checkpoint['best_model']
    best_alpha = checkpoint['best_alpha']
    best_epoch = checkpoint['best_epoch']
    best_loss = checkpoint['best_loss']
    index = checkpoint['indices']

    model.load_state_dict(checkpoint['model'])
    weight_optimizer.load_state_dict(checkpoint['weight_optimizer'])
    try:
        scheduler.load_state_dict(checkpoint['scheduler_state'])
    except:
        pass

    alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])

    return best_model, epoch, loss_progress, alpha_progress, best_alpha, best_epoch, best_loss, index

def initilialize_save_parameters(model):
    epoch = 0
    best_epoch = 0
    best_model = 0
    best_loss = 100
    loss_progress = {'train': [], 'validation': [], 'test': []}
    alpha_progress = [get_alpha( model)]
    best_alpha = get_alpha(model)
    index = 0
    return best_model, epoch, loss_progress,  alpha_progress, best_alpha, best_epoch, best_loss, index

def save_checkpoint(save_dir, model, best_model, weight_optimizer, scheduler, alpha_optimizer, epoch, loss_progress, alpha_progress, best_alpha, best_epoch, best_loss, index):
    state = {
        'test_properties': vars(args),
        'seed': args.seed,
        'indices': index,
        'best_epoch': best_epoch,
        'best_model': best_model,
        'loss_progress': loss_progress,
        'alpha_progress': alpha_progress,
        'best_alpha': best_alpha,
        'best_loss': best_loss,
        'model': model.state_dict(),
        'epoch': epoch,
        'weight_optimizer': weight_optimizer.state_dict(),
        'alpha_optimizer': alpha_optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
    }
    if not os.path.isdir('checkpoint_autoencoder'):
        os.mkdir('checkpoint_autoencoder')
    torch.save(state, save_dir)


def get_alpha( model):
    alpha = []
    for name, param in model.named_parameters():
        if param.requires_grad and 'alpha' in name:
            alpha.append(param.cpu().detach())
    alpha = (torch.stack(alpha, 0)).numpy()
    return alpha


def parameters(model):
    all_parameter_names = [x for x, y in model.named_parameters()]
    weight_param_idx = [idx for idx, x in enumerate(all_parameter_names) if x.find('alpha') is -1]
    alpha_param_idx = [idx for idx, x in enumerate(all_parameter_names) if x.find('alpha') is not -1]
    all_parameters = [y for x, y in model.named_parameters()]
    weight_parameters = [all_parameters[idx] for idx in weight_param_idx]
    alpha_parameters = [all_parameters[idx] for idx in alpha_param_idx]
    return weight_parameters, alpha_parameters


def train_valid(train_queue, validation_queue, model, weight_optimizer, alpha_optimizer, criterion=nn.MSELoss(), epoch=0):
    model.train()
    train_loss = 0
    validation_loss = 0
    validation_iterator = iter(validation_queue)
    for batch_idx, (train_inputs, train_targets) in enumerate(train_queue):
        train_inputs, train_targets = train_inputs.cuda(), train_targets.cuda()
        train_outputs = model(train_inputs)
        train_minibatch_loss = criterion(train_outputs, train_inputs)
        train_loss += train_minibatch_loss.detach().cpu().item()
    train_loss = train_loss / (batch_idx + 1)
    validation_loss = validation_loss / (batch_idx + 1)
    feature_maps = [train_inputs.detach().data, train_outputs.detach().data]
    save_dir = './weights/8404/inp_out.t7'
    torch.save(feature_maps, save_dir)

    return train_loss, validation_loss

def test(test_queue, model, criterion=nn.MSELoss()):
    model.eval()
    test_loss = 0
    test_accuracy = 0
    with torch.no_grad():
        for batch_idx, (test_inputs, test_targets) in enumerate(test_queue):
            test_inputs, test_targets = test_inputs.cuda(), test_targets.cuda()
            test_outputs = model(test_inputs)
            test_minibatch_loss = criterion(test_outputs, test_inputs)
            test_loss += test_minibatch_loss.detach().cpu().item()
    test_loss = test_loss / (batch_idx + 1)
    feature_maps = [test_inputs.detach().data, test_outputs.detach().data]
    save_dir = './weights/8404/inp_out.t7'
    torch.save(feature_maps, save_dir)
    return test_loss, test_accuracy


if __name__ == '__main__':
  main()
