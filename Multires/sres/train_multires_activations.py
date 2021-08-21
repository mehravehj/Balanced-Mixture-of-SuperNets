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
from model_with_activations import multires_model

parser = argparse.ArgumentParser(description='PyTorch Mutiresolution Training')
parser.add_argument('--dataset', '-d', default='CIFAR10', type=str, help='dataset name')
parser.add_argument('--channels', '-c', type=str, default='32', help='number of channels')
parser.add_argument('--leng', '-l', type=int, default=6, help='depth of network')
parser.add_argument('--batchsize', '-b', type=int, default=200, help='batch size')
parser.add_argument('--test_name', '-tn', type=int, default=666, help='test name for saving model')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--epochs', '-e', type=int, default=100, help='number of epochs to train')

parser.add_argument('--validation_percent', '-vp', type=float, default=0.5, help='percent of train data for validation')

parser.add_argument('--learning_rate', '-lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--min_learning_rate', '-mlr', default=0.001, type=float, help='min learning rate')
parser.add_argument('--weight_momentum', '-wm', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', '-wd', default=0, type=float, help='weight decay')

parser.add_argument('-alr', default=0.01, type=float, help=' alpha learning rate')
parser.add_argument('-am', default=0.9, type=float, help='alpha momentum')
parser.add_argument('-awd', default=0, type=float, help='weight decay')
parser.add_argument('--alpha_train_start', '-ats', type=int, default=0, help='epoch to start training alpha')

parser.add_argument('--mscale', '-ms', type=str, default='sconv', help='sconv or sres, block type')
parser.add_argument('--max_scales', '-mx', type=int, default=4, help='number of scales to use')

parser.add_argument('--data_dir', '-dd', type=str, default='./data/', help='dataset directory')

parser.add_argument('--workers', '-wr', type=int, default=0, help='number of workers to load data')

parser.add_argument('--usf', '-g', default=False, action='store_true', help='use same filters')
args = parser.parse_args()


def main():
    startTime = datetime.now()
    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)

    print('Test ', args.test_name)
    print(args)
    ncat = 10
    net = multires_model(ncat=ncat, channels=args.channels, leng=args.leng, max_scales=args.max_scales, usf=args.usf)
    net.cuda()
    cudnn.benchmark = True
    print(net)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    weight_parameters, alpha_parameters = parameters(net)

    weight_optimizer = optim.SGD(weight_parameters, lr=args.learning_rate, momentum=args.weight_momentum, weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(weight_optimizer, float(args.epochs), eta_min=args.min_learning_rate)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(weight_optimizer, args.epochs, eta_min=args.min_learning_rate)
    # alpha_optimizer = optim.Adam(alpha_parameters, lr=args.alr, betas=(0.9, 0.999), weight_decay=args.awd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(weight_optimizer, mode='min', factor=0.5, patience=80, cooldown=20, min_lr=args.min_learning_rate, verbose=True)
    alpha_optimizer = optim.Adam(alpha_parameters, lr=args.alr, weight_decay=args.awd)

    save_dir = './checkpoint/ckpt_' + str(args.test_name) + '.t7'
    net, weight_optimizer, scheduler, alpha_optimizer, best_model, epoch, loss_progress, accuracy_progress, \
    alpha_progress, best_alpha, best_epoch, best_accuracy, index = load_checkpoint(save_dir, net, weight_optimizer, scheduler, alpha_optimizer)

    if path.exists(args.data_dir):
        dataset_dir = args.data_dir
    else:
        dataset_dir = '~/Desktop/codes/multires/data/'

    ###loading data
    train_loader, validation_loader, test_loader, indices, num_class = data_loader(args.dataset, args.validation_percent, args.batchsize,
                                                                                   indices=index,
                                                                                   dataset_dir=dataset_dir,
                                                                                   workers=args.workers)



    train_loss, validation_loss, train_accuracy, validation_accuracy = train_valid(train_loader, validation_loader, net, weight_optimizer, alpha_optimizer, criterion, epoch)


    print('Testing...')
    print(get_alpha(net))
    print(best_alpha)
    test_loss, test_accuracy = test(test_loader, net)


    print('Training time: ', datetime.now() - startTime)


def load_checkpoint(save_dir, model, weight_optimizer, scheduler, alpha_optimizer):
    epoch = 0
    index = 0
    best_epoch = 0
    best_model = 0
    best_accuracy = 0
    loss_progress = {'train': [], 'validation': [], 'test': []}
    accuracy_progress = {'train': [], 'validation': [], 'test': []}
    alpha_progress = [get_alpha( model)]
    best_alpha = get_alpha(model)

    if path.exists(save_dir):
        print('Loading from checkpoint...')
        checkpoint = torch.load(save_dir)
        epoch = checkpoint['epoch']
        loss_progress = checkpoint['loss_progress']
        accuracy_progress = checkpoint['accuracy_progress']
        alpha_progress = checkpoint['alpha_progress']
        best_model = checkpoint['best_model']
        best_alpha = checkpoint['best_alpha']
        best_epoch = checkpoint['best_epoch']
        best_accuracy = checkpoint['best_accuracy']
        index = checkpoint['indices']

        model.load_state_dict(checkpoint['best_model'])
        weight_optimizer.load_state_dict(checkpoint['weight_optimizer'])
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state'])
        except:
            pass

        alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])

    return model, weight_optimizer, scheduler, alpha_optimizer, best_model, epoch, loss_progress, accuracy_progress, alpha_progress, best_alpha, best_epoch, best_accuracy, index


def save_checkpoint(save_dir, model, best_model, weight_optimizer, scheduler, alpha_optimizer, epoch, loss_progress, accuracy_progress, alpha_progress, best_alpha, best_epoch, best_accuracy, index):
    state = {
        'test_properties': vars(args),
        'seed': args.seed,
        'indices': index,
        'best_epoch': best_epoch,
        'best_model': best_model,
        'loss_progress': loss_progress,
        'accuracy_progress': accuracy_progress,
        'alpha_progress': alpha_progress,
        'best_alpha': best_alpha,
        'best_accuracy': best_accuracy,
        'model': model.state_dict(),
        'epoch': epoch,
        'weight_optimizer': weight_optimizer.state_dict(),
        'alpha_optimizer': alpha_optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, save_dir)


def calculate_accuracy(logits, target, cul_total=0, cul_prediction=0):
    _, test_predicted = logits.max(1)
    test_total = target.size(0)
    correct_prediction = test_predicted.eq(target).sum().item()
    cul_prediction += correct_prediction
    cul_total += test_total
    return cul_prediction, cul_total


def get_alpha( model):
    alpha = []
    for name, param in model.named_parameters():
        if param.requires_grad and 'alpha' in name:
            alpha.append(param.detach().cpu())
    alpha = (torch.stack(alpha, 0)).numpy()
    return alpha


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters())/1e6


def parameters(model):
    all_parameter_names = [x for x, y in model.named_parameters()]
    print(all_parameter_names)
    weight_param_idx = [idx for idx, x in enumerate(all_parameter_names) if x.find('alpha') is -1]
    alpha_param_idx = [idx for idx, x in enumerate(all_parameter_names) if x.find('alpha') is not -1]
    all_parameters = [y for x, y in model.named_parameters()]
    weight_parameters = [all_parameters[idx] for idx in weight_param_idx]
    alpha_parameters = [all_parameters[idx] for idx in alpha_param_idx]
    # print('check parameters')
    # print([all_parameter_names[idx] for idx in weight_param_idx])
    # print([i.size() for i in weight_parameters])
    # print([all_parameter_names[idx] for idx in alpha_param_idx])
    # print([i.size() for i in alpha_parameters])
    return weight_parameters, alpha_parameters


def train_valid(train_queue, validation_queue, model, weight_optimizer, alpha_optimizer, criterion=nn.CrossEntropyLoss(), epoch=0):
    model.train()
    train_loss = 0
    validation_loss = 0

    train_correct = 0
    validation_correct = 0

    train_total = 0
    validation_total = 0

    train_accuracy = 0
    validation_accuracy = 0

    return train_loss, validation_loss, train_accuracy, validation_accuracy

def test(test_queue, model, criterion=nn.CrossEntropyLoss()):
    model.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        test_inputs, test_targets = next(iter(test_queue))
        test_inputs, test_targets = test_inputs.cuda(), test_targets.cuda()
        test_outputs = model(test_inputs)
        test_minibatch_loss = criterion(test_outputs, test_targets)
        test_loss += test_minibatch_loss.detach().cpu().item()
        test_correct, test_total = calculate_accuracy(test_outputs, test_targets, test_total, test_correct)
    return test_loss, test_correct/test_total

if __name__ == '__main__':
  main()