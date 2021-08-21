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
import torch.distributions.categorical.Categorical as Categorical
from data_loader import data_loader
from model_TD import path_model
from thop import profile
from thop import clever_format
from ptflops import get_model_complexity_info

parser = argparse.ArgumentParser(description='PyTorch Mutiresolution Training')
parser.add_argument('--dataset', '-d', default='CIFAR10', type=str, help='dataset name')
parser.add_argument('--channels', '-c', type=str, default='16', help='number of channels')
parser.add_argument('--leng', '-l', type=int, default=6, help='depth of network')
parser.add_argument('--batchsize', '-b', type=int, default=200, help='batch size')
parser.add_argument('--test_name', '-tn', type=int, default=666, help='test name for saving model')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--epochs', '-e', type=int, default=100, help='epochs to train')

parser.add_argument('--number_train', '-nb', type=int, default=0, help='number of training examples')

parser.add_argument('--validation_percent', '-vp', type=float, default=0.2, help='percent of train data for validation')

parser.add_argument('--learning_rate', '-lr', default=0.1, type=float, help='learning rate')

parser.add_argument('--weight_momentum', '-wm', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', '-wd', default=0, type=float, help='weight decay')

parser.add_argument('--wma_scheme', '-wma', default='multires', type=str, help='normal or multires')

parser.add_argument('--net_type', '-ty', default='multires', type=str, help='normal or multires')
parser.add_argument('--mscale', '-ms', type=str, default='sconv', help='sconv or sres')
parser.add_argument('--pooling', '-p', type=str, default='0', help='use multiscales')
parser.add_argument('--max_scales', '-mx', type=int, default=4, help='number of scales to use')

parser.add_argument('--data_dir', '-dd', type=str, default='./data/', help='dataset directory')

parser.add_argument('--workers', '-wr', type=int, default=0, help='number of workers to load data')

# optimizer and scheduler parameters
parser.add_argument('--learning_rate', '-lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--weight_momentum', '-wm', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', '-wd', default=0, type=float, help='weight decay')
parser.add_argument('--scheduler_epochs', '-se', type=int, default=50, help='Number of iterations for the first restart')
parser.add_argument('--min_learning_rate', '-mlr', default=0.001, type=float, help='min learning rate')
parser.add_argument('--lr_mult', '-lrmt', default=2, type=int, help='An factor increasing epochs after a restart')

# parser.add_argument('--scheduler', '-sc', default=False, action='store_true', help='resume from checkpoint')

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

    '''
    1. define probability P
    2. create a supernet
    3. sample path from P
    4. choose the path in supernet
    5. train the path 
    6. calculate loss in validation set 
    7. use -loss to update only the M of the resolution in the path
    8. update all the probabilities using softmax(M)
    '''

    if args.dataset == 'TIN':
        ncat = 200
    else:
        ncat = 10

    # # 1. initialize probability as uniform
    # probability_alpha = Categorical(torch.ones(args.leng, args.max_scales)) # uniform probability for (batch, alpha)

    # 2. create model
    net = path_model(ncat=ncat, net_type=args.net_type, mscale=args.mscale, channels=args.channels, leng=args.leng, max_scales=args.max_scales)
    net.cuda()
    net = nn.DataParallel(net)
    print(net)

    # loss criterion defined
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    # do I need weight parameters? might help with not storing disconnected BN
    # optimizer and scheduler
    weight_optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.weight_momentum, weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(weight_optimizer, float(args.epochs), eta_min=args.min_learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(weight_optimizer, T_0=args.scheduler_epochs, T_mult=args.lr_mult,
                                                                 eta_min=args.min_learning_rate)
    # train function
    train_function = train_valid
    # loading models to resume training
    save_dir = './checkpoint_loss/ckpt_loss_' + str(args.test_name) + '.t7'

    epoch, loss_progress, accuracy_progress, best_accuracy, index ,valid_loss, loss_per_res = load_checkpoint(save_dir, net, weight_optimizer, scheduler)

    ###loading data
    if path.exists(args.data_dir):
        dataset_dir = args.data_dir
    else:
        dataset_dir = '~/Desktop/codes/multires/data/'

    train_loader, validation_loader, test_loader, indices, num_class = data_loader(args.dataset, args.validation_percent, args.batchsize, num_train=args.number_train,
                                                                                   indices=index,
                                                                                   dataset_dir=dataset_dir,
                                                                                    workers=args.workers)

    #training
    for epoch in range(epoch+1, args.epochs+1):
        print('epoch ', epoch)
        print('net learning rate: ', weight_optimizer.param_groups[0]['lr'])
        train_loss, validation_loss, train_accuracy, validation_accuracy = train_function(train_loader, validation_loader, net, weight_optimizer, criterion, loss_per_res, valid_loss)
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            print('Testing...')
            test_loss, test_accuracy = test(test_loader, net)

            loss_progress['train'].append(train_loss)
            loss_progress['validation'].append(validation_loss)
            loss_progress['test'].append(test_loss)

            accuracy_progress['train'].append(train_accuracy)
            accuracy_progress['validation'].append(validation_accuracy)
            accuracy_progress['test'].append(test_accuracy)

            if test_accuracy > best_accuracy:
                print('-----------> Best accuracy')
                best_model = {}
                for key in net.state_dict():
                    best_model[key] = net.state_dict()[key].clone()
                # best_model = net.state_dict()
                best_accuracy = test_accuracy
                best_epoch = epoch

            print('train accuracy: ', train_accuracy, ' ....... validation accuracy: ', validation_accuracy, ' ....... test accuracy: ', test_accuracy)
            print('best accuracy:', best_accuracy,' at epoch ', best_epoch)


            save_checkpoint(save_dir, model, weight_optimizer, scheduler, epoch, loss_progress, accuracy_progress, valid_loss, loss_per_res, index)
        print('Training time: ', datetime.now() - startTime)


def load_checkpoint(save_dir, model, weight_optimizer, scheduler):
    epoch = 0
    index = 0
    best_epoch = 0
    best_model = 0
    best_accuracy = 0
    loss_progress = {'train': [], 'validation': [], 'test': []}
    accuracy_progress = {'train': [], 'validation': [], 'test': []}
    loss_per_res = torch.zeros(args.leng, args.max_scales)
    valid_loss = torch.zeros(args.leng, args.max_scales, 31000)


    if path.exists(save_dir):
        print('Loading from checkpoint...')
        checkpoint = torch.load(save_dir)
        epoch = checkpoint['epoch']
        loss_progress = checkpoint['loss_progress']
        accuracy_progress = checkpoint['accuracy_progress']
        alpha_progress = checkpoint['alpha_progress']
        best_accuracy = checkpoint['best_accuracy']
        index = checkpoint['indices']
        valid_loss = checkpoint['valid_loss']
        loss_per_res = checkpoint['loss_per_res']

        model.load_state_dict(checkpoint['model'])
        weight_optimizer.load_state_dict(checkpoint['weight_optimizer'])
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state'])
        except:
            pass

    return epoch, loss_progress, accuracy_progress, best_accuracy, index ,valid_loss, loss_per_res


def save_checkpoint(save_dir, model, weight_optimizer, scheduler, epoch, loss_progress, accuracy_progress, valid_loss, loss_per_res, index):
    state = {
        'test_properties': vars(args),
        'seed': args.seed,
        'factor' : args.factor,
        'indices': index,
        'loss_progress': loss_progress,
        'accuracy_progress': accuracy_progress,
        'model': model.state_dict(),
        'epoch': epoch,
        'weight_optimizer': weight_optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'valid_loss' : valid_loss,
        'loss_per_res' : loss_per_res
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

def train_valid(train_queue, validation_queue, model, weight_optimizer, criterion, loss_per_res, valid_loss):
    # sample path
    train_loss = 0
    train_correct = 0
    validation_correct = 0
    train_total = 0
    validation_total = 0
    validation_accuracy = 0
    validation_iterator = iter(validation_queue)
    for batch_idx, (train_inputs, train_targets) in enumerate(train_queue):
        # sample path
        prob = Categorical(find_path2(loss_per_res))
        path = prob.sample()
        model.set_path(path)

        model.train()
        train_inputs, train_targets = train_inputs.cuda(), train_targets.cuda()
        weight_optimizer.zero_grad()
        train_outputs = model(train_inputs)
        train_minibatch_loss = criterion(train_outputs, train_targets)
        train_minibatch_loss.backward()  # TODO: activate for MiLeNAS
        weight_optimizer.step()

        train_loss += train_minibatch_loss.cpu().item()
        train_correct, train_total = calculate_accuracy(train_outputs, train_targets, train_total, train_correct)

        #inference
        model.eval()
        try:
            validation_inputs, validation_targets = next(validation_iterator)
        except:
            validation_iterator = iter(validation_queue)
            validation_inputs, validation_targets = next(validation_iterator)
        validation_inputs, validation_targets = validation_inputs.cuda(), validation_targets.cuda()
        validation_outputs = model(validation_inputs)
        valid_loss = criterion(validation_outputs, validation_targets)
        validation_correct, validation_total = calculate_accuracy(validation_outputs, validation_targets, validation_total, validation_correct)
        validation_accuracy = validation_correct/validation_total
        loss_per_res = find_path1(path, loss_per_res, valid_loss)

    return train_loss, valid_loss, train_correct/train_total, validation_accuracy


def find_path1(path, loss_per_res, valid_loss):
    torch.roll(loss_per_res,-1,2)
    for l in range(loss_per_res.size[0]):
        torch.roll(loss_per_res[l, path[l]], -1, 2)
        loss_per_res[l, path[l], -1] = -valid_loss
    return loss_per_res

def find_path2(loss_per_res):
    avg = torch.mean(loss_per_res, 2)
    prob = torch.nn.functional.Softmax(avg, dim=1)
    return prob


def test(test_queue, model, criterion=nn.CrossEntropyLoss()):
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
    return test_loss, test_correct/test_total





if __name__ == '__main__':
  main()
