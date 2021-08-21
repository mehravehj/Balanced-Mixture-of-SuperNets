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
from torch.distributions.categorical import Categorical as Categorical
from data_loader import data_loader
from model_TD import path_model
# print

parser = argparse.ArgumentParser(description='PyTorch Mutiresolution Training')
parser.add_argument('--dataset', '-d', default='CIFAR10', type=str, help='dataset name')
parser.add_argument('--channels', '-c', type=str, default='16', help='number of channels')
parser.add_argument('--leng', '-l', type=int, default=6, help='depth of network')
parser.add_argument('--batchsize', '-b', type=int, default=200, help='batch size')
parser.add_argument('--test_name', '-tn', type=int, default=666, help='test name for saving model')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--epochs', '-e', type=int, default=100, help='epochs to train')

parser.add_argument('--number_train', '-nb', type=int, default=0, help='number of training examples')

parser.add_argument('--validation_percent', '-vp', type=float, default=0.5, help='percent of train data for validation')
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
parser.add_argument('--factor', '-f', default=1, type=float, help='An multiplier for softmax(loss)')
parser.add_argument('--gamma', '-gm', default=1, type=float, help='factor for learning rate')
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
    # net = nn.DataParallel(net)
    # print(net)

    # loss criterion defined
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    # do I need weight parameters? might help with not storing disconnected BN
    # optimizer and scheduler
    parameters_by_layer = []
    for l in range(args.leng):
        layer_param_name = 'layer.' + str(l)
        layer_param = [y for x, y in net.named_parameters() if x.find(layer_param_name) is not -1]
        parameters_by_layer.append(layer_param)
    list_optimizer = [{'params': i, 'lr': args.learning_rate} for i in parameters_by_layer]
    list_optimizer[0] = {'params' : list_optimizer[0]['params']}

    # weight_optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.weight_momentum, weight_decay=args.weight_decay)
    weight_optimizer = optim.SGD(list_optimizer, lr=args.learning_rate, momentum=args.weight_momentum, weight_decay=args.weight_decay)
    del list_optimizer
    del parameters_by_layer
    # print(weight_optimizer)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(weight_optimizer, float(args.epochs), eta_min=args.min_learning_rate)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(weight_optimizer, T_0=args.scheduler_epochs, T_mult=args.lr_mult,
    #                                                              eta_min=args.min_learning_rate)
    # train function
    train_function = train_valid
    # loading models to resume training
    save_dir = './checkpoint_loss/ckpt_TD_' + str(args.test_name) + '.t7'

    epoch, loss_progress, accuracy_progress, best_accuracy, index , loss_per_res_vector, probs_progress, res_loss_progress, best_paths_progress, worst_paths_progress\
        = load_checkpoint(save_dir, net, weight_optimizer, scheduler)

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
        train_loss, validation_loss, train_accuracy, validation_accuracy, loss_per_res_vector, \
        res_loss, res_prob, best_paths, worst_paths = \
            train_function(train_loader, validation_loader, net, weight_optimizer, criterion, loss_per_res_vector, epoch)
        scheduler.step()
        probs_progress.append(res_prob)
        res_loss_progress.append(res_loss)
        loss_progress['train'].append(train_loss)
        loss_progress['validation'].append(validation_loss)
        best_paths_progress.append(best_paths)
        worst_paths_progress.append(worst_paths)


        if epoch % 5 == 0 or epoch == 1:
            print('Testing...')
            test_loss, test_accuracy = test(test_loader, net)
            loss_progress['test'].append(test_loss)

            accuracy_progress['train'].append(train_accuracy)
            accuracy_progress['validation'].append(validation_accuracy)
            accuracy_progress['test'].append(test_accuracy)
            print('train accuracy: ', train_accuracy, ' ....... validation accuracy: ', validation_accuracy, ' ....... test accuracy: ', test_accuracy)
            print(net.path)


            save_checkpoint(save_dir, net, weight_optimizer, scheduler, epoch, loss_progress, accuracy_progress, loss_per_res_vector, index, probs_progress, res_loss_progress, best_paths_progress, worst_paths_progress)
        print('Training time: ', datetime.now() - startTime)


def load_checkpoint(save_dir, model, weight_optimizer, scheduler):
    epoch = 0
    index = 0
    best_accuracy = 0
    loss_progress = {'train': [], 'validation': [], 'test': []}
    accuracy_progress = {'train': [], 'validation': [], 'test': []}
    loss_per_res = torch.zeros(args.leng, args.max_scales, 97)
    res_probs_progress = []
    res_loss_progress = []
    best_paths_progress = []
    worst_paths_progress = []

    if path.exists(save_dir):
        print('Loading from checkpoint...')
        checkpoint = torch.load(save_dir)
        epoch = checkpoint['epoch']
        loss_progress = checkpoint['loss_progress']
        accuracy_progress = checkpoint['accuracy_progress']
        index = checkpoint['indices']
        loss_per_res = checkpoint['loss_per_res_vector']
        res_probs_progress = checkpoint['probs_progress']
        res_loss_progress = checkpoint['loss_per_res_progress']
        best_paths_progress = checkpoint['best_paths_progress']
        worst_paths_progress = checkpoint['worst_paths_progress']

        model.load_state_dict(checkpoint['model'])
        weight_optimizer.load_state_dict(checkpoint['weight_optimizer'])
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state'])
        except:
            pass

    return epoch, loss_progress, accuracy_progress, best_accuracy, index , loss_per_res, res_probs_progress, res_loss_progress, best_paths_progress, worst_paths_progress


def save_checkpoint(save_dir, model, weight_optimizer, scheduler, epoch, loss_progress, accuracy_progress, loss_per_res, index, probs, loss_per_res_progress, best_paths_progress, worst_paths_progress):
    state = {
        'test_properties': vars(args),
        'seed': args.seed,
        'indices': index,
        'loss_progress': loss_progress,
        'accuracy_progress': accuracy_progress,
        'model': model.state_dict(),
        'epoch': epoch,
        'weight_optimizer': weight_optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'loss_per_res_vector' : loss_per_res,
        'probs_progress' : probs,
        'loss_per_res_progress': loss_per_res_progress,
         'best_paths_progress': best_paths_progress,
         'worst_paths_progress': worst_paths_progress,
    }
    if not os.path.isdir('checkpoint_loss'):
        os.mkdir('checkpoint_loss')
    torch.save(state, save_dir)


def calculate_accuracy(logits, target, cul_total=0, cul_prediction=0):
    _, test_predicted = logits.max(1)
    test_total = target.size(0)
    correct_prediction = test_predicted.eq(target).sum().item()
    cul_prediction += correct_prediction
    cul_total += test_total
    return cul_prediction, cul_total

def train_valid(train_queue, validation_queue, model, weight_optimizer, criterion, loss_per_res, epoch):
    '''

    :param train_queue: training data
    :param validation_queue: validation data
    :param model:
    :param weight_optimizer:
    :param criterion:
    :param loss_per_res: 3d matrix (layer, max_scales, avg_window) containing validation batch loss in time
    :return:

    '''
    # sample path
    train_loss = 0
    validation_loss = 0
    train_correct = 0
    validation_correct = 0
    train_total = 0
    validation_total = 0
    validation_accuracy = 0
    validation_iterator = iter(validation_queue)
    paths = {}
    # find max lr for cosine annealing
    lr_base = args.min_learning_rate + 1/2 *(args.learning_rate - args.min_learning_rate) (1+np.cos(epoch/args.epochs))
    # ratios for different resolutions
    lr_res_ratio = [((1/4)**i)**args.gamma for i in range(args.max_scales)]

    for batch_idx, (train_inputs, train_targets) in enumerate(train_queue):
        # calculate factor for epoch
        factor = (args.factor - 0.5) / (args.epochs * 97) * (epoch * 97 + batch_idx)
        # sample and set path
        avg_loss, prob = res_distribution(loss_per_res, factor)
        prob = Categorical(prob)
        path = prob.sample().data
        # print(path)
        # change learning rate depending on path
        r = 0
        for g in weight_optimizer.param_groups:
            g['lr'] = lr_base * lr_res_ratio[path[r]]
            # g['lr'] = args.learning_rate * lr_res_ratio[path[r]]
            r += 1
        #     print(g['lr'])
        # print(weight_optimizer)
        model.set_path(path)
        #train model
        model.train()
        train_inputs, train_targets = train_inputs.cuda(), train_targets.cuda()
        weight_optimizer.zero_grad()
        train_outputs = model(train_inputs)
        train_minibatch_loss = criterion(train_outputs, train_targets)
        train_minibatch_loss.backward()
        weight_optimizer.step()
        train_loss += train_minibatch_loss.cpu().item() # sum of train loss in one epoch
        train_correct, train_total = calculate_accuracy(train_outputs, train_targets, train_total, train_correct)
        # print(path)
        # print(model.layer[0].block.conv1.grad)
        #inference, validation loss to update probabilities
        model.eval()
        with torch.no_grad():
            try:
                validation_inputs, validation_targets = next(validation_iterator)
            except:
                validation_iterator = iter(validation_queue)
                validation_inputs, validation_targets = next(validation_iterator)
            validation_inputs, validation_targets = validation_inputs.cuda(), validation_targets.cuda()
            validation_outputs = model(validation_inputs)
            valid_minibatch_loss = criterion(validation_outputs, validation_targets) # batch loss will be added to loss_per_res
            validation_loss += valid_minibatch_loss.cpu().item() # sum of validation loss in one epoch
            validation_correct, validation_total = calculate_accuracy(validation_outputs, validation_targets, validation_total, validation_correct)
            validation_accuracy = validation_correct/validation_total
            loss_per_res = update_loss_per_res(path, loss_per_res, valid_minibatch_loss) # updated

            #paths
            paths[path] = valid_minibatch_loss
    best_paths = {key: paths[key] for key in sorted(paths, key=paths.get)[:10]}
    worst_paths = {key: paths[key] for key in sorted(paths, key=paths.get, reverse=True)[:10]}
    print(best_paths)
    print(worst_paths)
    # print(loss_per_res[0,0,:])
    # print(loss_per_res[-1, -1,:])
    # epoch average statistics
    train_loss = train_loss / (batch_idx + 1)
    validation_loss = validation_loss / (batch_idx + 1)

    avg_loss_end, prob_end = res_distribution(loss_per_res, factor)
    print('probabilities', res_distribution(loss_per_res,factor))
    return train_loss, validation_loss, train_correct/train_total, validation_accuracy, loss_per_res, avg_loss_end, prob_end, best_paths, worst_paths


def update_loss_per_res(path, loss_per_res, valid_loss):
    # print(loss_per_res[0,0,-10:])
    for l in range(loss_per_res.size()[0]):
        loss_per_res[l, path[l]] = torch.roll(loss_per_res[l, path[l]], -1, 0)
        loss_per_res[l, path[l], -1] = -valid_loss
    return loss_per_res

def update_loss_per_res_epoch(path, loss_per_res, valid_loss, visit_last_epoch):
    # print(loss_per_res[0,0,-10:])
    for l in range(loss_per_res.size()[0]):
        loss_per_res[l, path[l]] = torch.roll(loss_per_res[l, path[l]], -1, 0)
        loss_per_res[l, path[l], -1] = -valid_loss
    return loss_per_res, visit_last_epoch

def res_distribution(loss_per_res, factor):
    # print(loss_per_res)
    avg = torch.mean(loss_per_res, 2)
    # print(avg)
    # print(avg.size())
    prob = nn.functional.softmax(avg.data*factor, dim=1)
    # print(prob.size())
    # print('prob', prob)
    return avg, prob.data


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
