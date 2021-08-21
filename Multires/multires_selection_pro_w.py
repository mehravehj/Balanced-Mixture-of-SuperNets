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
from model_selection_pro import multires_model

parser = argparse.ArgumentParser(description='PyTorch Mutiresolution Training')
parser.add_argument('--dataset', '-d', default='CIFAR10', type=str, help='dataset name')
parser.add_argument('--channels', '-c', type=str, default='16', help='number of channels')
parser.add_argument('--leng', '-l', type=int, default=6, help='depth of network')
parser.add_argument('--batchsize', '-b', type=int, default=200, help='batch size')
parser.add_argument('--test_name', '-tn', type=int, default=666, help='test name for saving model')
# parser.add_argument('--selection_name', '-sn', type=int, default=666, help='test name for saving model')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--epochs', '-e', type=int, default=100, help='epochs to train')
parser.add_argument('--fine_epochs', '-fe', type=int, default=100, help='epochs to finetune for each step')

parser.add_argument('--number_train', '-nb', type=int, default=0, help='number of training examples')

parser.add_argument('--validation_percent', '-vp', type=float, default=0.2, help='percent of train data for validation')

parser.add_argument('--learning_rate', '-lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--min_learning_rate', '-mlr', default=0.001, type=float, help='min learning rate')
parser.add_argument('--weight_momentum', '-wm', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', '-wd', default=0, type=float, help='weight decay')

parser.add_argument('-alr', default=0.01, type=float, help=' alpha learning rate')
parser.add_argument('-am', default=0.9, type=float, help='alpha momentum')
parser.add_argument('-awd', default=0, type=float, help='weight decay')
parser.add_argument('--alpha_train_start', '-ats', type=int, default=0, help='epochs to start training alpha')
parser.add_argument('--initial_alpha', '-ina', type=str, default='0', help='alpha initialization 0,1,2,3')  ##
parser.add_argument('--net_type', '-ty', default='multires', type=str, help='normal or multires')
parser.add_argument('--mscale', '-ms', type=str, default='sconv', help='sconv or sres')
parser.add_argument('--pooling', '-p', type=str, default='0', help='use multiscales')
parser.add_argument('--max_scales', '-mx', type=int, default=4, help='number of scales to use')

parser.add_argument('--data_dir', '-dd', type=str, default='./data/', help='dataset directory')

parser.add_argument('--workers', '-wr', type=int, default=0, help='number of workers to load data')
parser.add_argument('--batch_norm', '-bn', type=bool, default=True, help='use batchnorm at on output of layer')

args = parser.parse_args()


def main():
    startTime = datetime.now()
    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    finetune_epoch = args.fine_epochs

    if args.dataset == 'TIN': #number of categories
        ncat=200
    else:
        ncat = 10
    print('Test ', args.test_name)
    print(args)

    #load network
    save_dir = './checkpoint_pro/ckpt_pro_' + str(args.test_name) + '.t7' #selection test number
    if path.exists(save_dir): # if already selected from trained model, continue loading
        checkpoint = torch.load(save_dir)
        layer = checkpoint['current_layer']
        selected_resolutions = checkpoint['selected_res']
        net = multires_model(ncat=ncat, net_type=args.net_type, mscale=args.mscale, channels=args.channels,
                             leng=args.leng, max_scales=args.max_scales, factor=1,
                             initial_alpha=args.initial_alpha, pool=selected_resolutions, current_layer=layer)

    else:
        selected_resolutions = [1]
        layer = 0
        net = multires_model(ncat=ncat, net_type=args.net_type, mscale=args.mscale, channels=args.channels,
                             leng=args.leng, max_scales=args.max_scales, factor=1,
                             initial_alpha=args.initial_alpha, pool=selected_resolutions, current_layer=layer)
    net.cuda()
    # net = nn.DataParallel(net)
    print(net)
    # print(net.state_dict())

    print("param size = %fMB" %(count_parameters_in_MB(net)))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    weight_parameters, alpha_parameters = parameters(net)
    print([x for x, y in net.named_parameters()])
    weight_optimizer = optim.SGD(weight_parameters, lr=args.learning_rate, momentum=args.weight_momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(weight_optimizer, T_0=args.fine_epochs, T_mult=1, eta_min=args.min_learning_rate)
    alpha_optimizer = optim.Adam(alpha_parameters, lr=args.alr, betas=(0.9, 0.999), weight_decay=args.awd)

    train_function = train_valid
    best_model, current_epoch, loss_progress, accuracy_progress, alpha_progress, best_alpha, best_epoch, best_accuracy,\
    index, layer, selected_resolution = load_checkpoint_selection(save_dir, net, weight_optimizer, scheduler,
                                                                  alpha_optimizer)

    #load dataset
    if path.exists(args.data_dir):
        dataset_dir = args.data_dir
    else:
        dataset_dir = '~/Desktop/codes/multires/data/'

    ###loading data
    train_loader, validation_loader, test_loader, indices, num_class = data_loader(args.dataset, args.validation_percent,
                                                                                   args.batchsize, num_train=args.number_train,
                                                                                   indices=index, dataset_dir=dataset_dir,
                                                                                   workers=args.workers)
    # resolution_change_epoch = {'epoch':[current_epoch], 'resolution_selection': [selected_resolutions]}
    for epoch in range(current_epoch+1, args.epochs+1):
        print('epoch ', epoch)
        # print(best_model['layer.1.block.block.conv2.weight'][0,0,:])
        if epoch % finetune_epoch == 0:
            print('alpha')
            print(get_alpha(args.net_type, net))
            # move to next layer
            layer += 1
            print('....selection for layer ', layer)
            selected_resolutions = select_max(selected_resolutions, best_alpha)
            print('resolution ', selected_resolutions[-1], ' selected....')
            net = multires_model(ncat=ncat, net_type=args.net_type, mscale=args.mscale, channels=args.channels,
                                 leng=args.leng, max_scales=args.max_scales, factor=1,
                                 initial_alpha=args.initial_alpha, pool=selected_resolutions, current_layer=layer)
            #copy weights from best model
            print('currenct model')
            print(net.state_dict()['layer.1.block.block.conv2.weight'][0,0,:])
            load_checkpoint_new_layer(save_dir, net)
            print('best model')
            print(net.state_dict()['layer.1.block.block.conv2.weight'][0, 0, :])
            net.cuda()
            # net = nn.DataParallel(net)
            print(net)
            _, alpha_parameters = parameters(net)
            # reinitialize alpha optimizer
            alpha_optimizer = optim.Adam(alpha_parameters, lr=args.alr, betas=(0.9, 0.999), weight_decay=args.awd)
            # reinitialize and recreate network
            best_alpha = 0
            best_accuracy = 0

        print('net learning rate: ', weight_optimizer.param_groups[0]['lr'])
        train_loss, validation_loss, train_accuracy, validation_accuracy = \
            train_function(train_loader, validation_loader, net, weight_optimizer, alpha_optimizer, criterion, epoch)
        scheduler.step()
        # print(train_loss)

        if epoch % 1 == 0 or epoch == 1:
            print('Testing...')
            test_loss, test_accuracy = test(test_loader, net)

            loss_progress['train'].append(train_loss)
            loss_progress['validation'].append(validation_loss)
            loss_progress['test'].append(test_loss)

            accuracy_progress['train'].append(train_accuracy)
            accuracy_progress['validation'].append(validation_accuracy)
            accuracy_progress['test'].append(test_accuracy)

            alpha_progress.append(get_alpha(args.net_type, net))
            print(get_alpha(args.net_type, net))

            if test_accuracy > best_accuracy:
                print('-----------> Best accuracy')
                best_model = {}
                for key in net.state_dict():
                    best_model[key] = net.state_dict()[key].clone()
                best_accuracy = test_accuracy
                best_epoch = epoch
                best_alpha = get_alpha(args.net_type, net)

            print('train accuracy: ', train_accuracy, ' ....... validation accuracy: ', validation_accuracy, ' ....... test accuracy: ', test_accuracy)
            print('best accuracy:', best_accuracy,' at epoch ', best_epoch)

            print('.....SAVING.....')
            if args.net_type == 'normal':
                alpha_optimizer_state = 0

            else:
                alpha_optimizer_state = alpha_optimizer.state_dict()

            save_checkpoint_selection(save_dir, net, best_model, weight_optimizer, scheduler, alpha_optimizer_state, epoch, loss_progress,
                            accuracy_progress, alpha_progress, best_alpha, best_epoch, best_accuracy, index, layer, selected_resolutions)
        # print(len(loss_progress['train']))
        # print(loss_progress['train'])
        print('Training time: ', datetime.now() - startTime)

def select_max(selected_resolutions, alpha):
    res = int(np.argmax(alpha.cpu()[0])) + 1
    selected_resolutions.append(res)
    return selected_resolutions

def load_checkpoint_selection(save_dir, model, weight_optimizer, scheduler, alpha_optimizer):
    epoch = 0
    index = 0
    best_epoch = 0
    best_model = 0
    best_accuracy = 0
    loss_progress = {'train': [], 'validation': [], 'test': []}
    accuracy_progress = {'train': [], 'validation': [], 'test': []}
    alpha_progress = [get_alpha(args.net_type, model)]
    best_alpha = get_alpha(args.net_type, model)
    current_layer = 0
    selected_res = [1]

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
        current_layer = checkpoint['current_layer']
        selected_res = checkpoint['selected_res']

        model.load_state_dict(checkpoint['model'])
        weight_optimizer.load_state_dict(checkpoint['weight_optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        if args.net_type == 'multires':
            alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])

    return best_model, epoch, loss_progress, accuracy_progress, alpha_progress, best_alpha, best_epoch, best_accuracy, index, current_layer, selected_res

def load_checkpoint_new_layer(save_dir, model):
    print('Loading from checkpoint...')
    checkpoint = torch.load(save_dir)
    best_model = checkpoint['best_model']
    best_model_dict = best_model#.state_dict()
    weight_dict = {k: v for k,v in best_model_dict.items() if not ('alpha' in k)}
    best_model_dict.update(weight_dict)
    model.load_state_dict(best_model_dict, strict=False)

def save_checkpoint_selection(save_dir, model, best_model, weight_optimizer, scheduler, alpha_optimizer_state, epoch,
                              loss_progress, accuracy_progress, alpha_progress, best_alpha, best_epoch, best_accuracy,
                              index, current_layer, selected_res):
    state = {
        'test_properties': vars(args),
        'seed': args.seed,
        'factor' : 1,
        'indices': index,
        'best_net': best_model,
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
        'alpha_optimizer': alpha_optimizer_state,
        'scheduler_state': scheduler.state_dict(),
        'current_layer' : current_layer,
        'selected_res' : selected_res
    }
    if not os.path.isdir('checkpoint_pro'):
        os.mkdir('checkpoint_pro')
    torch.save(state, save_dir)


def calculate_accuracy(logits, target, cul_total=0, cul_prediction=0):
    _, test_predicted = logits.max(1)
    test_total = target.size(0)
    correct_prediction = test_predicted.eq(target).sum().item()
    cul_prediction += correct_prediction
    cul_total += test_total
    return cul_prediction, cul_total


def get_alpha(net_type, model):
    alpha = []
    if net_type == 'multires':
        for name, param in model.named_parameters():
            if param.requires_grad and 'alpha' in name:
                # alpha.append(param.cpu().detach())
                alpha.append(param.detach().clone())
        alpha = (torch.stack(alpha, 0))#.numpy()
    return alpha


def count_parameters_in_MB(model): #number of parameters in millions
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters())/1e6


def parameters(model):
    all_parameter_names = [x for x, y in model.named_parameters()]
    weight_param_idx = [idx for idx, x in enumerate(all_parameter_names) if x.find('alpha') is -1]
    alpha_param_idx = [idx for idx, x in enumerate(all_parameter_names) if x.find('alpha') is not -1]
    all_parameters = [y for x, y in model.named_parameters()]
    weight_parameters = [all_parameters[idx] for idx in weight_param_idx]
    alpha_parameters = [all_parameters[idx] for idx in alpha_param_idx]

    return weight_parameters, alpha_parameters


def train_valid(train_queue, validation_queue, model, weight_optimizer, alpha_optimizer, criterion=nn.CrossEntropyLoss(), epoch=0):
    model.train()
    train_loss = []
    validation_loss = []
    train_correct = 0
    validation_correct = 0
    train_total = 0
    validation_total = 0
    validation_accuracy = 0
    validation_iterator = iter(validation_queue)
    for batch_idx, (train_inputs, train_targets) in enumerate(train_queue):
        train_inputs, train_targets = train_inputs.cuda(), train_targets.cuda()
        weight_optimizer.zero_grad()
        train_outputs = model(train_inputs)
        train_minibatch_loss = criterion(train_outputs, train_targets)
        train_minibatch_loss.backward()

        # for l in range(args.leng):
        #     print('layer ', l)
        #     print(torch.norm(model.layer[l].block.block.conv1.weight.grad.detach()) / torch.norm(
        #         model.layer[l].block.block.conv1.weight.detach()) * args.learning_rate)
        #     print(torch.norm(model.layer[l].block.block.conv2.weight.grad.detach()) / torch.norm(
        #         model.layer[l].block.block.conv2.weight.detach()) * args.learning_rate)
        # print(model.layer[2].block.alpha.grad)
        # print(model.layer[3].block.alpha.grad)


        weight_optimizer.step()

        train_loss.append(train_minibatch_loss.detach().clone())
        train_correct, train_total = calculate_accuracy(train_outputs, train_targets, train_total, train_correct)

        if epoch % 100 >= args.alpha_train_start:
            try:
                validation_inputs, validation_targets = next(validation_iterator)
            except:
                validation_iterator = iter(validation_queue)
                validation_inputs, validation_targets = next(validation_iterator)
            validation_inputs, validation_targets = validation_inputs.cuda(), validation_targets.cuda()
            alpha_optimizer.zero_grad()
            validation_outputs = model(validation_inputs)
            validation_minibatch_loss = criterion(validation_outputs, validation_targets)
            # get alpha grads
            # print('alpha')
            # print(get_alpha(args.net_type, model))
            validation_minibatch_loss.backward()

            # print('alpha grads')
            # for l in range(1, args.leng):
            #     print('layer ', l)
            #     print(torch.norm(model.layer[l].block.alpha.grad.detach()) / torch.norm(
            #         model.layer[l].block.alpha.detach()) * args.alr)
            #     print(torch.norm(model.layer[l].block.alpha.grad.detach()) / torch.norm(
            #         model.layer[l].block.alpha.detach()) * args.alr)

            alpha_optimizer.step()

            validation_loss.append(validation_minibatch_loss.detach().clone())
            validation_correct, validation_total = calculate_accuracy(validation_outputs, validation_targets, validation_total, validation_correct)
            validation_accuracy = validation_correct/validation_total
        # print(train_loss.dtype, batch_idx)
        # train_loss = train_loss / (batch_idx + 1)
        # validation_loss = validation_loss / (batch_idx + 1)
    # print(len(train_loss))

    return train_loss, validation_loss, train_correct/train_total, validation_accuracy


def train(train_queue, validation_queue, model, weight_optimizer, alpha_optimizer, criterion=nn.CrossEntropyLoss(), epoch=0):
    model.train()
    train_loss = 0
    validation_loss = 0
    train_correct = 0
    validation_correct = 0
    train_total = 0
    validation_total = 0
    for batch_idx, (train_inputs, train_targets) in enumerate(train_queue):
        # print(batch_idx)
        train_inputs, train_targets = train_inputs.cuda(), train_targets.cuda()
        weight_optimizer.zero_grad()
        train_outputs = model(train_inputs)
        train_minibatch_loss = criterion(train_outputs, train_targets)
        train_minibatch_loss.backward()
        weight_optimizer.step()

        train_loss += train_minibatch_loss.detach().clone()
        train_correct, train_total = calculate_accuracy(train_outputs, train_targets, train_total, train_correct)
    # print('done')
    return train_loss, 0, train_correct/train_total, 0


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
            test_loss += test_minibatch_loss.detach().clone()
            test_correct, test_total = calculate_accuracy(test_outputs, test_targets, test_total, test_correct)
    return test_loss, test_correct/test_total





if __name__ == '__main__':
  main()
