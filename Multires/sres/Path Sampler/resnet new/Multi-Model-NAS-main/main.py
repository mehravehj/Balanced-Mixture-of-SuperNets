import argparse
import copy
import os
from datetime import datetime
from os import path

import torch
import torch.nn as nn
import torch.utils.data

from utils.NAS_trainer import create_models, create_optimizers, train_valid, validate_all
from utils.data_loader import data_loader
from utils.search_space_design import create_search_space, intialize_prob_matrix
from utils.utility_functions import string_to_list

parser = argparse.ArgumentParser(description='PyTorch Resnet multi model NAS Training')
parser.add_argument('--dataset', '-d', default='CIFAR10', type=str, help='dataset name')
parser.add_argument('--channels_string', '-c', type=str, default='16,16,16,16,32,32,32,64,64,10', help='number of channels per layer')
parser.add_argument('--leng', '-l', type=int, default=10, help='depth of network')
parser.add_argument('--batchsize', '-b', type=int, default=128, help='batch size')
parser.add_argument('--test_name', '-tn', type=int, default=50, help='test name for saving model')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--epochs', '-e', type=int, default=100, help='epochs to train')
parser.add_argument('--validation_percent', '-vp', type=float, default=0.5, help='percent of train data for validation')

parser.add_argument('--learning_rate', '-lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--min_learning_rate', '-mlr', default=0.001, type=float, help='min learning rate')
parser.add_argument('--weight_momentum', '-wm', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', '-wd', default=0, type=float, help='weight decay')

parser.add_argument('--max_scales', '-mx', type=int, default=3, help='number of scales to use')
parser.add_argument('--data_dir', '-dd', type=str, default='./data/', help='dataset directory')
parser.add_argument('--workers', '-wr', type=int, default=0, help='number of workers to load data')

parser.add_argument('--n_models', '-nm', type=int, default=2, help='number of models')
parser.add_argument('--temperature', '-tmp', default=1.0, type=float, help='logit multiplier')
parser.add_argument('--max_temperature', '-mtp', default=1.0, type=float, help='max multiplier')
parser.add_argument('--ema_decay', '-emd', default=0.9, type=float, help='exponential moving average decay')
parser.add_argument('--init_logit', '-ilog', default=1.0, type=float, help='initial logits for path probabilitites')
# parser.add_argument('--local_rank', '-lrank', type=int, default=0)
# parser.add_argument('-ngpu', type=int, default=4)

args = parser.parse_args()

def main():
    print(args)
    startTime = datetime.now()
    epochs = args.epochs
    decay = args.ema_decay
    max_scales = args.max_scales
    init_logit = args.init_logit
    temperature = args.temperature
    max_temp = args.max_temperature
    num_layers = args.leng
    channels = string_to_list(args.channels_string, num_layers)
    num_models = args.n_models
    lr = args.learning_rate
    mlr = args.min_learning_rate
    moment = args.weight_momentum
    w_decay = args.weight_decay

    save_dir = './checkpoint/multilodel_chpt_' + str(args.test_name) + '.t7'  # checkpoint save directory

    #linear temperature
    temp = [((max_temp-temperature)/epochs * i + temperature) for i in range(epochs+1)]
    print('temperature:' ,temp)

    # create network
    if args.dataset == 'CIFAR10':
        ncat = 10
    print('creating %d models....' %num_models)
    nets = create_models(num_layers, channels, num_models)
    for net in nets:
        net.cuda()
    print(nets[0])
    optimizers, schedulers = create_optimizers('SGD', nets, num_models, lr, moment, w_decay, epochs, mlr)

    criterion = nn.CrossEntropyLoss()  # classification loss criterion
    criterion = criterion.cuda()

    current_epoch = 0

    ###loading data
    if path.exists(args.data_dir):
        dataset_dir = args.data_dir
    else:
        dataset_dir = '~/Desktop/codes/multires/data/'

    index = 0
    train_loader, validation_loader, test_loader, indices, num_class = data_loader(args.dataset, args.validation_percent,
                                                                                   args.batchsize,
                                                                                   indices=index,
                                                                                   dataset_dir=dataset_dir,
                                                                                   workers=args.workers)

    ### intialize paths
    paths, num_paths = create_search_space(args.leng, max_scales)
    path_w, weight_mat = intialize_prob_matrix(num_paths, num_models, init_paths_w = 1, init_models_w = 1) # create initial probabilities
    # counter_matrix = torch.FloatTensor([[0 for i in range(num_models)] for j in range(num_paths)])

    counter_matrix = torch.zeros((num_paths, num_models), dtype=int)
    c_matrix = []
    w_matrix = []
    t_loss = []
    t_acc = []
    v_loss = []
    v_acc = []
    acc_mat = [0 for i in range(epochs)]


    for epoch in range(current_epoch, args.epochs + 1):
        print('epoch ', epoch)
        print('net learning rate: ', optimizers[0].param_groups[0]['lr'])
        # train
        counter_matrix, weight_mat, train_loss, validation_loss, t_accuracy, v_accuracy = train_valid(nets, train_loader, optimizers, path_w, paths, weight_mat,
                  temp[epoch], validation_loader, decay, counter_matrix, criterion=nn.CrossEntropyLoss())

        c_matrix.append(copy.deepcopy(counter_matrix))
        w_matrix.append(copy.deepcopy(weight_mat))
        t_loss.append(train_loss)
        t_acc.append(t_accuracy[0]/t_accuracy[1])
        v_loss.append(validation_loss)
        v_acc.append(v_accuracy[0]/v_accuracy[1])
        print('train  acc', t_accuracy[0]/v_accuracy[1])
        print('v acc', v_accuracy[0]/v_accuracy[1])

        for m in range(num_models):
            schedulers[m].step()
        print('Training time: ', datetime.now() - startTime)

        if epoch != 0 and (epoch % 2 == 0 or epoch == args.epochs):  # test and save checkpoint every 5 epochs
            print('Saving models and progress...')
            if epoch != 0 and (epoch % 4 == 0 or epoch == args.epochs):
                init_acc_mat = validate_all(nets, num_models, paths, num_paths, validation_loader)
                acc_mat[epoch] = copy.deepcopy(init_acc_mat)
                print(init_acc_mat)

            save_checkpoint(save_dir, nets, optimizers, schedulers, epoch,t_acc, v_acc, t_loss, v_loss, c_matrix, w_matrix, index, num_models, acc_mat, temp)

def save_checkpoint(save_dir, models, optimizers, schedulers, epoch,t_acc, v_acc, t_loss, v_loss, c_matrix, w_matrix, index, num_models, acc_mat, temp):
    state = {
        'test_properties': vars(args),
        'seed': args.seed,
        'indices': index,
        't_loss': t_loss,
        't_acc': t_acc,
        'v_loss': v_loss,
        'v_acc': v_acc,
        'c_matrix': c_matrix,
        'w_matrix': w_matrix,
        'acc_mat': acc_mat,
        'model': [models[i].state_dict() for i in range(num_models)],
        'epoch': epoch,
        'temperature' : temp,
        'weight_optimizer': [optimizers[i].state_dict() for i in range(num_models)],
        'scheduler_state': [schedulers[i].state_dict() for i in range(num_models)],
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, save_dir)


if __name__ == '__main__':
  main()
