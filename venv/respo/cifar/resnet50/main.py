import argparse
import copy
import os
from datetime import datetime
from os import path

import torch
import torch.nn as nn
import torch.utils.data

import random
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn

from utils.NAS_trainer import create_models, create_optimizers, train_valid, validate_all
from utils.data_loader import data_loader
from utils.search_space_design import create_search_space, intialize_prob_matrix
from utils.utility_functions import string_to_list
from create_model import ResNet, BasicBlock, Bottleneck

parser = argparse.ArgumentParser(description='PyTorch Resnet multi model NAS Training')
parser.add_argument('--dataset', '-d', default='CIFAR10', type=str, help='dataset name')
parser.add_argument('--batch_size', '-b', type=int, default=256, help='batch size')
parser.add_argument('--test_name', '-tn', type=int, default=1, help='test name for saving model')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--epochs', '-e', type=int, default=75, help='epochs to train')
parser.add_argument('--validation_percent', '-vp', type=float, default=0.5, help='percent of train data for validation')

parser.add_argument('--learning_rate', '-lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--min_learning_rate', '-mlr', default=0.001, type=float, help='min learning rate')
parser.add_argument('--weight_momentum', '-wm', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', '-wd', default=0.001, type=float, help='weight decay')

parser.add_argument('--sched_type', '-st', default='cosine_anneal', type=str, help='scheduler type, cosine annealing')
parser.add_argument('--first_cycle_steps', '-fcs', type=int, default=100, help='first cycle epochs')
parser.add_argument('--cycle_mult', '-cm', default=1.0, type=float, help='Cycle steps magnification')
parser.add_argument('--warmup_steps', '-ws', type=int, default=0, help='Linear warmup step size')
parser.add_argument('--gamma', '-gm', default=1.0, type=float, help='Decrease rate of max learning rate by cycle')

parser.add_argument('--data_dir', '-dd', type=str, default='./data/', help='dataset directory')
parser.add_argument('--workers', '-wr', type=int, default=8, help='number of workers to load data')

parser.add_argument('--n_models', '-nm', type=int, default=1, help='number of models')
parser.add_argument('--temperature', '-tmp', default=1, type=float, help='logit multiplier')
parser.add_argument('--max_temperature', '-mtp', default=150.0, type=float, help='max multiplier')
parser.add_argument('--ema_decay', '-emd', default=0.9, type=float, help='exponential moving average decay')
parser.add_argument('--init_logit', '-ilog', default=1.0, type=float, help='initial logits for path probabilitites')

parser.add_argument('--fp32', action='store_true',
                    help='train in full precision (instead of fp16)')
parser.add_argument('--dist-url', default=f'tcp://127.0.0.1:{random.randint(1, 9999) + 30000}', type=str,
                    help='url used to set up distributed training')


parser.add_argument('--threshold', '-thr', default=0.0001, type=float, help='how many times to normalize prob')
# parser.add_argument('--local_rank', '-lrank', type=int, default=0)
# parser.add_argument('-ngpu', type=int, default=4)

def main():
    args = parser.parse_args()
    args.ngpus = torch.cuda.device_count()
    mp.spawn(main_worker, nprocs=args.ngpus, args=(args,))


def main_worker(gpu, args):
    args.gpu = gpu
    print("Use GPU: {} for training".format(args.gpu))
    print('percision:', args.fp32)
    dist.init_process_group('nccl', init_method=args.dist_url, rank=args.gpu, world_size=args.ngpus)
    torch.cuda.set_device(args.gpu)

    seed_np = int(np.random.randint(low=0, high=9999, size=None, dtype=int))
    print('randomseed is:', seed_np)
    torch.manual_seed(seed_np)
    np.random.seed(seed_np)

    cudnn.benchmark = True
    print('Test: ', args.test_name)
    print('-------------------')
    print(args)
    print('-------------------')
    startTime = datetime.now()
    epochs = args.epochs
    decay = args.ema_decay
    init_logit = args.init_logit
    temperature = args.temperature
    max_temp = args.max_temperature
    num_models = args.n_models
    # optimizer parameters
    lr = args.learning_rate
    mlr = args.min_learning_rate
    moment = args.weight_momentum
    w_decay = args.weight_decay
    # schaduler parameters
    sched_type = args.sched_type
    first_cycle_steps = args.first_cycle_steps
    cycle_mult = args.cycle_mult
    warmup_steps = args.warmup_steps
    gamma = args.gamma
    threshold = args.threshold

    save_dir = './checkpoint/food_mm_chpt_' + str(args.test_name) + '.t7'  # checkpoint save directory

    #linear temperature
    temp = [((max_temp-temperature)/epochs * i + temperature) for i in range(epochs+1)]
    print('-------------------')
    print('temperature:' ,temp)

    # create network
    print('creating %d models....' %num_models)

    """
    create n identical models
    """
    nets = []
    # nets = [ResNet(Bottleneck, [3, 4, 6, 3]).to(memory_format=torch.channels_last).cuda() for _ in range(num_models)]  # create multiple models
    for n in range(num_models):
        net = ResNet(Bottleneck, [3, 4, 6, 3])
        net = net.to(memory_format=torch.channels_last).cuda()
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu])
        nets.append(net)


    # nets = create_models(num_models)
    # for net in nets:
    #     net.cuda()
    print('-------------------')
    print(nets[0])
    optimizers, schedulers = create_optimizers(sched_type, nets, num_models, lr, moment, w_decay, epochs, mlr, first_cycle_steps, cycle_mult, warmup_steps, gamma)

    criterion = nn.CrossEntropyLoss()  # classification loss criterion
    criterion = criterion.cuda()

    current_epoch = 0

    ###loading data
    # if path.exists(args.data_dir):
    #     dataset_dir = args.data_dir
    # else:
    #     dataset_dir = '~/Desktop/codes/multires/data/'

    # index = 0
    # train_loader, validation_loader, test_loader, indices, num_class = data_loader(args.dataset, args.validation_percent,
    #                                                                                args.batchsize,
    #                                                                                indices=index,
    #                                                                                dataset_dir=dataset_dir,
    #
    #                                                                                workers=args.workers)
    dataset_dir = args.data_dir
    train_file = 'train_50_256_1.0_90.ffcv'
    val_file = 'val_50_256_1.0_90.ffcv'
    train_loader, validation_loader = data_loader(gpu, args.fp32, dataset_dir, train_file, val_file, args.batch_size, args.workers, np_seed)

    ### intialize paths
    paths, num_paths = create_search_space()
    path_w, weight_mat = intialize_prob_matrix(num_paths, num_models, init_paths_w = 1, init_models_w = 1) # create initial probabilities
    # counter_matrix = torch.FloatTensor([[0 for i in range(num_models)] for j in range(num_paths)])

    counter_matrix = torch.zeros((num_paths, num_models), dtype=int)
    c_matrix = []
    w_matrix = []
    p_matrix = []
    p_marginal = []
    t_loss = []
    t_acc = []
    v_loss = []
    v_acc = []
    acc_mat = [0 for i in range(epochs+1)]
    acc_mat_per_class = [0 for i in range(epochs+1)]
    fp32 = args.fp32
    scaler = torch.cuda.amp.GradScaler(enabled=not fp32)

    for epoch in range(current_epoch, args.epochs + 1):
        print('epoch ', epoch, flush=True)
        print('net learning rate: ', optimizers[0].param_groups[0]['lr'])
        #print('model weights')
        #print(nets[0].module.layer1[0].conv2.weight[0,0,:,:])
        # train
        counter_matrix, weight_mat, prob_mat, p_marg, train_loss, validation_loss, t_accuracy, v_accuracy = train_valid(nets, train_loader, optimizers, path_w, paths, weight_mat,
                  temp[epoch], validation_loader, decay, counter_matrix, threshold, scaler, fp32, criterion=nn.CrossEntropyLoss())

        c_matrix.append(copy.deepcopy(counter_matrix))
        w_matrix.append(copy.deepcopy(weight_mat))
        p_matrix.append(copy.deepcopy(prob_mat))
        p_marginal.append(copy.deepcopy(p_marg))
        t_loss.append(train_loss)
        t_acc.append(t_accuracy[0]/t_accuracy[1])
        v_loss.append(validation_loss)
        v_acc.append(v_accuracy[0]/v_accuracy[1])
        print('train  acc', t_accuracy[0]/t_accuracy[1])
        print('v acc', v_accuracy[0]/v_accuracy[1])

        for m in range(num_models):
            schedulers[m].step()
        print('Training time: ', datetime.now() - startTime, flush=True)

        if epoch == args.epochs:  # test and save checkpoint every 10 epochs
            print('Saving models and progress...', flush=True)
            if epoch == args.epochs:
                init_acc_mat, init_acc_mat_per_class = validate_all(nets, num_models, paths, num_paths, validation_loader, prob_mat, weight_mat)
                acc_mat[epoch] = copy.deepcopy(init_acc_mat)
                acc_mat_per_class[epoch] = copy.deepcopy(init_acc_mat_per_class)
                # print(init_acc_mat)

            save_checkpoint(save_dir, nets, optimizers, schedulers, epoch,t_acc, v_acc, t_loss, v_loss, c_matrix, w_matrix, p_matrix, p_marginal, num_models, acc_mat, acc_mat_per_class, temp)

def save_checkpoint(save_dir, models, optimizers, schedulers, epoch,t_acc, v_acc, t_loss, v_loss, c_matrix, w_matrix, p_matrix, p_marginal, num_models, acc_mat, acc_mat_per_class, temp):
    state = {
        't_loss': t_loss,
        't_acc': t_acc,
        'v_loss': v_loss,
        'v_acc': v_acc,
        'c_matrix': c_matrix,
        'w_matrix': w_matrix,
        'p_matrix': p_matrix,
        'p_marginal': p_marginal,
        'acc_mat': acc_mat,
        'acc_mat_per_class': acc_mat_per_class,
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
