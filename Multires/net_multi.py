from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import os
import argparse
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from multi_functions import data_loader
from multi_functions import sConv2d
from multi_functions import normal_net
from datetime import datetime
import random
#torch.multiprocessing.set_sharing_strategy('file_system')


'''inputs:
    -d dataset: CIFAR10 , STL10 , TIN
    -c channels per layer: 6, 12, 32, etc.
    -l network depth: min 2
    -b batch size: 32, 64, 128
    -tn test name: 1001
    -sn (-r) change savename
    -r resume ----> sn, athr
    --gpu [-r]
    --seed [-r]
    --epochs
    --vp 0, 0.2, 0.5
    -bn 
    --lr --lr_min --wm --wd
    --alr --am --awd
    --ats
    --alinit (--init_alpha) : True, False
    --init_alpha (--alinit) tensor (resolution, layer)
    --athr (r) or (--alinit) :threshold that eliminates some resolutions permanently
    {{--prun}}
    -f 1, 0.5, 3
    --lf --flr
    --dense
    --blr --lfb {{--bthr}} {{--bts}} (--dense)
    -m sconv , sconc , normal
    --p (-m normal): pooling
    -x (-m sconv , sconc)
    -g same filters (-m sconv , sconc)

    '''
startTime = datetime.now()
parser = argparse.ArgumentParser(description='PyTorch Mutiresolution Training')
parser.add_argument('--dataset', '-d', default='CIFAR10', type=str, help='dataset name')
parser.add_argument('--channels', '-c', type=int, default=10, help='number of channels')
parser.add_argument('--leng', '-l', type=int, default=6, help='depth of network')
parser.add_argument('--batchsize', '-b', type=int, default=200, help='batch size')
parser.add_argument('--testname', '-tn', type=int, default=666, help='test name for saving model')
parser.add_argument('--save_name', '--sn', default=0, type=int, help='change save name when resuming')  #
parser.add_argument('--resume', '-r', default=False, action='store_true', help='resume from checkpoint')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--epochs', '-e', type=int, default=100, help='epochs to train')

parser.add_argument('--vp', type=float, default=0.2, help='percent of train data for validation')
parser.add_argument('--bnorm', '-bn', action='store_true', help='use batch normlization')

parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--lr_min', default=0.001, type=float, help='min learning rate')
parser.add_argument('--wm', default=0.9, type=float, help='momentum')
parser.add_argument('--wd', default=0, type=float, help='weight decay')

parser.add_argument('--alr', default=0.01, type=float, help=' alphalearning rate')
parser.add_argument('--am', default=0.9, type=float, help='alpha momentum')
parser.add_argument('--awd', default=0, type=float, help='weight decay')
parser.add_argument('--a_train_start', '--ats', type=int, default=0, help='epochs to start training alpha')
parser.add_argument('--alinit', type=int, default=0, help='alpha initialization [0,1,2,3]')  ##
parser.add_argument('--init_alpha', type=str, default='0', help='alpha initialization [0,1,2,3]')  ##
parser.add_argument('--athr', type=float, default=0, help='alpha threshold')
parser.add_argument('--prun', type=float, default=0, help='alpha threshold')

parser.add_argument('--factor', '-f', type=float, default=1, help='initial softmax factor')
parser.add_argument('--lf', default=False, action='store_true', help='learn f for each layer')
parser.add_argument('--flr', default=0, type=float, help='factor learning rate')

parser.add_argument('--dense', '--dense', action='store_true', help='Dense seq')
parser.add_argument('--blr', default=0, type=float, help='beta learning rate')
parser.add_argument('--lfb', default=False, action='store_true', help='learn f for each layer')
parser.add_argument('--bthr', type=float, default=0, help='threshold for beta')  #
parser.add_argument('--bts', type=int, default=0, help='epochs to start training beta')

parser.add_argument('--mscale', '-m', type=str, default='sconv', help='use multiscales')
parser.add_argument('--pooling', '--p', type=str, default=0, help='use multiscales')
parser.add_argument('--max_scales', '-x', type=int, default=4, help='number of scales to use')
parser.add_argument('--stoch', '-s', action='store_true', help='Stochastic')

parser.add_argument('--usf', '-g', default=False, action='store_true', help='use same filters')  #

# parser.add_argument('--reset_weights', '--rt_w', default=False, action='store_true', help='reset weights upon resuming')#

parser.add_argument('--test_alpha', '--ta', default=0, help='alpha initialization from model')
#
parser.add_argument('--get_actv', '--actv', default=False, action='store_true', help='get activations')

args = parser.parse_args()
test_name = args.testname

print('test: ', test_name)
print(args)

scheduler = True  # scheduler for optimizer (network only)
save_interm_values = True  # save intermediate values such as loss, etc

if args.resume:
    print('==> Resuming from checkpoint...')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt_' + str(test_name) + '.t7')
    if 'seed' in checkpoint:
        seed = checkpoint['seed']
    else:
        print('Warning: unknown loading seed!')
    if 'gpu' in checkpoint:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(checkpoint['gpu'])
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    if args.save_name:  # change name from loaded one
        test_name = args.save_name
    indices = checkpoint['indices']
else:
    seed = random.randint(1,9999)
    indices = []

# for deterministic behavior
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.manual_seed(seed)
np.random.seed(seed)

trainloader, valloader, testloader, indices, ncat = data_loader(args.dataset, args.vp, args.batchsize, indices)

print('==> Building model...')
class multi(nn.Module):
    def __init__(self, ncat=10, mscale='sconv', channels=32, leng=10, max_scales=4, usf=True, factor=1, lf=False,
                 alinit=False, init_alpha=0, athr=0, prun=0, dense=False, lfb=False, pooling=0):
        super(multi, self).__init__()
        self.leng = leng
        self.dense = dense
        self.lfb = lfb
        if self.dense:
            self.beta = nn.Parameter(torch.ones(leng - 2))  # ???????
            print('beta', self.beta)
            if self.lfb:
                self.bfactor = nn.Parameter(torch.ones(1))
            else:
                self.bfactor = 1
        if alinit:
            self.init_alpha = init_alpha
        else:
            self.init_alpha = [0 for _ in range(leng)]
        if mscale == 'normal':
            if pooling:
                pooling = pooling.split(',')
                pooling = [int(i) for i in pooling]
                pooling = [i in pooling for i in range(leng)]
                print(pooling)
            else:
                pooling = [0 for _ in range(leng)]
            listc = [normal_net(3, channels, kernel_size=3, pooling=pooling[0])]
            listc += [normal_net(channels, channels, kernel_size=3, pooling=pooling[i + 1]) for i in range(leng - 2)]
            listc += [normal_net(channels, ncat, kernel_size=3, pooling=pooling[-1])]
        elif mscale == 'sconv':
            if 1:
                listc = [
                    sConv2d(3, channels, 3, max_scales, usf, factor=1, alinit=0, athr=0, lf=0, prun=0, alpha=self.init_alpha[0])]
                listc += [sConv2d(channels, channels, 3, max_scales, usf, factor=1, alinit=0, athr=0, lf=0, prun=0,
                                  alpha=self.init_alpha[i + 1]) for i in range(leng - 2)]
                listc += [sConv2d(channels, ncat, 3, max_scales, usf, factor=1, alinit=0, athr=0, lf=0, prun=0,
                                  alpha=self.init_alpha[-1])]

        self.conv = nn.ModuleList(listc)

    def forward(self, x):
        out = [x]
        # if self.dense:
        #     self.nbeta = F.softmax(self.beta, 0).view(-1, 1)
        #     for c in range(self.leng - 1):
        #         #                print('c',c)
        #         out_layer = self.conv[c](out[-1])
        #         out.append(out_layer)
        #     #            print('out size', len(out))
        #     outall = torch.stack(out[2:], 0)  # all layer except first one
        #     #            if self.bthr:
        #     out = (outall * self.nbeta.view(-1, 1, 1, 1, 1)).sum(0)
        #     out = self.conv[-1](out)
        # else:
        if 1:
            out = x
            for c in range(self.leng):
                out = self.conv[c](out)
        out = F.avg_pool2d(out, kernel_size=out.size()[2:])
        out = out.view(out.size(0), -1)
        return out


# net = multi(ncat=ncat, mscale=args.mscale, channels=args.channels, leng=args.leng, max_scales=args.max_scales,
#             usf=args.usf, factor=args.factor,
#             lf=args.lf, alinit=args.alinit, init_alpha=args.init_alpha, athr=args.athr, prun=args.prun,
#             dense=args.dense,
#             lfb=args.lfb, pooling=args.pooling)
'''modified to fix all param'''
net = multi(ncat=ncat, mscale=args.mscale, channels=args.channels, leng=args.leng, max_scales=args.max_scales,
            usf=args.usf, factor=1,
            lf=0, alinit=0, init_alpha=0, athr=0, prun=0,
            dense=0,
            lfb=0, pooling=args.pooling)

net = net.cuda()
net = torch.nn.DataParallel(net)
cudnn.benchmark = True
print('Test properties:')
print('network:', args.mscale, 'dense', args.dense)
print(net)
# loss
criterion = nn.CrossEntropyLoss()
allnames = [x for x, y in net.named_parameters()]
allparam = [y for x, y in net.named_parameters()]

netidx = [idx for idx, x in enumerate(allnames) if
          x.find('alpha') is -1 and x.find('beta') is -1 and x.find('factor') is -1 and x.find('bfactor') is -1]
alphaidx = [idx for idx, x in enumerate(allnames) if x.find('alpha') is not -1]
factoridx = [idx for idx, x in enumerate(allnames) if x.find('factor') is not -1 and x.find('bfactor') is -1]
betaidx = [idx for idx, x in enumerate(allnames) if x.find('beta') is not -1 and x.find('factor') is -1]
bfactoridx = [idx for idx, x in enumerate(allnames) if x.find('bfactor') is not -1]

netparam = [allparam[idx] for idx in netidx]
optimizernet = optim.SGD(netparam, lr=args.lr, momentum=args.wm, weight_decay=args.wd)
if args.mscale != 'normal':
    if not args.flr and not args.blr:
        alphaidx += factoridx
        alphaidx += betaidx
        alphaidx += bfactoridx
        alphaparam = [allparam[idx] for idx in alphaidx]
        optimizeralpha = torch.optim.Adam(alphaparam, lr=args.alr, betas=(0.9, 0.999), weight_decay=args.awd)
    elif args.flr and not args.blr:
        alphaidx += betaidx
        factoridx += bfactoridx
        alphaparam = [allparam[idx] for idx in alphaidx]
        factorparam = [allparam[idx] for idx in factoridx]
        optimizeralpha = torch.optim.Adam([{'params': alphaparam},
                                           {'params': factorparam, 'lr': args.flr}],
                                          lr=args.alr, betas=(0.9, 0.999), weight_decay=args.awd)

if args.resume:
    # Load checkpoint
    net.load_state_dict(checkpoint['current_net'])
    if 'optimizernet_best' in checkpoint:
         optimizernet.load_state_dict(checkpoint['current_optimizer'])
         optimizer_alpha = checkpoint['current_optimizeralpha']
         if optimizer_alpha:
             optimizeralpha.load_state_dict(checkpoint['current_optimizeralpha'])
         else:
             optimizeralpha = 0
         print('loading optimizer state...')
    if not args.save_name:
        state_net = checkpoint['best_net'],
        best_epoch = checkpoint['best_epoch'],
        best_factors = checkpoint['best_factors'],
        best_nbetas = checkpoint['best_nbetas'],
        best_bfactors = checkpoint['best_bfactors'],
        test_best = checkpoint['best_test']
        train_best = checkpoint['best_train']
        val_best = checkpoint['best_val']
        best_nalphas = checkpoint['best_nalpha']
        train_l = checkpoint['all_train_loss']
        train_a = checkpoint['all_train_acc']
        val_l = checkpoint['all_val_loss']
        val_a = checkpoint['all_val_acc']
        test_l = checkpoint['all_test_loss']
        test_a = checkpoint['all_test_acc']
        start_epoch = checkpoint['current_epoch']
        if scheduler:
            total_epochs = checkpoint['test_prop']['epochs']
            print('scheduler modified!')
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizernet, float(total_epochs), eta_min=args.lr_min)
            scheduler_dict = scheduler.state_dict()
            scheduler_dict['last_epoch'] = checkpoint['current_epoch']
            scheduler.load_state_dict(scheduler_dict)
#            print(scheduler.state_dict())
        if 'current_net' in checkpoint:
            net.load_state_dict(checkpoint['current_net']),
            optimizernet.load_state_dict(checkpoint['current_optimizer'])
            if checkpoint['current_optimizeralpha']:
                optimizeralpha.load_state_dict(checkpoint['current_optimizeralpha'])
            start_epoch = checkpoint['current_epoch']
            scheduler_dict = scheduler.state_dict()
            scheduler_dict['last_epoch'] = start_epoch
            scheduler.load_state_dict(scheduler_dict)
            state_optimizernet = checkpoint['optimizernet_best'],
            state_optimizeralpha = checkpoint['optimizeralpha_best'],
        else:
            state_optimizernet = checkpoint['optimizernet_best']
            state_optimizeralpha = checkpoint['optimizeralpha_best']
#            print(scheduler.state_dict())
            if 'scheduler' in checkpoint:
                print('loading scheduler form checkpoint!')
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizernet, float(total_epochs), eta_min=args.lr_min)
                scheduler.load_state_dict(checkpoint['scheduler'])
        print('resuming from epoch:', start_epoch)
        print('scheduler:',scheduler.state_dict())
    else:
        best_acc = 0
else:
    best_acc = 0
    start_epoch = 0

'''
    train valid used for sconv net
    trains w and alpha
'''
def train_valid(epoch):
    print('Training ...')
    if scheduler:
        scheduler.step()
        lr = scheduler.get_lr()[0]
        print('net learning rate: ', lr)
    net.train()
    loss_train = 0
    loss_valid = 0
    train_correct = 0
    valid_correct = 0
    train_total = 0
    valid_total = 0
    acc_train = 0
    acc_valid = 0
    valset_iterator = iter(valloader)
    for batch_idx, (train_inputs, train_targets) in enumerate(trainloader):
        train_inputs, train_targets = train_inputs.cuda(), train_targets.cuda()
        optimizernet.zero_grad()
        train_outputs = net(train_inputs)
        t_loss = criterion(train_outputs, train_targets)
        t_loss.backward()
        optimizernet.step()
        loss_train += t_loss.item()
        _, train_predicted = train_outputs.max(1)
        train_total += train_targets.size(0)
        t_correct = train_predicted.eq(train_targets).sum().item()
        train_correct += t_correct
        acc_train = train_correct / train_total * 100
        if epoch > args.a_train_start - 1:
            try:
                val_inputs, val_targets = next(valset_iterator)
            except:
                valset_iterator = iter(valloader)
                val_inputs, val_targets = next(valset_iterator)
            val_inputs, val_targets = val_inputs.cuda(), val_targets.cuda()
            optimizeralpha.zero_grad()
            val_outputs = net(val_inputs)
            v_loss = criterion(val_outputs, val_targets)
            v_loss.backward()
            optimizeralpha.step()
            loss_valid += v_loss.item()
            _, val_predicted = val_outputs.max(1)
            valid_total += val_targets.size(0)
            v_correct = val_predicted.eq(val_targets).sum().item()
            valid_correct += v_correct
            acc_valid = valid_correct / valid_total * 100
    return loss_train, acc_train, loss_valid, acc_valid

'''
    train weights
'''
def train(epoch):
    print('Training ...')
    if scheduler:
        scheduler.step()
        lr = scheduler.get_lr()[0]
        print('net learning rate: ', lr)
    net.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    train_acc = 0
    for batch_idx, (train_inputs, train_targets) in enumerate(trainloader):
        train_inputs, train_targets = train_inputs.cuda(), train_targets.cuda()
        optimizernet.zero_grad()
        train_outputs = net(train_inputs)
        t_loss = criterion(train_outputs, train_targets)
        t_loss.backward()
        optimizernet.step()
        train_loss += t_loss.item()
        _, train_predicted = train_outputs.max(1)
        train_total += train_targets.size(0)
        t_correct = train_predicted.eq(train_targets).sum().item()
        train_correct += t_correct
        train_acc = train_correct / train_total * 100
    return train_loss, train_acc, 0, 0

'''
    trains alpha
'''
def validation(epoch):
    print('Validating ...')
    net.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    train_acc = 0
    for batch_idx, (train_inputs, train_targets) in enumerate(valloader):
        train_inputs, train_targets = train_inputs.cuda(), train_targets.cuda()
        optimizeralpha.zero_grad()
        train_outputs = net(train_inputs)
        t_loss = criterion(train_outputs, train_targets)
        t_loss.backward()
        optimizeralpha.step()
        train_loss += t_loss.item()
        _, train_predicted = train_outputs.max(1)
        train_total += train_targets.size(0)
        t_correct = train_predicted.eq(train_targets).sum().item()
        train_correct += t_correct
        train_acc = train_correct / train_total * 100
    return train_loss, train_acc, 0, 0


def test(epoch):
    print('\nTesting...')
    net.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0
    test_acc = 0
    with torch.no_grad():
        for batch_idx, (test_inputs, test_targets) in enumerate(testloader):
            test_inputs, test_targets = test_inputs.cuda(), test_targets.cuda()
            test_outputs = net(test_inputs)
            te_loss = criterion(test_outputs, test_targets)
            test_loss += te_loss.item()
            _, test_predicted = test_outputs.max(1)
            test_total += test_targets.size(0)
            te_correct = test_predicted.eq(test_targets).sum().item()
            test_correct += te_correct
            test_acc = test_correct / test_total * 100

    return test_loss, test_acc


def alpha_parameters():
    if args.mscale == 'normal':
        pooling = [i in args.pooling for i in range(args.leng)]
    #    alpha = t

    elif args.mscale == 'sconv':
        nalphas = []
        factors = []
        nbetas = []
        bfactors = []
        for l in net.modules():
            if type(l) == sConv2d:
                nalphas.append(l.nalpha.cpu().detach().numpy())
                #                filters.append(l.filter)
                if args.lf:
                    factors.append(l.factor.cpu().detach().numpy())
            if args.dense:
                nbeta = net.module.nbeta.data.cpu().numpy()
                nbetas.append(nbeta)
                if args.lfb:
                    bfactor = net.module.bfactor.data.cpu().numpy()
                    bfactors.append(bfactor)

        return nalphas, factors, nbetas, bfactors
if not args.resume:
    train_l = []
    train_a = []
    val_l = []
    val_a = []
    test_l = []
    test_a = []
    test_best = 0
    train_best = 0
    val_best = 0
    best_epoch = 0
    if scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizernet, float(args.epochs), eta_min=args.lr_min)
if start_epoch>=598:
    optimizernet = optim.SGD(netparam, lr=args.lr, momentum=args.wm, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizernet, float(args.epochs), eta_min=args.lr_min)
if args.mscale == 'normal':
    training = train
else:
    training = train_valid
scheduler_state = scheduler.state_dict()
for epoch in range(start_epoch,args.epochs):
    print('Epoch', epoch)
    if 1:
        train_loss, train_acc, valid_loss, valid_acc = training(epoch)
    if 0:
        train_loss, train_acc, _, _ = train(epoch)
        valid_loss, valid_acc, _, _ = validation(epoch)
    print('train accuracy:', train_acc, 'validation accuracy', valid_acc)
    test_loss, test_acc = test(epoch)
    print('test accuracy:', test_acc)
    train_l.append(train_loss)
    train_a.append(train_acc)
    val_l.append(valid_loss)
    val_a.append(valid_acc)
    test_l.append(test_loss)
    test_a.append(test_acc)
    if test_acc > test_best:
            print('-----------> Best accuracy')
            best_epoch = epoch
            test_best = test_acc
            train_best = train_acc
            val_best = valid_acc
            state_optimizernet = optimizernet.state_dict()
            state_net = net.state_dict()
            scheduler_state = scheduler.state_dict()
            if args.mscale != 'normal':
                best_nalphas, best_factors, best_nbetas, best_bfactors = alpha_parameters()
                state_optimizeralpha = optimizeralpha.state_dict()
            else:
                best_nalphas, best_factors, best_nbetas, best_bfactors = 0, 0, 0, 0
                state_optimizeralpha = 0
    if epoch % 5 == 0 or epoch == args.epochs-1:
        print('.....SAVING.....')
        scheduler_state = scheduler.state_dict()
        if args.mscale != 'normal':
            nalphas, factors, nbetas, bfactors = alpha_parameters()
            current_optimizeralpha = optimizeralpha.state_dict()
        else:
            nalphas, factors, nbetas, bfactors = 0, 0, 0, 0
            current_optimizeralpha = 0
        state = {
            'test_prop': vars(args),
            'seed': seed,
            'indices': indices,
            'best_net': state_net,
            'best_epoch': best_epoch,
            'best_test': test_best,
            'best_train': train_best,
            'best_val': val_best,
            'best_nalpha': best_nalphas,
            'best_factors': best_factors,
            'best_nbetas': best_nbetas,
            'best_bfactors': best_bfactors,
            'all_train_acc': train_a,
            'all_val_acc': val_a,
            'all_test_acc': test_a,
            'all_train_loss': train_l,
            'all_val_loss': val_l,
            'all_test_loss': test_l,
            'optimizernet_best': state_optimizernet,
            'optimizeralpha_best': state_optimizeralpha,
            'scheduler' : scheduler_state,            
            'current_net' : net.state_dict(),
            'current_epoch' : epoch,
            'current_optimizer': optimizernet.state_dict(),
            'current_optimizeralpha': current_optimizeralpha,
            'nalpha': nalphas,
            'factors': factors,
            'nbetas': nbetas,
            'bfactors': bfactors,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        save_name = './checkpoint/ckpt_' + str(test_name) + '.t7'
        torch.save(state, save_name)

print('best accuracy', test_best, 'at epoch', best_epoch, 'train accuracy', train_best)
# print(state)
print(datetime.now() - startTime)











