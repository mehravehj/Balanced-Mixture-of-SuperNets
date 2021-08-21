import numpy as np
import torch
import torch.nn as nn
import os
import argparse
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from func_multi_simp import data_loader
from func_multi_simp import sConv2d
from func_multi_simp import normal_net
from datetime import datetime
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


'''
    1. load the state dict
    2. calculate L2 of each filter
    3. visulalize
'''

plt.close('all')
test_name = 6007
depth = 10
res = 4

checkpoint = torch.load('./checkpoint/ckpt_' + str(test_name) + '.t7')
state_net = checkpoint['best_net']
try:
    state_net = state_net[0]#TODO: some runs need this
except:
    pass

weights_l2 = np.zeros((10,4))

w = state_net['module.conv.9.conv.3.weight']
print(w)
# l2 = torch.norm(w)
print(w.size())
# print(l2)
for l in range(depth):
    for r in range(res):
        weight_key = 'module.conv.' + str(l) + '.conv.' + str(r) + '.weight'
        weights_l2[l,r] = torch.norm(state_net[weight_key])
        weights_l2[l, r] = round(weights_l2[l,r],2)
print(weights_l2)


def plot_res(test_name, weights_l2):
    checkpoint = torch.load('./checkpoint/ckpt_' + str(test_name) + '.t7')
    test_prop = checkpoint['test_prop']
    best_epoch = checkpoint['best_epoch']
    test_best = checkpoint['best_test']
    train_best = checkpoint['best_train']
    val_best = checkpoint['best_val']
    best_nalphas = checkpoint['best_nalpha']
    best_factors = checkpoint['best_factors']
    best_nbetas = checkpoint['best_nbetas']
    best_bfactors = checkpoint['best_bfactors']
    train_a = checkpoint['all_train_acc']
    val_a = checkpoint['all_val_acc']
    test_a = checkpoint['all_test_acc']
    train_l = checkpoint['all_train_loss']
    val_l = checkpoint['all_val_loss']
    best_nalpha = np.array(best_nalphas)
    dense = test_prop['dense']
    res = test_prop['max_scales']
    #    res = len(best_nalphas[0])

    layers = test_prop['leng']
    factors = best_factors
    beta_factor = best_bfactors
    # print(len(best_nalphas))
    # print(best_nalpha)
    print(test_prop)
    print('test name:', test_prop['testname'])
    print('dataset: ', test_prop['dataset'])
    print('model:', test_prop['mscale'])
    print('validation percent:', test_prop['vp'])
    print('test:', test_best)
    print('validation:', val_best)
    print('train:', train_best)
    print('epoch:', best_epoch)
    print('total epochs:', test_prop['epochs'])
    print('lr:', test_prop['lr'], '-', test_prop['lr_min'])
    #    print('scales:', test_prop['max_scales'])
    print('pooling:', test_prop['pooling'])
    print('same filters:', test_prop['usf'])
    print('alr - awd:', test_prop['alr'], '-', test_prop['awd'])
    print(max(test_a))
    print('seed:', checkpoint['seed'])
    x = np.arange(len(test_a))
    print('epochs', 600 - len(test_a))
    #    print(best_nalpha)
    # if test_prop['mscale'] == 'normal':
    #     pooling = test_prop['pooling']
    #     factors = []
    #     if pooling:
    #         pp = [int(i) for i in pooling if i != ',']
    #         pp.append(10)
    #         pp.insert(0, 0)
    #         mm = 0
    #         zz = []
    #         for i in range(len(pp)):
    #             oo = pp[i] - pp[i - 1]
    #             for i in range(oo):
    #                 zz.append(mm - 1)
    #
    #             mm += 1
    #         print('zz', zz)
    #         res = max(zz) + 1
    #         nalpha = []
    #         ie = np.eye(res)
    #         for i in zz:
    #             nalpha.append(ie[i].tolist())
    #         print(nalpha)
    #     else:
    #         res = 1
    #         nalpha = [[1] for i in range(10)]
    #
    #
    # else:
    #     yy = []
    #     tt = []
    #     for i in range(layers):
    #         yy = []
    #         for j in range(res):
    #             xx = best_nalpha[i][j][0]
    #             yy.append(xx)
    #         tt.append(np.array(yy))
    #     # print('nalpha')
    #     # print(tt)
    #     nalpha = [l.tolist() for l in tt]
    #     nalpha = np.array(nalpha).T
    #     nal = np.round_(nalpha, decimals=2)
    #     nalpha = nal.T.tolist()
    #        print(factors)

    # nalpha = best_nalphas

    if 1:
        plt.ioff()
        fig2 = plt.figure()

        #        print(nalpha)
        #         if dense:
        #             beta_factor = checkpoint['beta_factor']
        #             text3 = 'beta \n (factor: \n %.2f )' %(beta_factor[0])
        #         else:
        #             text3 = ' '
        #         fig2, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(res,res*layers))
        #         gs1 = gridspec.GridSpec(layers, res+2)
        gs1 = gridspec.GridSpec(layers, res)
        #     gs1.update(left=0.1, right=1)
        #     gs1.update(left=0, right=1, wspace=0, hspace=0)
        #     ax1 = plt.subplot(gs1[:, :-2])
        ax1 = fig2.add_subplot(gs1[:, :])
        # ax2 = plt.subplot(gs1[:, -2])
        # ax3 = plt.subplot(gs1[:, -1])
        # print(nalpha)
        im1 = ax1.imshow(weights_l2, cmap='Purples')#, vmin=0, vmax=1)
        ax1.set_xticks(np.arange(res))
        ax1.set_yticks(np.arange(layers))
        ax1.set_yticklabels(np.arange(layers) + 1)
        ax1.set_xticklabels(np.arange(res) + 1)
        ax1.xaxis.tick_top()
        ax1.set_xlabel('resolutions', fontsize=10)
        ax1.set_ylabel('layers', fontsize=10)
        ax1.xaxis.set_label_position("top")
        if test_prop['usf']:
            filt = 'same filetrs'
        else:
            filt = 'different filters'
        test_title = 'Filter L2, test ' + str(test_name) + '\n' + test_prop['dataset'] + ', ' + filt + '\n tr: ' + str(round(train_best,2))  + ', val: ' + str(round(val_best,2)) + ', test: ' + str(round(test_best,2)) + '\n epoch ' + str(checkpoint['best_epoch']) + '/' + str(checkpoint['current_epoch'])

        ax1.set_title(test_title, fontsize=10)

        for i in range(layers):
            for j in range(res):
                text1 = ax1.text(j, i, weights_l2[i][j], ha="center", va="center", color="Black", fontsize=8)
        if not test_prop['lf']:
            factors = np.ones((layers, 1))

        plt.show()
    fig_name = str(test_name) + '_' + str(checkpoint['current_epoch']) + '_filter_norm'
    fig2.savefig(fig_name)
    plt.close('all')



#    return checkpoint
checkpoint1 = plot_res(test_name, weights_l2)