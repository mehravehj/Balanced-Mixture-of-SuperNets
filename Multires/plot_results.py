import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as anim

plt.close('all')
test_name = 5088
epoch_plot = 120

def plot_res(test_name, epoch_plot):
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
    if epoch_plot:
        print('plotting best epoch')
        best_nalpha = np.array(best_nalphas)
        best_epoch = checkpoint['best_epoch']
    # else:
    #     print('plotting prev epoch')
    #     all_nalphas = checkpoint['all_nalphas']
    #     idx = int(epoch_plot/10)
    #     print('index',idx)
    #     print(all_nalphas[0][0])
    #     print(all_nalphas[10][0])
    #     print(all_nalphas[40][0])
    #     print(all_nalphas[100][0])
    #     best_nalpha = np.array(all_nalphas[idx])
    #     best_epoch = epoch_plot

    # best nalphas = [[0.69, 0.08, 0.06, 0.05, 0.05, 0.07], [0.05, 0.08, 0.16, 0.2, 0.23, 0.27], [0.1, 0.11, 0.12, 0.17, 0.15, 0.35], [0.14, 0.13, 0.13, 0.14, 0.16, 0.3], [0.12, 0.13, 0.13, 0.14, 0.14, 0.34], [0.22, 0.22, 0.19, 0.16, 0.12, 0.09]]
    dense = test_prop['dense']
    res = test_prop['max_scales']
    #    res = len(best_nalphas[0])

    layers = test_prop['leng']
    if test_prop['mscale'] == 'sres':
        # layers = layers-2
        conv_type = 'ResNet block'
    else:
        conv_type = ' '
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
    if test_prop['mscale'] == 'normal':
        pooling = test_prop['pooling']
        factors = []
        if pooling:
            pp = [int(i) for i in pooling if i != ',']
            pp.append(layers)
            pp.insert(0, 0)
            mm = 0
            zz = []
            for i in range(len(pp)):
                oo = pp[i] - pp[i - 1]
                for i in range(oo):
                    zz.append(mm - 1)

                mm += 1
            print('zz', zz)
            res = max(zz) + 1
            nalpha = []
            ie = np.eye(res)
            for i in zz:
                nalpha.append(ie[i].tolist())
            print(nalpha)
        else:
            res = 1
            nalpha = [[1] for i in range(10)]


    else:
        yy = []
        tt = []
        for i in range(layers):
            yy = []
            for j in range(res):
                xx = best_nalpha[i][j][0]
                yy.append(xx)
            tt.append(np.array(yy))
        # print('nalpha')
        # print(tt)
        nalpha = [l.tolist() for l in tt]
        nalpha = np.array(nalpha).T
        nal = np.round_(nalpha, decimals=2)
        nalpha = nal.T.tolist()
        # nalpha = [[0.86, 0.08, 0.02, 0.01, 0.01, 0.01], [0.05, 0.08, 0.21, 0.3, 0.20, 0.15], [0.08, 0.09, 0.10, 0.17, 0.21, 0.35], [0.02, 0.05, 0.11, 0.14, 0.17, 0.51], [0.02, 0.03, 0.03, 0.14, 0.14, 0.64], [0.01, 0.02, 0.09, 0.06, 0.1, 0.72]]
        # print(nalpha)
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
        print(nalpha)
        im1 = ax1.imshow(nalpha, cmap='Oranges', vmin=0, vmax=1)
        ax1.set_xticks(np.arange(res))
        ax1.set_yticks(np.arange(layers))
        ax1.set_yticklabels(np.arange(layers) + 1)
        ax1.set_xticklabels(np.arange(res) + 1)
        ax1.xaxis.tick_top()
        ax1.set_xlabel('resolutions', fontsize=10)
        ax1.set_ylabel('layers', fontsize=10)
        ax1.xaxis.set_label_position("top")
        if test_prop['usf']:
            filt = 'same filters'
        else:
            filt = 'different filters'
            # str(round(train_best, 2)) + ', val: ' + str(round(val_best,2)) + ', test: ' + str(round(test_best,2)) + '\n epoch ' + str(best_epoch) + '/' + str(checkpoint['current_epoch'])
        test_title = 'test ' + str(test_name) +  '\n' + test_prop['dataset'] + ', ' + conv_type + ', ' + filt + '\n tr: ' + str(99.56) + ', val: ' + str(81.31) + ', test: ' + str(82.15) + '\n epoch ' + str(585) + '/' + str(600)
        # test_title = 'test ' + str(test_name) + '\n' + test_prop[
        #     'dataset'] + ', ' + conv_type + ', ' + filt + '\n tr: ' + str(round(train_best, 2)) + ', val: ' + str(
        #     round(val_best, 2)) + ', test: ' + str(74.55) + '\n epoch ' + str(
        #     567) + '/' + str(600)

        ax1.set_title(test_title, fontsize=10)

        for i in range(layers):
            for j in range(res):
                # print(layers)
                # print(res)
                text1 = ax1.text(j, i, nalpha[i][j], ha="center", va="center", color="Black", fontsize=8)
        if not test_prop['lf']:
            factors = np.ones((layers, 1))
        #            factors = np.expand_dims(np.array(factors), 1)
        #        print(factors)
        else:
            factors = np.round_(factors, decimals=2)
        #
        # im2 = ax2.imshow(factors,cmap='Oranges')
        # ax2.set_yticks([])
        # ax2.set_yticklabels([])
        # ax2.set_xticks([])
        # ax2.set_xticklabels([])
        # ax2.set_xlabel('alpha \n factor',fontsize=13)
        # ax2.xaxis.set_label_position("top")
        #
        # for i in range(layers):
        #     if test_prop['lf']:
        #         text2 = ax2.text(0,i, factors[i][0],ha="center", va="center", color="Black",fontsize=14)
        #
        # if dense:
        #     beta.insert(0,0)
        #     beta.insert(layers-1,0)
        #     beta = np.expand_dims(np.array(beta).T, 1)
        # else:
        #     beta =np.zeros((layers,1))
        # im3 = ax3.imshow(beta,cmap='Purples',vmin=0, vmax=1)
        # ax3.set_yticks([])
        # ax3.set_yticklabels([])
        # ax3.set_xticks([])
        # ax3.set_xticklabels([])
        # ax3.set_xlabel(text3,fontsize=13)
        # ax3.xaxis.set_label_position("top")
        # if beta_factor:
        #     for i in range(layers-2):
        #         text3 = ax3.text(0,i+1, beta[i+1,0],ha="center", va="center", color="Black",fontsize=14)
        plt.show()
    #         if 0:
    fig_name = str(test_name) + '_' + str(checkpoint['current_epoch'])
    fig2.savefig(fig_name)
    plt.close('all')
    print(len(x))

    fig1 = plt.figure(1)
    plt.title('Accuracy vs epoch ' + str(test_name))
    plt.xlabel('epoch')
    plt.ylabel('acuuracy')
    plt.ylim(bottom=0, top=100)
    plt.plot(x, train_a, 'b', label='train')
    plt.plot(x, val_a, 'g', label='validation')
    plt.plot(x, test_a, 'r', label='test')
    plt.minorticks_on()
    plt.grid(which='both')
    plt.grid(True)
    plt.grid(b=True, which='minor', linestyle='-', alpha=0.2)
    plt.legend()
    plt.show()


#    return checkpoint
checkpoint1 = plot_res(test_name, epoch_plot)
# for t in range(test_name,test_name+100):
#     try:
#         checkpoint1 = plot_res(t, epoch_plot)
#     except: pass





# checkpoint1 = checkpoint1['best_net']
# weight1 = np.array(checkpoint1['module.conv.0.conv.weight'].cpu())
# test_name = 1695
# checkpoint2 = plot_res(test_name)
# checkpoint2 = checkpoint2['best_net']
# weight2 = np.array(checkpoint2['module.conv.0.conv.weight'].cpu())
# train_a1, val_a1, test_a1 = plot_res(test_name)
# train_a2, val_a2, test_a2 = plot_res(test_name+1)
# train_a3, val_a3, test_a3 = plot_res(test_name+2)
# train_a4, val_a4, test_a4 = plot_res(test_name+3)
# train_a5, val_a5, test_a5 = plot_res(test_name+4)
# train_a6, val_a6, test_a6 = plot_res(test_name+5)
#
# train_a = train_a1 + train_a2 + train_a3 + train_a4 #+ train_a5 + train_a6
# val_a = val_a1 + val_a2 + val_a3 + val_a4 #+ val_a5 + val_a6
# test_a = test_a1 + test_a2 + test_a3 + test_a4 #+ test_a5 + test_a6
# x = np.arange(len(test_a))


# fig1 = plt.figure(1)
## plt.title('Accuracy vs epoch')
## plt.xlabel('epoch')
## plt.ylabel('acuuracy')
# plt.ylim(bottom=0, top=100)
# plt.plot(x,train_a, 'b', label='train')
# plt.plot(x,val_a, 'g',label='validation')
# plt.plot(x,test_a, 'r',label='test')
# plt.grid(True)
## plt.legend()
# plt.show()

##import numpy as np
##import matplotlib.pyplot as plt
##import torch
##import torch.nn.functional as F
##import matplotlib.pyplot as plt
##import matplotlib.gridspec as gridspec 
##import matplotlib.animation as anim
##plt.close('all')
##
##test_name = 3017
##res = 4
##checkpoint = torch.load('./checkpoint/ckpt_'+str(test_name)+'.t7')
##test_prop = checkpoint['test_prop']
##best_epoch = checkpoint['best_epoch']
##test_best = checkpoint['best_test']
##train_best = checkpoint['best_train']
##val_best = checkpoint['best_val']
##best_nalphas = checkpoint['best_nalpha']
##best_factors=checkpoint['best_factors']
##best_nbetas = checkpoint['best_nbetas']
##best_bfactors = checkpoint['best_bfactors']
##train_a = checkpoint['all_train_acc']
##val_a = checkpoint['all_val_acc']
##test_a = checkpoint['all_test_acc']
##train_l = checkpoint['all_train_loss']
##val_l = checkpoint['all_val_loss']
##best_nalpha = np.array(best_nalphas)
##dense = test_prop['dense']
##
##layers = test_prop['leng']
##factors = best_factors
##beta_factor = best_bfactors 
### print(len(best_nalphas))
### print(best_nalpha)
### print(test_prop)
##print('dataset: ', test_prop['dataset'])
##print('test:', test_best)
##print('train:', train_best)
##print('epoch:', best_epoch)
##print('total epochs:', test_prop['epochs'])
##print('scales:', test_prop['max_scales'])
##print('modle:', test_prop['mscale'])
##print('pooling:', test_prop['pooling'])
##print('same filters:', test_prop['usf'])
##
##
##x = np.arange(len(test_a))
##yy = []
##tt = []
##for i in range(layers):
##    yy = []
##    for j in range(res):
##        xx = best_nalpha[i][j][0]
##        yy.append(xx)
##    tt.append(np.array(yy))
### print('nalpha')
### print(tt)
##nalpha = [l.tolist() for l in tt]
##nalpha = np.array(nalpha).T
##nal = np.round_(nalpha, decimals=2)
##nalpha = nal.T.tolist()
### print(nalpha)
##
##
### nalpha = best_nalphas
##
##if 1:
##    plt.ioff()
##    if dense:
##        beta_factor = checkpoint['beta_factor']
##        text3 = 'beta \n (factor: \n %.2f )' %(beta_factor[0])
##    else:
##        text3 = ' '
##    fig2, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(res,res*layers))
##    gs1 = gridspec.GridSpec(layers, res+2)
###     gs1.update(left=0.1, right=1)
##    gs1.update(left=0, right=1, wspace=0, hspace=0)
##    ax1 = plt.subplot(gs1[:, :-2])
##    ax2 = plt.subplot(gs1[:, -2])
##    ax3 = plt.subplot(gs1[:, -1])
##
##    im1 = ax1.imshow(nalpha,cmap='Greens',vmin=0, vmax=1)
##    ax1.set_xticks(np.arange(res))
##    ax1.set_yticks(np.arange(layers))
##    ax1.set_yticklabels(np.arange(layers)+1)
##    ax1.set_xticklabels(np.arange(res)+1)
##    ax1.xaxis.tick_top()
##    ax1.set_xlabel('resolutions',fontsize=13)
##    ax1.set_ylabel('layers',fontsize=13)
##    ax1.xaxis.set_label_position("top")
##
##    for i in range(layers):
##        for j in range(res):
##            text1 = ax1.text(j, i, nalpha[i][j],ha="center", va="center", color="Black",fontsize=14)
##    factors = np.expand_dims(np.array(factors), 1)
##    im2 = ax2.imshow(factors,cmap='Oranges')
##    ax2.set_yticks([])
##    ax2.set_yticklabels([])
##    ax2.set_xticks([])
##    ax2.set_xticklabels([])
##    ax2.set_xlabel('alpha \n factor',fontsize=13)
##    ax2.xaxis.set_label_position("top")
##
##    for i in range(layers):
##        if test_prop['lf']:
##            text2 = ax2.text(0,i, factors[i,0],ha="center", va="center", color="Black",fontsize=14)
##
##    if dense:
##        beta.insert(0,0) 
##        beta.insert(layers-1,0)
##        beta = np.expand_dims(np.array(beta).T, 1)
##    else:
##        beta =np.zeros((layers,1))
##    im3 = ax3.imshow(beta,cmap='Purples',vmin=0, vmax=1)
##    ax3.set_yticks([])
##    ax3.set_yticklabels([])
##    ax3.set_xticks([])
##    ax3.set_xticklabels([])
##    ax3.set_xlabel(text3,fontsize=13)
##    ax3.xaxis.set_label_position("top")
##    if beta_factor:
##        for i in range(layers-2):
##            text3 = ax3.text(0,i+1, beta[i+1,0],ha="center", va="center", color="Black",fontsize=14)
##    plt.show()
###         if 0:
##
##plt.close('all')
##
##
##
##fig1 = plt.figure(1)
### plt.title('Accuracy vs epoch')
### plt.xlabel('epoch')
### plt.ylabel('acuuracy')
### # plt.ylim(bottom=70, top=100)
##plt.plot(x,train_a, 'b', label='train')
### # plt.plot(x,all_acc_val, 'g',label='validation')usf
##plt.plot(x,test_a, 'r',label='test')
### plt.grid(True)
### plt.legend()
##plt.show()
#
# import numpy as np
# import matplotlib.pyplot as plt
# import torch
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# import matplotlib.animation as anim
# plt.close('all')
#
# test_name = 2521
# checkpoint = torch.load('./checkpoint/ckpt_'+str(test_name)+'.t7')
# test_prop = checkpoint['test_prop']
# best_epoch = checkpoint['best_epoch']
# test_best = checkpoint['best_test']
# train_best = checkpoint['best_train']
# val_best = checkpoint['best_val']
# best_nalphas = checkpoint['best_nalpha']
# best_factors=checkpoint['best_factors']
# best_nbetas = checkpoint['best_nbetas']
# best_bfactors = checkpoint['best_bfactors']
# train_a = checkpoint['all_train_acc']
# val_a = checkpoint['all_val_acc']
# test_a = checkpoint['all_test_acc']
# train_l = checkpoint['all_train_loss']
# val_l = checkpoint['all_val_loss']
##test_l = checkpoint['all_test_loss']
# best_nalpha = np.array(best_nalphas)
# dense = test_prop['dense']
# res = test_prop['max_scales']
#
# layers = test_prop['leng']
# factors = best_factors
# beta_factor = best_bfactors
## print(len(best_nalphas))
## print(best_nalpha)
# print(test_prop)
# print('dataset: ', test_prop['dataset'])
# print('test:', test_best)
# print('train:', train_best)
# print('epoch:', best_epoch)
# print('total epochs:', test_prop['epochs'])
# print('scales:', test_prop['max_scales'])
# print('model:', test_prop['mscale'])
# print('pooling:', test_prop['pooling'])
# print('same filters:', test_prop['usf'])
#
##
# x = np.arange(len(test_a))
# yy = []
# tt = []
# for i in range(layers):
#    yy = []
#    for j in range(res):
#        xx = best_nalpha[i][j][0]
#        yy.append(xx)
#    tt.append(np.array(yy))
## print('nalpha')
## print(tt)
# nalpha = [l.tolist() for l in tt]
# nalpha = np.array(nalpha).T
# nal = np.round_(nalpha, decimals=2)
# nalpha = nal.T.tolist()
## print(nalpha)
#
#
# nalpha = best_nalphas
#
# if 1:
#    plt.ioff()
#    if dense:
#        beta_factor = checkpoint['beta_factor']
#        text3 = 'beta \n (factor: \n %.2f )' %(beta_factor[0])
#    else:
#        text3 = ' '
#    fig2, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(res,res*layers))
#    gs1 = gridspec.GridSpec(layers, res+2)
#     gs1.update(left=0.1, right=1)
#    gs1.update(left=0, right=1, wspace=0, hspace=0)
#    ax1 = plt.subplot(gs1[:, :-2])
#    ax2 = plt.subplot(gs1[:, -2])
#    ax3 = plt.subplot(gs1[:, -1])
#
#    im1 = ax1.imshow(nalpha,cmap='Greens',vmin=0, vmax=1)
#    ax1.set_xticks(np.arange(res))
#    ax1.set_yticks(np.arange(layers))
#    ax1.set_yticklabels(np.arange(layers)+1)
#    ax1.set_xticklabels(np.arange(res)+1)
#    ax1.xaxis.tick_top()
#    ax1.set_xlabel('resolutions',fontsize=13)
#    ax1.set_ylabel('layers',fontsize=13)
#    ax1.xaxis.set_label_position("top")
#
#    for i in range(layers):
#        for j in range(res):
#            text1 = ax1.text(j, i, nalpha[i][j],ha="center", va="center", color="Black",fontsize=14)
#    factors = np.expand_dims(np.array(factors), 1)
#    im2 = ax2.imshow(factors,cmap='Oranges')
#    ax2.set_yticks([])
#    ax2.set_yticklabels([])
#    ax2.set_xticks([])
#    ax2.set_xticklabels([])
#    ax2.set_xlabel('alpha \n factor',fontsize=13)
#    ax2.xaxis.set_label_position("top")
#
#    for i in range(layers):
#        if test_prop['lf']:
#            text2 = ax2.text(0,i, factors[i,0],ha="center", va="center", color="Black",fontsize=14)
#
#    if dense:
#        beta.insert(0,0) 
#        beta.insert(layers-1,0)
#        beta = np.expand_dims(np.array(beta).T, 1)
#    else:
#        beta =np.zeros((layers,1))
#    im3 = ax3.imshow(beta,cmap='Purples',vmin=0, vmax=1)
#    ax3.set_yticks([])
#    ax3.set_yticklabels([])
#    ax3.set_xticks([])
#    ax3.set_xticklabels([])
#    ax3.set_xlabel(text3,fontsize=13)
#    ax3.xaxis.set_label_position("top")
#    if beta_factor:
#        for i in range(layers-2):
#            text3 = ax3.text(0,i+1, beta[i+1,0],ha="center", va="center", color="Black",fontsize=14)
#    plt.show()
##         if 0:
#
# plt.close('all')
#
#
#
# fig1 = plt.figure(1)
## plt.title('Accuracy vs epoch')
## plt.xlabel('epoch')
## plt.ylabel('acuuracy')
## # plt.ylim(bottom=70, top=100)
# plt.plot(x,train_a, 'b', label='train')
# plt.plot(x,val_a, 'g',label='validation')
# plt.plot(x,test_a, 'r',label='test')
## plt.grid(True)
## plt.legend()
# plt.show()
#
# fig1 = plt.figure(1)
## plt.title('Accuracy vs epoch')
## plt.xlabel('epoch')
## plt.ylabel('acuuracy')
## # plt.ylim(bottom=70, top=100)
# plt.plot(x,train_l, 'b', label='train')
##plt.plot(x,val_l, 'g',label='validation')
# plt.plot(x,test_l, 'r',label='test')
## plt.grid(True)
## plt.legend()
# plt.show()
