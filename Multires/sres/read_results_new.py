import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as anim

plt.close('all')
test_name = 8685
test_5 = 0
test_6 = 0
in_epoch = 20
pool = 0
print_interval = 5
pro = 0
# pool = [1,2,3,4,4,4,4,4,4,4] # 8307
# pool = [1,1,2,2,2,3,3,4,4,4] #8308
# pool = [1,2,3,2,1,1,1,1,1,1] #8309
# pool = [1,1,2,2,3,3,4,4,4,4] #8310
# pool = [1,2,2,3,3,3,3,4,4,4] #8322
# pool = [1,2,3,3,3,3,3,4,4,4] #8323
# pool = [1,2,2,1,3,4,4,4,4,4] #8336
# pool = [1,2,3,3,3,3,4,4,4,4] #8401
# pool = [1,2,3,3,3,3,3,4,4,4] #8400
# pool = [1,2,2,2,3,3,3,4,4,3] #8399
# pool = [1,2,3,3,3,3,4,4,4,4] #8398
# pool = [1,2,3,3,3,3,4,4,4,4] #8397
# pool = [1,2,4,3,4,1,4,4,4,4] #8396
# pool = [1,2,3,3,3,3,3,3,4,4] #8395
# pool = [1,4,3,4,1,4,1,4,4,4] #8394
# pool = [1,4,3,4,1,4,1,4,4,4] #8394
# pool = [1,2,3,3,3,3,4,4,4,4] #8406
# pool = [1,1,2,2,3,3,3,1,4,4] #8415


def string_to_list(x):
    x = x.split(',')
    res = [int(i) for i in x]
    return res

def plotting(test_name, in_epoch=0, factor=1, pro=0, test_5=0, test_6=0, pool=0):
    if pro:
        checkpoint = torch.load('./checkpoint_pro/ckpt_pro_' + str(test_name) + '.t7')
    else:
        checkpoint = torch.load('./checkpoint/ckpt_' + str(test_name) + '.t7')
    test_properties = checkpoint['test_properties']
    print(test_properties)
    best_epoch = checkpoint['best_epoch']
    loss_progress = checkpoint['loss_progress']
    accuracy_progress = checkpoint['accuracy_progress']
    alpha_progress = checkpoint['alpha_progress']
    best_alpha = checkpoint['best_alpha']
    best_accuracy = checkpoint['best_accuracy']
    epoch = checkpoint['epoch']
    ##########
    print(best_accuracy)


    if in_epoch:
        in_epoch = in_epoch
        c_epoch = in_epoch // print_interval - 1 #index
        alpha = alpha_progress[c_epoch]

    else:
        in_epoch = best_epoch
        alpha = best_alpha
        c_epoch = best_epoch // print_interval - 1  #index
    print(c_epoch)
    print('epoch ', in_epoch,' from ', epoch)
    print(alpha)
    print(len(alpha_progress))
    # print(alpha_progress[c_epoch])
    # print(alpha_progress[0])

    if pool:
        n = []
        for i in pool:
            n.append(np.eye(4)[i - 1])
        m = [j.tolist() for j in n]
        nalpha = np.array(m)

    fig_name = str(test_name) + '_' + str(in_epoch)
    # plt.close('all')

    prop = accuracy_progress
    x = [i for i in range(0,epoch) if not i%print_interval]
    x = np.array(x)
    # x = np.arange(0, epoch+1, 5)
    fig1 = plt.figure(1)
    plt.title('accuracy vs epoch ' + str(test_name))
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.ylim(bottom=30, top=100)
    plt.plot(x, np.asarray(prop['train']) * 100.0, 'b', label='train')
    plt.plot(x, np.asarray(prop['validation']) * 100.0, 'g', label='validation')
    plt.plot(x, np.asarray(prop['test'])* 100.0, 'r', label='test')
    plt.minorticks_on()
    plt.grid(which='both')
    plt.grid(True)
    plt.grid(b=True, which='minor', linestyle='-', alpha=0.2)
    plt.legend()
    plt.show()
    print('last test accuracy:', prop['test'][-1])

    prop = loss_progress
    print(len(prop['test']))
    print(prop['train'][-1])
    # pt = [np.mean(i) for i in prop['train']]
    fig1 = plt.figure(2)
    plt.title('loss vs epoch ' + str(test_name))
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(x, np.asarray(prop['train']), 'b', label='train')
    # plt.plot(x, np.asarray(pt), 'b', label='train')
    plt.plot(x, np.asarray(prop['validation']), 'g', label='validation')
    plt.plot(x, np.asarray(prop['test']), 'r', label='test')
    plt.minorticks_on()
    plt.grid(which='both')
    plt.grid(True)
    plt.grid(b=True, which='minor', linestyle='-', alpha=0.2)
    plt.legend()
    plt.show()

    prop = accuracy_progress
    test_title = 'test ' + str(test_name) +  '\n' + test_properties['dataset'] + ', ' + 'sres'  + '\n tr: ' +\
                 str(round(prop['train'][c_epoch]*100,2)) + ', val: ' + str(round(prop['validation'][c_epoch]*100, 2)) +\
                 ', test: ' + str(round(prop['test'][c_epoch]*100, 2)) + '\n epoch ' + str(in_epoch) + '/' + str(epoch)
    layers = test_properties['leng']
    res = test_properties['max_scales']
    # res = 4
    if test_5 or test_6:
        res = test_properties['max_scales']   + 1
    alpha = torch.FloatTensor(alpha)
    #####################
    if test_6:
        alpha[0,-1] = -20000
        alpha[-1,-1] = -20000

    nalpha = np.array(F.softmax(alpha , 1))

    # nalpha = np.array(alpha)
    ####
    if pro:
        nn = []
        for i in range(layers-len(nalpha)):
            r = selected_res[i]
            nn.append(np.eye(res)[r-1].tolist())
        for j in nalpha:
            nn.append(j.tolist())
        nalpha = np.array(nn)

    if pool:
        n = []
        for i in pool:
            n.append(np.eye(4)[i-1])
        m = [j.tolist() for j in n]
        nalpha = np.array(m)
    print(nalpha)
    print(nalpha)
    nalpha = np.round_(nalpha, decimals=2)
    plt.ioff()
    fig2 = plt.figure(figsize=[2.5, 4.8])
    gs1 = gridspec.GridSpec(layers, res)
    ax1 = fig2.add_subplot(gs1[:, :])
    im1 = ax1.imshow(nalpha, cmap='Blues', vmin=0, vmax=1)
    # im1 = ax1.imshow(nalpha, cmap='Greens')#, vmin=0, vmax=1)
    ax1.set_xticks(np.arange(res))
    ax1.set_yticks(np.arange(layers))
    ax1.set_yticklabels(np.arange(layers) + 1)
    ax1.set_xticklabels(np.arange(res) + 1)
    ax1.xaxis.tick_top()
    ax1.set_xlabel('resolutions', fontsize=10)
    ax1.set_ylabel('layers', fontsize=10)
    ax1.xaxis.set_label_position("top")

    ax1.set_title(test_title, fontsize=10)

    for i in range(layers):
        for j in range(res):
            text1 = ax1.text(j, i, nalpha[i][j], ha="center", va="center", color="Black", fontsize=8)
    plt.show()

    fig2.savefig(fig_name)
    plt.close('all')
    print(best_alpha)
    print(best_alpha)
    for k, v in checkpoint['best_model'].items():
        if 'alpha' in k:
            print(k)
            print(v)

checkpoint1 = plotting(test_name,in_epoch=in_epoch,pro=pro, test_5=test_5, test_6 = test_6, pool=pool)

