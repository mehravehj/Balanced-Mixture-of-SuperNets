import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as anim

plt.close('all')
layer = 0
channel = 2
image = 3



def string_to_list(x):
    x = x.split(',')
    res = [int(i) for i in x]
    return res

def plotting(layer,image, channel):
    layers = 10
    res = 4
    nalpha = []
    targets = torch.load('./checkpoint/layer_target_.t7')
    targets = torch.cat(targets)
    targets = targets.numpy()
    print('class ind')
    class_ind = np.where(targets==3)[0]
    print(class_ind)
    for layer in range(layers):
        checkpoint = torch.load('./checkpoint/layer_std_' + str(layer) + '.t7')
        # print(checkpoint[0].size())
        checkpoint = torch.cat(checkpoint, -1)
        checkpoint = checkpoint[:, class_ind]
        print(checkpoint.size())
        # alpha = torch.mean(checkpoint, -1)
        alpha = checkpoint
        print(alpha.size())

        # alpha = checkpoint[:, image, channel]
        nalpha.append(alpha.detach().data.cpu())
    nalpha = (torch.stack(nalpha, 0)[:,:,0])
    print(nalpha.size())
    # print(alpha.size())
    # alpha = torch.FloatTensor(alpha)
    # print(nalpha)
    nalpha = np.array(nalpha)
    # print(nalpha)
    # print(nalpha)
    nalpha = np.round_(nalpha, decimals=2)
    plt.ioff()
    fig2 = plt.figure(figsize=[2.5, 4.8])
    gs1 = gridspec.GridSpec(layers, res)
    ax1 = fig2.add_subplot(gs1[:, :])
    im1 = ax1.imshow(nalpha, cmap='Purples')#, vmin=0, vmax=1)
    # im1 = ax1.imshow(nalpha, cmap='Greens')#, vmin=0, vmax=1)
    ax1.set_xticks(np.arange(res))
    ax1.set_yticks(np.arange(layers))
    ax1.set_yticklabels(np.arange(layers) + 1)
    ax1.set_xticklabels(np.arange(res) + 1)
    ax1.xaxis.tick_top()
    ax1.set_xlabel('resolutions', fontsize=10)
    ax1.set_ylabel('layers', fontsize=10)
    ax1.xaxis.set_label_position("top")

    # ax1.set_title(test_title, fontsize=10)

    for i in range(layers):
        for j in range(res):
            text1 = ax1.text(j, i, nalpha[i][j], ha="center", va="center", color="Black", fontsize=8)
    plt.show()
    # plt.close('all')

checkpoint1 = plotting(layer,image, channel)

