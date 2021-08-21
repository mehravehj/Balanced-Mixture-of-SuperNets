import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as anim

plt.close('all')
test_name = 8026


checkpoint = torch.load('./checkpoint/ckpt_' + str(test_name) + '.t7')
for k,v in checkpoint.items():
    print(k)

train_loss = checkpoint['loss_progress']['train']
t_loss = []
for i in train_loss:
    # l = [j.cpu().item() for j in i]
    l = [j for j in i]
    t_loss.append(l)

t_l = [item for sl in t_loss for item in sl]

valid_loss = checkpoint['loss_progress']['validation']
v_loss = []
try:
    for i in valid_loss:
        # print(i)
        l = [j.cpu().item() for j in i]
        v_loss.append(l)
    v_l = [item for sl in v_loss for item in sl]
except:
    pass
# print(t_l)

fig1 = plt.figure(1)
plt.title('loss vs epoch ' + str(test_name))
plt.xlabel('epoch')
plt.ylabel('loss')
plt.ylim(bottom=0, top=2.5)
plt.plot(t_l, 'b', label='train')
# plt.plot(t_l, 'g', label='valid')
# plt.plot(x, np.asarray(prop['validation']) * 100.0, 'g', label='validation')
# plt.plot(x, np.asarray(prop['test']) * 100.0, 'r', label='test')
plt.minorticks_on()
plt.grid(which='both')
plt.grid(True)
plt.grid(b=True, which='minor', linestyle='-', alpha=0.2)
plt.legend()
plt.show()
# print('last test accuracy:', prop['test'][-1])


def string_to_list(x):
    x = x.split(',')
    res = [int(i) for i in x]
    return res

def plotting(test_name, in_epoch=0, factor=1, pro=0):
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
    # print(best_alpha)
    ee = -1
    print(alpha_progress[ee])
    if pro:
        current_layer = checkpoint['current_layer']
        selected_res = checkpoint['selected_res']
        print(current_layer)
        print(selected_res)
    print(best_accuracy)
    if 0:
    # if in_epoch:
        in_epoch = 170
        c_epoch = in_epoch // 5
        alpha = alpha_progress[c_epoch]
        # alpha = alpha_progress[20]
    else:
        in_epoch = best_epoch
        alpha = best_alpha
        c_epoch = best_epoch //5
    alpha = alpha_progress[ee]
    print('epoch ', in_epoch,' from ', epoch)
    fig_name = str(test_name) + '_' + str(in_epoch)
    # plt.close('all')

    prop = accuracy_progress
    x = np.arange(1, epoch+1)
    # x = np.arange(0, epoch*2+1, 10)
    # print(prop['test'][-20:])
    fig1 = plt.figure(1)
    plt.title('accuracy vs epoch ' + str(test_name))
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.ylim(bottom=40, top=100)
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

    # prop = loss_progress
    # fig1 = plt.figure(2)
    # plt.title('loss vs epoch ' + str(test_name))
    # plt.xlabel('epoch')
    # plt.ylabel('accuracy')
    # plt.plot(x, np.asarray(prop['train']) * 100.0, 'b', label='train')
    # plt.plot(x, np.asarray(prop['validation']) * 100.0, 'g', label='validation')
    # plt.plot(x, np.asarray(prop['test'])* 100.0, 'r', label='test')
    # plt.minorticks_on()
    # plt.grid(which='both')
    # plt.grid(True)
    # plt.grid(b=True, which='minor', linestyle='-', alpha=0.2)
    # plt.legend()
    # plt.show()


    # test_title = 'test ' + str(test_name+20) +  '\n' + test_properties['dataset'] + ', ' + test_properties['mscale']  + '\n tr: ' + str(round(prop['train'][c_epoch]*100,2)-0.5) + ', val: ' + str(round(prop['validation'][c_epoch]*100, 2)-0.32) + ', test: ' + str(round(prop['test'][c_epoch]*100, 2)-0.5) + '\n epoch ' + str(in_epoch+25) + '/' + str(epoch+100)
    # test_title = 'test ' + str(7092) +  '\n' + test_properties['dataset'] + ', ' + test_properties['mscale']  + '\n tr: ' + str(48.23) + ', val: ' + str(round(0, 2)) + ', test: ' + str(round(43.14, 2)) + '\n epoch ' + str(325) + '/' + str(500)
    # test_title = 'test ' + str(7085) +  '\n' + test_properties['dataset'] + ', ' + test_properties['mscale']  + '\n tr: ' + str(47.18,2)) + ', val: ' + str(round(0, 2)) + ', test: ' + str(round(42.67, 2)) + '\n epoch ' + str(319) + '/' + str(480)
    net_type = test_properties['net_type']
    block_type = test_properties['mscale']
    pooling = test_properties['pooling']
    initial_alpha = test_properties['initial_alpha']
    layers = test_properties['leng']
    res = test_properties['max_scales'] #+ 1
    # res = 5

        # nalpha = nalpha.tolist()
        # print(nalpha)

    if net_type == 'multires':
        # print(best_alpha)
        alpha = torch.FloatTensor(alpha)
        # alpha[3,4] += 3
        # alpha[2, 4] += -1
        # alpha[4, 4] += 1
        # alpha = torch.abs(alpha) #uncomment for absolute alpha tests
        nalpha = np.array(F.softmax(alpha, 1))

        # print(nalpha)
        # print(len(nalpha))
        ####
        if pro:
            # selected_res.append(5)
            # selected_res.append(5)
            # nalpha = nalpha[2:]
            nn = []
            for i in range(layers-len(nalpha)):
                r = selected_res[i]
                nn.append(np.eye(res)[r-1].tolist())
            for j in nalpha:
                nn.append(j.tolist())
            nalpha = np.array(nn)

        # nalpha = [[2.4, 46.17,70.76, 70.19, 70.32],
        #           [14.78, 69.89, 69.53, 69.73, 69.99],
        #           [63.96, 46.34, 65.4, 64.28, ]
        #           ]
        # nalpha = np.array([[2.5, 48.16, 48.2, 48.22, 48.28],
        #           [10.62, 47.7, 47.88, 48.38, 48.7],
        #           [44.22, 35.9, 46.24, 45.68, 46.14],
        #           [47.06, 39.32,45.32, 47.94, 48.38],
        #           [47.84, 47.1, 47.92, 48.02, 48.08],
        #           [47.92, 47.84, 47.16, 41.22, 48.18],
        #           [47.98, 48.04, 45.52, 42.16, 46.8],
        #           [46.46, 46.36, 39.44, 41.44, 44.12],
        #           [44.04, 44, 43.48, 42.76, 30.12],
        #           [30.08, 30.12, 30.18, 30.0, 22.6]
        #           ])

        # nalpha = nalpha
        # nalpha = np.array([[0.2069, 0.2117, 0.2072, 0.1971, 0.1771],
        #         [0.1975, 0.1978, 0.2028, 0.2032, 0.1987],
        #         [0.1990, 0.1993, 0.1994, 0.2022, 0.2001],
        #         [0.2008, 0.1965, 0.2015, 0.2040, 0.1973],
        #         [0.2005, 0.1950, 0.2017, 0.2018, 0.2010],
        #         [0.1980, 0.1982, 0.1986, 0.2001, 0.2051],
        #         [0.1997, 0.1978, 0.1984, 0.2015, 0.2026],
        #         [0.1997, 0.1990, 0.1996, 0.1996, 0.2020],
        #         [0.1964, 0.1998, 0.2006, 0.2012, 0.2019],
        #         [0.2000, 0.1998, 0.2001, 0.1983, 0.2019]])
        # nalpha = np.array([[1, 0, 0, 0, 0],
        #         [1, 0, 0, 0, 0],
        #         [0, 1, 0, 0, 0],
        #         [0, 0, 1, 0, 0],
        #         [0, 0, 0, 1, 0],
        #         [0, 0, 0, 0, 1],
        #         [0, 0, 0, 0, 1],
        #         [0, 0, 0, 0, 1],
        #         [0, 0, 0, 0, 1],
        #         [0, 0, 0, 0, 1]])
        #
        # nalpha = np.array([[1, 0, 0, 0, 0],
        #  [0.8,  0.07, 0.07, 0.03, 0.03],
        #  [0.13, 0.46, 0.27, 0.09, 0.05],
        #  [0.13, 0.19, 0.29, 0.24, 0.15],
        #  [0.28, 0.24, 0.23, 0.15, 0.1 ],
        #  [0.14, 0.13, 0.2, 0.34, 0.18],
        #  [0.15, 0.15, 0.14, 0.22, 0.35],
        #  [0.15, 0.15, 0.14, 0.21, 0.35],
        #  [0.12, 0.1,  0.11, 0.16, 0.51],
        #  [0.04, 0.03, 0.03, 0.06, 0.84]])

        ####
        # nalpha = np.array(nalpha)
        # print(nalpha)
        nalpha = np.round_(nalpha, decimals=2)
        print(nalpha)
    else:

        nalpha = []
        if pooling != '0':
            pooling = string_to_list(pooling)
            for l in range(layers):
                r = pooling[l]-1
                nalpha.append(np.eye(res)[r])
        else:
            for l in range(layers):
                nalpha.append(np.eye(res)[0])
    # nalpha = np.array([
    #     [14.1, 22.26, 23.52, 24.84, 25.26],
    #     [21.20, 23.4, 20.28, 20.24, 21.52],
    #     [16.88, 19.54, 18.16, 18.18, 15.9],
    #     [17.96, 17.1, 15.0, 12.44, 12.44],
    #     [12.34, 12.24, 12.18, 11.74, 11.3],
    #     [11.8, 11.66, 11.6, 11.11, 10.20],
    #     [10.36, 10.24, 10.14, 10.04, 9.84],
    #     [9.72, 9.78, 9.84, 10.28, 9.18],
    #     [9.8, 9.7, 9.52, 9.2, 8.58],
    #     [8.46, 8.44, 8.5, 8.5, 8.3],
    # ])
    # nalpha = np.array([
    #     [15.38, 31.58, 30.12, 30.74, 30.54],
    #     [15.04, 31.3, 32.06, 33.18, 31.9],
    #     [32.06, 29.52, 28.98, 32.1, 32.44],
    #     [31.8, 31.0, 26.6, 31.82, 30.25],
    #     [32.18, 32.74, 30.98, 31.96, 32.18],
    #     [30.74, 29.64, 30.98, 31.44, 31.18],
    #     [31.58, 31.18, 29.92, 32.68, 32.28],
    #     [31.44, 30.56, 31.46, 30.66, 31.52],
    #     [30.56, 31.4, 31.24, 30.78, 31.54],
    #     [30.88, 30.21, 31.56, 30.15, 20.65]
    #     ])
    plt.ioff()
    fig2 = plt.figure(figsize=[2.5, 4.8])
    gs1 = gridspec.GridSpec(layers, res)
    ax1 = fig2.add_subplot(gs1[:, :])
    im1 = ax1.imshow(nalpha, cmap='Blues', vmin=0, vmax=1)
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

    fig2.savefig(fig_name)
    plt.close('all')

# for t in range(test_name,test_name+100):
#     try:
#         checkpoint1 = plotting(t)
#     except: pass



checkpoint1 = plotting(test_name,factor=1,pro=0)