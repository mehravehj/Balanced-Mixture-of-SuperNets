import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as anim

plt.close('all')
test_name = 8418
test_5 = 0
test_6 = 0
factor = 1
pro = 0
in_epoch = 0
pool = 0
# pool = [1,2,3,4,4,4,4,4,4,4] # 8307
pool = [1,1,2,2,2,3,3,4,4,4] #8308
# pool = [1,2,3,2,1,1,1,1,1,1] #8309
# pool = [1,1,2,2,3,3,4,4,4,4] #8310
# pool = [1,2,2,3,3,3,3,4,4,4] #8322
# pool = [1,2,3,3,3,3,3,4,4,4] #8323
# pool = [1,2,2,1,3,4,4,4,4,4] #8336
# pool = [1,1,1,1,1,1,1,1,1,1] #8371
# pool = [1,2,2,2,2,1,1,1,1,3] #8372
# pool = [1,2,3,3,3,1,1,1,2,3] #8373
# pool = [1,1,2,2,3,3,3,1,4,4] #8415
# pool = [2,1,1,2,2,3,3,3,4,4] #8416
# pool = [3,2,1,1,2,2,3,3,4,4] #8417
# pool = [1,1,1,2,2,3,3,4,4,4] #8419
# pool = [1,1,2,3,3,3,1,1,4,4] #8432
# pool = [1,2,2,3,3,3,3,4,4,4] #8431
# pool = [1,1,1,2,2,2,3,3,4,4] #8428
# pool = [1,1,1,1,2,2,3,3,4,4] #8429




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
    # print('best alpha')
    # print(best_alpha)
    # print('last_alpha')
    # print(alpha_progress[-1])
    if pro:
        current_layer = checkpoint['current_layer']
        selected_res = checkpoint['selected_res']
        print(current_layer)
        print(selected_res)
    print(best_accuracy)
    # print(checkpoint['best_model'])
    for k,v in checkpoint['best_model'].items():
        print(k)
    # for l in range(test_properties['leng']):
    #     k = 'conv.'+str(l)+'.alpha'
    #     print(checkpoint['best_model'][k])
    # print(checkpoint['best_model']['module.layer.0.block.alpha'])
    # print(checkpoint['model']['module.layer.0.block.alpha'])

    # print(checkpoint['best_model']['module.layer.0.block.alpha'])
    #
    # try:
    #     print(all(checkpoint['best_model']['module.layer.0.block.alpha']==checkpoint['model']['module.layer.0.block.alpha']))
    #     print(torch.all(
    #         checkpoint['best_model']['module.layer.0.block.block.conv1.weight'] == checkpoint['model']['module.layer.0.block.block.conv1.weight']))
    #     print(torch.all(
    #         checkpoint['best_model']['module.layer.8.block.block.conv1.weight'] == checkpoint['model']['module.layer.8.block.block.conv1.weight']))
    # except:
    #     print(all(checkpoint['best_model']['layer.0.block.alpha']==checkpoint['model']['layer.0.block.alpha']))
    #     print(torch.all(
    #         checkpoint['best_model']['layer.0.block.block.conv1.weight'] == checkpoint['model']['layer.0.block.block.conv1.weight']))
    #     print(torch.all(
    #         checkpoint['best_model']['layer.8.block.block.conv1.weight'] == checkpoint['model']['layer.8.block.block.conv1.weight']))


    # print(checkpoint['best_model']['module.layer.0.block.block.conv1.weight'][0])
    # print(checkpoint['model']['module.layer.0.block.block.conv1.weight'][0])
    # # print(checkpoint['best_model']['module.layer.0.block.block.conv1.weight'][1]

    # print(all())
    # # print(checkpoint['model']['module.layer.0.block.block.conv1.weight'][1)
    # print(checkpoint['best_model']['module.layer.8.block.block.conv1.weight']==checkpoint['model']['module.layer.8.block.block.conv1.weight'])
    if in_epoch:
    # if in_epoch:
        in_epoch = in_epoch
        c_epoch = in_epoch // 2 - 1
        # c_epoch = -1
        alpha = alpha_progress[c_epoch]

        # alpha = alpha_progress[20]
    else:
        in_epoch = best_epoch
        alpha = best_alpha
        c_epoch = best_epoch //2  #- 1
    print('epoch ', in_epoch,' from ', epoch)
    print(alpha)
    fig_name = str(test_name) + '_' + str(in_epoch)
    # plt.close('all')

    prop = accuracy_progress
    print('acccc', len(prop['train']))
    print(c_epoch)
    x = np.arange(0, epoch+1, 2)
    print(prop['test'][-20:])
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
    # prop = loss_progress

    test_title = 'a'
    test_title = 'test ' + str(test_name) +  '\n' + test_properties['dataset'] + ', ' + test_properties['mscale']  + '\n tr: ' + str(round(prop['train'][c_epoch]*100,2)) + ', val: ' + str(round(prop['validation'][c_epoch]*100, 2)) + ', test: ' + str(round(prop['test'][c_epoch]*100, 2)) + '\n epoch ' + str(in_epoch) + '/' + str(epoch)
    # test_title = 'test ' + str(test_name) +  '\n' + test_properties['dataset'] + ', ' + test_properties['mscale']  + '\n tr: ' + str(round(prop['train'][c_epoch]*100,2)) + ', val: ' + str(round(prop['validation'][c_epoch]*100, 2)) + ', test: ' + str(round(prop['test'][c_epoch]*100, 2)) + '\n epoch ' + str(1986) + '/' + str(2000)
    # test_title = 'test ' + str(test_name) +  '\n' + test_properties['dataset'] + ', ' + 'sres'  + '\n tr: ' + str(round(prop['train'][c_epoch],3)) + ', val: ' + str(round(prop['validation'][c_epoch], 3)) + ', test: ' + str(round(prop['test'][c_epoch], 3)) + '\n epoch ' + str(in_epoch) + '/' + str(epoch)
    # test_title = 'test ' + str(test_name) +  '\n' + test_properties['dataset'] + ', ' + 'sres'  + '\n tr: ' + str(round(prop['train'][c_epoch]*100,2)) + ', val: ' + str(round(prop['validation'][c_epoch]*100, 2)) + ', test: ' + str(round(prop['test'][c_epoch]*100, 2)) + '\n epoch ' + str(in_epoch) + '/' + str(epoch)
    # test_title = 'test ' + str(7092) +  '\n' + test_properties['dataset'] + ', ' + test_properties['mscale']  + '\n tr: ' + str(48.23) + ', val: ' + str(round(0, 2)) + ', test: ' + str(round(43.14, 2)) + '\n epoch ' + str(325) + '/' + str(500)
    # test_title = 'test ' + str(7085) +  '\n' + test_properties['dataset'] + ', ' + test_properties['mscale']  + '\n tr: ' + str(47.18,2)) + ', val: ' + str(round(0, 2)) + ', test: ' + str(round(42.67, 2)) + '\n epoch ' + str(319) + '/' + str(480)
    # block_type = test_properties['mscale']
    layers = test_properties['leng']
    res = test_properties['max_scales']# + 1
    res = 4
    if test_5 or test_6:
        res = test_properties['max_scales']   + 1
    if 1:
        alpha = torch.FloatTensor(alpha)
        #####################
        if test_6:
            alpha[0,-1] = -20000
            alpha[-1,-1] = -20000
        ########################
        nalpha = np.array(F.softmax(alpha, 1))
        # nalpha = np.array(alpha)
        # nalpha = nalpha / np.sum(nalpha,1).reshape(-1, 1)
        # print(nalpha / np.sum(nalpha,1).reshape(-1, 1))
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

        if pool:
            n = []
            for i in pool:
                n.append(np.eye(4)[i-1])
            m = [j.tolist() for j in n]
            nalpha = np.array(m)




        # nalpha = np.array([[1, 0, 0, 0],
        #                    [0, 1, 0, 0],
        #                    [0, 1, 0, 0],
        #                    [0, 0, 1, 0],
        #                    [0, 0, 1, 0],
        #                    [0, 0, 1, 0],
        #                    [0, 0, 0, 1],
        #                    [0, 0, 0, 1],
        #                    [0, 0, 0, 1],
        #                    [0, 0, 0, 1]])

        # nalpha = np.round_(nalpha, decimals=2)
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



    # '''
    # # get the weights on each layer
    # # '''
    # print('------------------------------------------------')
    # print('weights')
    # # print(checkpoint['best_model'])
    # for k,v in checkpoint['best_model'].items():
    #     print(k)
    # for l in range(layers):
    #     for r in range(res):
    #         # k = 'layer.'+str(l)+'.block.block.conv2.'+str(r)+'.weight'
    #         # k = 'layer.'+str(l)+'.block.block.bn2.'+str(r)+'.running_var'
    #         k = 'conv.'+str(l)+'.conv2.'+str(r)+'.weight'
    #         # k = 'conv.'+str(l)+'.bn3.'+str(r)+'.running_var'
    #
    #         w1 = checkpoint['best_model'][k]
    #         print(w1.size())
    #         # print(torch.norm(w1))
    #         nalpha[l,r] = torch.norm(w1)
    nalpha = np.round_(nalpha, decimals=2)
    # #
    # # for k,v in checkpoint['best_model'].items():
    # #     print(k)
    # for l in range(layers):
    #     k = 'conv.'+str(l)+'.alpha'
    #     print(checkpoint['best_model'][k])



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

    # fig2.savefig(fig_name)
    plt.close('all')

# for t in range(test_name,test_name+100):
#     try:
#         checkpoint1 = plotting(t)
#     except: pass

    for k, v in checkpoint['best_model'].items():
        print(k)

checkpoint1 = plotting(test_name,in_epoch=in_epoch,pro=pro, test_5=test_5, test_6 = test_6, pool=pool)