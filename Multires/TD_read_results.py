import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as anim

plt.close('all')
test_name = 749
in_epoch = 0
#probs = probability per epoch
# loss_progress['valid']

def string_to_list(x):
    x = x.split(',')
    res = [int(i) for i in x]
    return res

#load properties
checkpoint = torch.load('./checkpoint_loss/ckpt_TD_' + str(test_name) + '.t7')
test_properties = checkpoint['test_properties']
layers = test_properties['leng']
factor = test_properties['factor']
res = test_properties['max_scales']
print(test_properties)
loss_progress = checkpoint['loss_progress']
epoch = checkpoint['epoch']
probs_progress = checkpoint['probs_progress']
loss_per_res_progress = checkpoint['loss_per_res_progress']
best_paths_progress = checkpoint['best_paths_progress']
worst_paths_progress = checkpoint['worst_paths_progress']


if in_epoch:
    c_epoch = in_epoch - 1
else:
    c_epoch = epoch - 1
fig_name = str(test_name) + '_' + str(c_epoch)

'''
plot average loss
'''
# def plot_avg_loss(epoch, loss_progress, test_name):
x = np.arange(1, epoch+1)# for properties saved evey epoch
prop = loss_progress
fig1 = plt.figure(1)
plt.title('average loss vs epoch ' + str(test_name))
plt.xlabel('epoch')
plt.ylabel('loss')
plt.ylim(bottom=0, top=2.5)
plt.plot(x, np.asarray(prop['train']) * 100.0 / 79, 'b', label='train')
plt.plot(x, np.asarray(prop['validation']) * 100.0 / 79, 'g', label='validation')
# plt.plot(x, np.asarray(prop['test']) * 100.0, 'r', label='test')
plt.minorticks_on()
plt.grid(which='both')
plt.grid(True)
plt.grid(b=True, which='minor', linestyle='-', alpha=0.2)
plt.legend()
plt.show()


'''
plot probability distribution
'''
def plot_prob(test_name, test_properties, factor, c_epoch, epoch, probs_progress, layers,res):
    # current probabilities
    print(probs_progress[c_epoch])
    test_title = 'test' + str(test_name) + '\n' + test_properties['dataset'] + ', ' + test_properties['mscale'] + \
                 '\n channels: ' + test_properties['channels'] + ', lr: ' + str(test_properties['learning_rate']) + \
                 ', factor: ' + str(factor) + '\n epoch ' + str(c_epoch) + '/' + str(epoch)
    final_prob = np.array(probs_progress[c_epoch])
    final_prob = np.round_(final_prob, decimals=2)
    plt.ioff()
    fig2 = plt.figure(figsize=[2.5, 4.8])
    gs1 = gridspec.GridSpec(layers, res)
    ax1 = fig2.add_subplot(gs1[:, :])
    im1 = ax1.imshow(final_prob, cmap='Purples', vmin=0, vmax=1)
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
            text1 = ax1.text(j, i, final_prob[i][j], ha="center", va="center", color="Black", fontsize=8)
    plt.show()

def create_anim(fsp, stride, layers, res, c_epoch, probs_progress, test_name):
    fsp = 1
    stride = 5
    ims = []
    fig3, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(2.5,0.5*layers))
    gs1 = gridspec.GridSpec(layers, res)
    gs1.update(left=0.2, right=1, wspace=0, hspace=0)
    ax1 = plt.subplot(gs1[:, :-2])
    ax1.set_xticks(np.arange(res))
    ax1.set_yticks(np.arange(layers))
    ax1.set_yticklabels(np.arange(layers) + 1)
    ax1.set_xticklabels(np.arange(res) + 1)
    ax1.xaxis.tick_top()
    ax1.set_xlabel('resolutions', fontsize=10)
    ax1.set_ylabel('layers', fontsize=10)
    ax1.xaxis.set_label_position("top")

    for i in range(c_epoch // stride):
        prob = np.array(probs_progress[i])
        prob = prob.tolist()
        im1 = ax1.imshow(prob, cmap='Purples', vmin=0, vmax=1, animated=True)
        t = ax1.annotate(str('epoch ' + str(i*stride)), (1, -20), xycoords='axes pixels')
        ims.append([im1, t])
    ani = anim.ArtistAnimation(fig3, ims)
    anim_name = 'anim' + str(test_name) + '.gif'
    ani.save(anim_name, writer='imagemagick', fps=fsp)
    print('gif created and saved!')


plot_prob(test_name, test_properties, factor, c_epoch, epoch, probs_progress, layers,res)