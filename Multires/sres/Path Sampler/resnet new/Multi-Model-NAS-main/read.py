import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions.categorical import Categorical as Categorical

test_name = 21
in_epoch =2000
save_dir = './checkpoint/multilodel_chpt_' + str(test_name) + '.t7'  # checkpoint save directory
checkpoint = torch.load(save_dir)
print(checkpoint['epoch'])
test_properties = checkpoint['test_properties']
num_models = test_properties['n_models']
num_paths = 36

t_loss = checkpoint['t_loss']
t_loss = [t/97 for t in t_loss]
v_loss = checkpoint['v_loss']
v_loss = [t/97 for t in v_loss]
t_acc = checkpoint['t_acc']
t_acc = [(i*100).view(1) for i in t_acc]
t_acc = torch.cat(t_acc, dim=0)
v_acc = checkpoint['v_acc']
v_acc = [(i*100).view(1) for i in v_acc]
v_acc = torch.cat(v_acc, dim=0)

temperature = checkpoint['temperature']
epoch = checkpoint['epoch']
if in_epoch > epoch:
    in_epoch = epoch

c_matrix = checkpoint['c_matrix']
counter_mat = c_matrix[in_epoch]
w_matrix = checkpoint['w_matrix']
weight_mat = w_matrix[in_epoch]
acc_mat = checkpoint['acc_mat']
# print('acc matrix')
# print(acc_mat[1900])
# print(w_matrix[1990])
# print(acc_mat[300])
color_scheme = {'counter':'Blues', 'prob':'Greens', 'eval_acc':'Purples'}



def plotting(a=0, l=0, counter=0, prob=0, eval_acc=0, in_epoch=in_epoch, num_models=num_models, num_paths=num_paths,
             color_scheme=color_scheme, temperature=temperature,t_acc=t_acc, v_acc=v_acc, t_loss=t_loss,
             c_matrix=c_matrix, w_matrix=w_matrix, acc_mat=acc_mat):
    if a:
        plot_loss_acc(t_acc, v_acc, epoch, test_name, 'accuracy')
    if l:
        plot_loss_acc(t_loss, v_loss, epoch, test_name, 'loss')
    if counter:
        show_matrix(c_matrix, in_epoch, 'counter', num_models, num_paths, color_scheme, temperature)
    if prob:
        show_matrix(w_matrix, in_epoch, 'prob', num_models, num_paths, color_scheme, temperature)
    if eval_acc:
        show_matrix(acc_mat, in_epoch, 'eval_acc', num_models, num_paths, color_scheme, temperature)


def plot_loss_acc(t_metric, v_metric, epoch, test_name, type):
    x = [i for i in range(0, epoch+1)]
    x = np.array(x)
    # plot
    fig1 = plt.figure(1)
    plt.title('Test: %d , %s vs epoch ' %(test_name, type))
    plt.xlabel('epoch')
    plt.ylabel('type')
    plt.ylim(bottom=0, top=4.5)
    if type == 'accuracy':
        plt.ylim(bottom=20, top=100)
    plt.plot(x, np.array(t_metric), 'b', label='train')
    plt.plot(x, np.array(v_metric), 'r', label='validation')
    plt.minorticks_on()
    plt.grid(which='both')
    plt.grid(True)
    plt.grid(b=True, which='minor', linestyle='-', alpha=0.2)
    plt.legend()
    plt.show()



def show_matrix(matrix, in_epoch, type, num_models, num_paths, color_scheme, temperature):
    mat = matrix[in_epoch]
    if type == 'prob':
        p = Categorical(logits=mat * temperature[in_epoch])
        print(temperature[in_epoch])
        print(mat)
        print(p)
        mat = p.probs
        mat = np.array(mat)
        mat = np.round_(mat, decimals=2)
    elif type == 'eval_acc':
        in_epoch = (in_epoch // 100) * 100
        print(in_epoch)
        mat = matrix[in_epoch]
        mat = np.array(mat) * 100
        mat = np.round_(mat, decimals=2)
    else:
        mat = np.array(mat)
    plt.ioff()
    # fig2 = plt.figure(figsize=[2, 10])
    # fig2 = plt.figure()
    # gs1 = gridspec.GridSpec(num_paths, num_models, width_ratios=[8 for _ in range(num_models)],
    #                       height_ratios=[1 for _ in range(num_paths)])
    # ax1 = fig2.add_subplot(gs1[:, :])
    fig, ax1 = plt.subplots(figsize=[1, 10])
    # ax1 = fig2.add_subplot()
    if type == 'counter' :
        im1 = ax1.imshow(mat, cmap=color_scheme[type])  # , vmin=0, vmax=1000)
    elif type == 'prob':
        im1 = ax1.imshow(mat, cmap=color_scheme[type], vmin=0, vmax=1)
    elif type == 'eval_acc':
        im1 = ax1.imshow(mat, cmap=color_scheme[type], vmin=50, vmax=100)
    ax1.set_xticks(np.arange(num_models))
    ax1.set_yticks(np.arange(num_paths))
    ax1.set_yticklabels(np.arange(num_paths) + 1)
    ax1.set_xticklabels(np.arange(num_models) + 1)
    ax1.xaxis.tick_top()
    # ax1.set_xlabel('resolutions', fontsize=10)
    # ax1.set_ylabel('layers', fontsize=10)
    # ax1.xaxis.set_label_position("top")
    test_title = 'Test %d: \n %s matrix \n in epoch %d' %(test_name, type, in_epoch)
    ax1.set_title(test_title, fontsize=8)

    for i in range(num_paths):
        for j in range(num_models):
            text1 = ax1.text(j, i, mat[i][j], ha="center", va="center", color="Black", fontsize=5)
    plt.show()

# plotting(1,1,1,1,1)
plotting(1,1,1,1,1)

# fig2.savefig(fig_name)
plt.close('all')


