import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as anim
from PIL import Image, ImageDraw

plt.close('all')
test_name = 7018
factor = 1

def string_to_list(x):
    x = x.split(',')
    res = [int(i) for i in x]
    return res


checkpoint = torch.load('./checkpoint/ckpt_' + str(test_name) + '.t7')
test_properties = checkpoint['test_properties']
best_epoch = checkpoint['best_epoch']
loss_progress = checkpoint['loss_progress']
accuracy_progress = checkpoint['accuracy_progress']
alpha_progress = checkpoint['alpha_progress']
best_alpha = checkpoint['best_alpha']
best_accuracy = checkpoint['best_accuracy']
epoch = checkpoint['epoch']

print(best_accuracy)

in_epoch = best_epoch

c_epoch = best_epoch //5
print('epoch ', in_epoch,' from ', epoch)
fig_name = str(test_name) + '_' + str(in_epoch)
# plt.close('all')

prop = accuracy_progress
x = np.arange(0, epoch+1, 5)


# test_title = 'test ' + str(test_name) +  '\n' + test_properties['dataset'] + ', ' + test_properties['mscale']  + '\n tr: ' + str(round(prop['train'][c_epoch]*100,2)) + ', val: ' + str(round(prop['validation'][c_epoch]*100, 2)) + ', test: ' + str(round(prop['test'][c_epoch]*100, 2)) + '\n epoch ' + str(in_epoch) + '/' + str(epoch)
test_title = 'test ' + str(test_name) +  '\n' + test_properties['dataset'] + ', ' + test_properties['mscale']  + '\n tr: ' + str(round(prop['train'][c_epoch]*100,2)) + ', val: ' + str(round(prop['validation'][c_epoch]*100, 2)) + ', test: ' + str(round(prop['test'][c_epoch]*100, 2)) + '\n epoch ' + str(0) + '/' + str(epoch)
# test_title = 'test ' + str(test_name) + '\n' + 'TIN' + ', ' + 'sres' + '\n tr: ' + str(90.53) + ', val: ' + str(0) + ', test: ' + str(60.1) + '\n epoch ' + str(580) + '/' + str(600)
net_type = test_properties['net_type']
block_type = test_properties['mscale']
pooling = test_properties['pooling']
initial_alpha = test_properties['initial_alpha']
layers = test_properties['leng']
res = test_properties['max_scales']
# res = 5

alpha = alpha_progress[0]
alpha = torch.FloatTensor(alpha)
nalpha = np.array(F.softmax(alpha, 1))
nalpha = np.round_(nalpha, decimals=2)

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

ax1.set_title(test_title, fontsize=10)

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

# FuncAnimation will call the 'update' function for each frame; here
# animating over 10 frames, with an interval of 200ms between frames.


ims = []
for i in range(5):
    test_title = 'test ' + str(test_name) + '\n' + test_properties['dataset'] + ', ' + test_properties[
        'mscale'] + '\n tr: ' + str(round(prop['train'][c_epoch] * 100, 2)) + ', val: ' + str(
        round(prop['validation'][c_epoch] * 100, 2)) + ', test: ' + str(
        round(prop['test'][c_epoch] * 100, 2)) + '\n epoch ' + str(i) + '/' + str(epoch)

    alpha = alpha_progress[i]
    alpha = torch.FloatTensor(alpha)
    nalpha = np.array(F.softmax(alpha, 1))
    nalpha = np.round_(nalpha, decimals=2)

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

    ax1.set_title(test_title, fontsize=10)

    for i in range(layers):
        for j in range(res):
            text1 = ax1.text(j, i, nalpha[i][j], ha="center", va="center", color="Black", fontsize=8)
    # im = plt.show()
    # print(im)
    ims.append([im1])

# anim = anim.ArtistAnimation(fig2, ims, interval=20)
# print(anim)
# anim.save('line.gif', dpi=80, writer='imagemagick')
# anim.save('dynamic_images.mp4')
for i in ims:
    print(ims)
im1.save('out.gif', save_all=True, append_images=ims)
# plt.show()
# #
# #
# #
# #
# import sys
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
#
# fig, ax = plt.subplots()
# fig.set_tight_layout(True)
#
# # Query the figure's on-screen size and DPI. Note that when saving the figure to
# # a file, we need to provide a DPI for that separately.
# print('fig size: {0} DPI, size in inches {1}'.format(
#     fig.get_dpi(), fig.get_size_inches()))
#
# # Plot a scatter that persists (isn't redrawn) and the initial line.
# x = np.arange(0, 20, 0.1)
# ax.scatter(x, x + np.random.normal(0, 3.0, len(x)))
# line, = ax.plot(x, x - 5, 'r-', linewidth=2)
#
# def update(i):
#     label = 'timestep {0}'.format(i)
#     print(label)
#     # Update the line and the axes (with a new xlabel). Return a tuple of
#     # "artists" that have to be redrawn for this frame.
#     line.set_ydata(x - 5 + i)
#     ax.set_xlabel(label)
#     return line, ax
#
# if __name__ == '__main__':
#     # FuncAnimation will call the 'update' function for each frame; here
#     # animating over 10 frames, with an interval of 200ms between frames.
#     anim = FuncAnimation(fig, update, frames=np.arange(0, 10), interval=200)
#     anim.save('line.gif', dpi=80, writer='imagemagick')
