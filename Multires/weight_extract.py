import matplotlib.pyplot as plt
import numpy as np
import torch

plt.close('all')
test_name = 5090

checkpoint = torch.load('./checkpoint/ckpt_' + str(test_name) + '.t7')
state_net = checkpoint['best_net']

try:
    state_net = state_net[0]#TODO: some runs need this
except:
    pass

w = state_net['module.conv.1.conv2.weight'].cpu().detach().numpy()
print(state_net.keys())
# l2 = torch.norm(w)
print(w.shape)

sa = 4
sb = 4
a = 4
b = 4
w = w[sa:,sb:,:,:]
plt.ioff()
fig, axs = plt.subplots(a, b)#, subplot_kw=dict(polar=True))

for i in range(a):
    print(i)
    for j in range(b):
        plt.axis('off')
        print(j)
        wl = w[i,j,:,:]
        print(wl.shape)
        axs[i,j].set_axis_off()
        for k in range(3):
            for m in range(3):
                ww = np.round_(wl[k,m]*100, decimals=2)
                text1 = axs[i,j].text(k, m, ww, ha="center", va="center", color="Red", fontsize=6)
        # text1 = axs[i,j].text(j, i, wl, ha="center", va="center", color="Black", fontsize=6)
        im3 = axs[i,j].imshow(wl, cmap='Greens')

plt.tight_layout()
plt.show()



# # Create four polar axes and access them through the returned array
# fig, axs = plt.subplots(2, 2, subplot_kw=dict(polar=True))
# axs[0, 0].plot(x, y)
# axs[1, 1].scatter(x, y)
#
# # Share a X axis with each column of subplots
# plt.subplots(2, 2, sharex='col')
#
# # Share a Y axis with each row of subplots
# plt.subplots(2, 2, sharey='row')
#
# # Share both X and Y axes with all subplots
# plt.subplots(2, 2, sharex='all', sharey='all')
#
# # Note that this is the same as
# plt.subplots(2, 2, sharex=True, sharey=True)
#
# # Create figure number 10 with a single subplot
# # and clears it if it already exists.
# fig, ax = plt.subplots(num=10, clear=True)