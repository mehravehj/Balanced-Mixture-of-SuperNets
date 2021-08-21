import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as anim

plt.close('all')

activations = torch.load('./jj.t7')
for act in activations:
    print(act.size())
# print(activations[0])
image = 12
layer = 9

act = activations[layer][image].cpu().detach().numpy()
act = (act - act.min()) / (act.max() - act.min())
r = act[0]
g = act[1]
b = act[2]
# act = [r, g, b]
# # act = np.array(act)
# # act = act.reshape((96,96,3))
# act = np.zeros((96,96,3))
# act[:,:,0] = r
# act[:,:,1] = g
# act[:,:,2] = b
# print(act.shape)
#
# plt.imshow(act)
# plt.show()

print(act.max())
act = (act - act.min()) / (act.max() - act.min())
print(act.max())
a = 4
b = 4
plt.ioff()
fig, axs = plt.subplots(a, b)
k = np.arange(32)
# k =  k.view(8, 4)
l = 0
for m in range(8):
    plt.ioff()
    fig, axs = plt.subplots(a, b)
    for i in range(a):
        print(i)
        for j in range(b):
            plt.axis('off')
            print(j)
            print(l)
            wl = act[l]
            l += 1

            axs[i,j].set_axis_off()
            # for k in range(3):
            #     for m in range(3):
            #         ww = np.round_(wl[k,m]*100, decimals=2)
            #         text1 = axs[i,j].text(k, m, ww, ha="center", va="center", color="Red", fontsize=6)
            # text1 = axs[i,j].text(j, i, wl, ha="center", va="center", color="Black", fontsize=6)
            im3 = axs[i,j].imshow(wl, cmap='Greys')
    plt.tight_layout()
    plt.show()


