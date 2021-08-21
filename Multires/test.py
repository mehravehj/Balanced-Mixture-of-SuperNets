#

import torch
import torch.nn.functional as F

features = torch.randn(1, 10, 8, 8, requires_grad=True)  # a big grad-enabled feature map
features.retain_grad()

# downsample
rf = F.max_pool2d(features, kernel_size=2, stride=2)
rf.retain_grad()
fed = F.interpolate(rf, scale_factor=2, mode='nearest')
fed.retain_grad()
# reduced_feature = F.interpolate(features, size=(3, 3), mode="bilinear")  # I reduced the size to (3,3)
# print(reduced_feature.size())
# reduced_feature.retain_grad()
# loss = reduced_feature[:, :, 1, 1].pow(
#     2).sum()  # I want to use the center of the reduced feature map to pass the gradient

print(fed.requires_grad)

loss = fed.pow(2).sum()
print(loss)
loss.backward()

print(features.grad.size())
print(fed.grad.size())

import matplotlib.pyplot as plt

plt.subplot(121)
plt.imshow(features.grad[0,0,:,:], cmap='jet')#  # the gradient should be in the large central part of the original feature map, as I have downsampled it. However, the gradient says that in fact, only one pixel in original feature affects.

plt.title("gradient for the reduced feature space")
plt.subplot(122)
plt.imshow(fed.grad[0,0,:,:], cmap='jet')  # the gradient is in the center, okay

plt.title("gradient for the original feature space")

plt.show()
# import torch
# import torchvision
# import torchvision.transforms as transforms
#
# ########################################################################
# # The output of torchvision datasets are PILImage images of range [0, 1].
# # We transform them to Tensors of normalized range [-1, 1].
# # .. note::
# #     If running on Windows and you get a BrokenPipeError, try setting
# #     the num_worker of torch.utils.data.DataLoader() to 0.
#
# # transform = transforms.Compose(
# #     [transforms.ToTensor(),
# #      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#
# cifar_mean = [0.49139968, 0.48215827, 0.44653124]
# cifar_std = [0.24703233, 0.24348505, 0.26158768]
#
# transform = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize(cifar_mean, cifar_std),
# ])
#
# trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                         download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=256,
#                                           shuffle=True, num_workers=4)
#
# testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                        download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=256,
#                                          shuffle=False, num_workers=4)
#
# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#
# ########################################################################
# # Let us show some of the training images, for fun.
#
# import matplotlib.pyplot as plt
# import numpy as np
#
# # functions to show an image
#
#
# def imshow(img):
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
#
#
# # get some random training images
# dataiter = iter(trainloader)
# images, labels = dataiter.next()
#
# # show images
# imshow(torchvision.utils.make_grid(images))
# # print labels
# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
#
#
# ########################################################################
# # 2. Define a Convolutional Neural Network
# # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# # Copy the neural network from the Neural Networks section before and modify it to
# # take 3-channel images (instead of 1-channel images as it was defined).
#
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.interp = F.interpolate
#
#         self.conv1 = nn.Conv2d(6, 6, 3, stride=1, padding=1, bias=False)
#         self.relu = nn.ReLU()
#         # self.bn1 = nn.BatchNorm2d(out_, affine=False)
#         self.bn1 = nn.BatchNorm2d(6, affine=False)
#         self.conv2 = nn.Conv2d(6, 6, 3, stride=1, padding=1, bias=False)
#         # self.bn2 = nn.BatchNorm2d(out_, affine=False)
#         # separate BN for each resolution
#         self.bn2 = nn.BatchNorm2d(6, affine=False)
#
#
#         self.conv3 = nn.Conv2d(6, 6, 3, stride=1, padding=1, bias=False)
#         self.relu = nn.ReLU()
#         # self.bn1 = nn.BatchNorm2d(out_, affine=False)
#         self.bn3 = nn.BatchNorm2d(6, affine=False)
#         self.conv4 = nn.Conv2d(6, 6, 3, stride=1, padding=1, bias=False)
#         # self.bn2 = nn.BatchNorm2d(out_, affine=False)
#         # separate BN for each resolution
#         self.bn4 = nn.BatchNorm2d(6, affine=False)
#
#
#     def forward(self, x):
#         y = self.conv1(x)
#         y = self.bn1(y)
#         y = self.relu(y)
#         y = self.conv2(y)
#         y = self.bn2(y)
#         y += x
#         y = self.relu(y)
#
#         y1 = self.conv1(y)
#         y1 = self.bn1(y1)
#         y1 = self.relu(y1)
#         y1 = self.conv2(y1)
#         y1 = self.bn2(y1)
#         y1 += y
#         y1 = self.relu(y1)
#
#
#         return y1
#
#
# net = Net()
# net.cuda()
# print(net)
#
# import torch.optim as optim
#
# criterion = nn.CrossEntropyLoss()
# criterion = criterion.cuda()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
#
#
# for epoch in range(2):  # loop over the dataset multiple times
#     net.train()
#     running_loss = 0.0
#     for i, data in enumerate(trainloader, 0):
#         # get the inputs; data is a list of [inputs, labels]
#         inputs, labels = data
#         inputs = inputs.cuda()
#         labels =  labels.cuda()
#
#         # zero the parameter gradients
#         optimizer.zero_grad()
#
#         # forward + backward + optimize
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         print(loss)
#         loss.backward()
#         optimizer.step()
#
#         # print statistics
#         running_loss += loss.item()
#         if i % 2000 == 1999:    # print every 2000 mini-batches
#             print('[%d, %5d] loss: %.3f' %
#                   (epoch + 1, i + 1, running_loss / 2000))
#             running_loss = 0.0
#
# print('Finished Training')
#
#
#
#
# correct = 0
# total = 0
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data
#         outputs = net(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
# print('Accuracy of the network on the 10000 test images: %d %%' % (
#     100 * correct / total))
#
# ########################################################################
# # That looks way better than chance, which is 10% accuracy (randomly picking
# # a class out of 10 classes).
# # Seems like the network learnt something.
# #
# # Hmmm, what are the classes that performed well, and the classes that did
# # not perform well:
#
# class_correct = list(0. for i in range(10))
# class_total = list(0. for i in range(10))
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data
#         outputs = net(images)
#         _, predicted = torch.max(outputs, 1)
#         c = (predicted == labels).squeeze()
#         for i in range(4):
#             label = labels[i]
#             class_correct[label] += c[i].item()
#             class_total[label] += 1
#
#
# for i in range(10):
#     print('Accuracy of %5s : %2d %%' % (
#         classes[i], 100 * class_correct[i] / class_total[i]))
#
# ########################################################################
# # Okay, so what next?
# #
# # How do we run these neural networks on the GPU?
# #
# # Training on GPU
# # ----------------
# # Just like how you transfer a Tensor onto the GPU, you transfer the neural
# # net onto the GPU.
# #
# # Let's first define our device as the first visible cuda device if we have
# # CUDA available:
#
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
# # Assuming that we are on a CUDA machine, this should print a CUDA device:
#
# print(device)
#
# ########################################################################
# # The rest of this section assumes that ``device`` is a CUDA device.
# #
# # Then these methods will recursively go over all modules and convert their
# # parameters and buffers to CUDA tensors:
# #
# # .. code:: python
# #
# #     net.to(device)
# #
# #
# # Remember that you will have to send the inputs and targets at every step
# # to the GPU too:
# #
# # .. code:: python
# #
# #         inputs, labels = data[0].to(device), data[1].to(device)
# #
# # Why dont I notice MASSIVE speedup compared to CPU? Because your network
# # is really small.
# #
# # **Exercise:** Try increasing the width of your network (argument 2 of
# # the first ``nn.Conv2d``, and argument 1 of the second ``nn.Conv2d`` –
# # they need to be the same number), see what kind of speedup you get.
# #
# # **Goals achieved**:
# #
# # - Understanding PyTorch's Tensor library and neural networks at a high level.
# # - Train a small neural network to classify images
# #
# # Training on multiple GPUs
# # -------------------------
# # If you want to see even more MASSIVE speedup using all of your GPUs,
# # please check out :doc:`data_parallel_tutorial`.
# #
# # Where do I go next?
# # -------------------
# #
# # -  :doc:`Train neural nets to play video games </intermediate/reinforcement_q_learning>`
# # -  `Train a state-of-the-art ResNet network on imagenet`_
# # -  `Train a face generator using Generative Adversarial Networks`_
# # -  `Train a word-level language model using Recurrent LSTM networks`_
# # -  `More examples`_
# # -  `More tutorials`_
# # -  `Discuss PyTorch on the Forums`_
# # -  `Chat with other users on Slack`_
# #
# # .. _Train a state-of-the-art ResNet network on imagenet: https://github.com/pytorch/examples/tree/master/imagenet
# # .. _Train a face generator using Generative Adversarial Networks: https://github.com/pytorch/examples/tree/master/dcgan
# # .. _Train a word-level language model using Recurrent LSTM networks: https://github.com/pytorch/examples/tree/master/word_language_model
# # .. _More examples: https://github.com/pytorch/examples
# # .. _More tutorials: https://github.com/pytorch/tutorials
# # .. _Discuss PyTorch on the Forums: https://discuss.pytorch.org/
# # .. _Chat with other users on Slack: https://pytorch.slack.com/messages/beginner/
#
# # %%%%%%INVISIBLE_CODE_BLOCK%%%%%%
# del dataiter
# # %%%%%%INVISIBLE_CODE_BLOCK%%%%%%
#
#
#
#
#
#
#
#
#
#
# from __future__ import print_function
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# def downsample(x, num_res): # eliminate append
#     '''
#     downasample input num_res times using maxpooling
#     :param x: input feature map
#     :param num_res: how many times to downsample
#     :return: list for downsampled features
#     '''
#     multi_scale_input = [x]
#     for idx in range(num_res - 1):
#         xx = F.max_pool2d(multi_scale_input[idx], kernel_size=2)
#         multi_scale_input.append(xx)
#     return multi_scale_input
#
# def define_alpha(max_scales, ini_alpha=0, factor=1):
#     '''
#     define parameter alpha either as uniform ones or supplied initilization
#     :param max_scales:
#     :param ini_alpha:
#     :param factor:
#     :return:
#     '''
#     if ini_alpha:
#         alpha = torch.eye(max_scales)[ini_alpha-1] * factor
#     else:
#         alpha = torch.ones(max_scales) * factor
#     return alpha
#
# class conv_block_same_filter(nn.Module):
#     def __init__(self, in_, out_, kernel_size, max_scales=4):
#         super(conv_block_same_filter, self).__init__()
#         self.interp = F.interpolate
#         self.max_scales = max_scales
#         self.conv = nn.Conv2d(in_, out_, kernel_size, stride=1, padding=1)
#         self.relu = nn.ReLU()
#         # separate BN for each resolution
#         # self.bn = nn.ModuleList([nn.BatchNorm2d(out_, affine=False)
#         #                          for _ in range(self.max_scales)])
#         self.bn = nn.BatchNorm2d(out_, affine=False)
#
#     def forward(self, x): # eliminate append
#         lx = downsample(x, self.max_scales)
#         ly = []
#         for r in range(self.max_scales):
#             y = self.conv(lx[r])
#             y = self.relu(y)
#             # separate BN for each resolution
#             # y = self.bn[r](y)
#             y = self.bn(y)
#             ly.append(self.interp(y, scale_factor=2 ** r, mode='nearest'))
#         out = (torch.stack(ly, 0))
#         return out
#
#
# class conv_block_normal(nn.Module):
#     def __init__(self, in_, out_, kernel_size):
#         super(conv_block_normal, self).__init__()
#         self.conv = nn.Conv2d(in_, out_, kernel_size, stride=1, padding=1)
#         self.relu = nn.ReLU()
#         self.bn = nn.BatchNorm2d(out_, affine=False)
#
#     def forward(self, x):
#         y = self.conv(x)
#         y = self.relu(y)
#         out = self.bn(y)
#         return out
#
#
# class ResBlock_same_filters(nn.Module):
#     def __init__(self, in_, out_, kernel_size, max_scales=4):
#         super(ResBlock_same_filters, self).__init__()
#         self.interp = F.interpolate
#         self.max_scales = max_scales
#         self.in_ = in_
#         self.out_ = out_
#
#         self.conv1 = nn.Conv2d(in_, out_, kernel_size, stride=1, padding=1, bias=False)
#         self.relu = nn.ReLU()
#         #self.bn1 = nn.BatchNorm2d(out_, affine=False)
#         self.bn1 = nn.ModuleList([nn.BatchNorm2d(out_, affine=False) for _ in range(self.max_scales)])
#         self.conv2 = nn.Conv2d(out_, out_, kernel_size, stride=1, padding=1, bias=False)
#         #self.bn2 = nn.BatchNorm2d(out_, affine=False)
#         self.bn2 = nn.ModuleList([nn.BatchNorm2d(out_, affine=False) for _ in range(self.max_scales)])
#         # separate BN for each resolution
#         # self.bn = nn.ModuleList([nn.BatchNorm2d(out_, affine=False) for _ in range(self.max_scales)])  # batchnorm not in default resnet block
#         ########################## xtfsaxtstax
#         if in_ != out_:
#             self.conv3 = nn.Conv2d(in_, out_, kernel_size=1, stride=1, padding=0, bias=False)
#             self.bn3 = nn.BatchNorm2d(out_, affine=False)
#
#     def forward(self, x): # eliminate append
#         lx = downsample(x, self.max_scales)
#         ly = []
#         for r in range(self.max_scales):
#             y = self.conv1(lx[r])
#             y = self.bn1[r](y)
#             y = self.relu(y)
#             y = self.conv2(y)
#             y = self.bn2[r](y)
#
#             if self.in_ != self.out_:
#                 resid = self.conv3(lx[r])
#                 resid = self.bn3(resid)
#             else:
#                 resid = lx[r]
#             y += resid
#             y = self.relu(y)
#             # separate BN for each resolution
#             # y = self.bn[r](y)# batchnorm not in default resnet block
#             ly.append(self.interp(y, scale_factor=2 ** r, mode='nearest'))
#         out = (torch.stack(ly, 0))
#         return out
#
#
# class ResBlock_normal(nn.Module):
#     def __init__(self, in_, out_, kernel_size):
#         super(ResBlock_normal, self).__init__()
#         self.in_ = in_
#         self.out_ = out_
#
#         self.conv1 = nn.Conv2d(in_, out_, kernel_size, stride=1, padding=1, bias=False)
#         self.relu = nn.ReLU()
#         self.bn1 = nn.BatchNorm2d(out_, affine=False)
#         self.conv2 = nn.Conv2d(out_, out_, kernel_size, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_, affine=False)
#         if in_ != out_:
#             self.conv3 = nn.Conv2d(in_, out_, kernel_size=1, stride=1, padding=0, bias=False)
#             self.bn3 = nn.BatchNorm2d(out_, affine=False)
#
#     def forward(self, x):
#         y = self.conv1(x)
#         y = self.bn1(y)
#         y = self.relu(y)
#         y = self.conv2(y)
#         y = self.bn2(y)
#
#         if self.in_ != self.out_:
#             resid = self.conv3(x)
#             resid = self.bn3(resid)
#         else:
#             resid = x
#         y += resid
#         out = self.relu(y)
#         return out