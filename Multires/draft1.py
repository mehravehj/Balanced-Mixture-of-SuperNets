import torch
from data_loader import data_loader
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ss = 3
torch.manual_seed(ss)

train_loader, validation_loader, test_loader, indices, num_class = data_loader('TIN', 0.5, 1, num_train=0, indices=[[1,2],[3,4]], dataset_dir='~/Desktop/codes/multires/data/', workers=4)
for batch_idx, (train_inputs, train_targets) in enumerate(train_loader):
    train_inputs, train_targets = train_inputs, train_targets
    img = train_inputs[0,0,:,:]
    print(img.size())
    plt.imshow(img)
    plt.show()
img = train_inputs[1,0,:,:]
print(img.size())
plt.imshow(img)
plt.show()
plt.close('all')

# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # import numpy as np
# #
# # def define_alpha(max_scales, ini_alpha=0, factor=1):
# #     if ini_alpha:
# #         alpha = torch.eye(max_scales)[ini_alpha-1] * factor
# #     else:
# #         alpha = torch.ones(max_scales) * factor
# #     return alpha
# #
# # initial_alpha = 0
# # # alpha = nn.Parameter(define_alpha(4, ini_alpha=initial_alpha, factor=1))
# # # # alpha = define_alpha(4, ini_alpha=initial_alpha, factor=1)
# # # alpha = alpha.type(torch.FloatTensor)
# # # print(alpha.type())
# # # print('alpha', alpha)
# # # alpha_dropout = F.dropout(alpha, p=0.5)
# # # alpha_dropout[alpha_dropout == 0] = float('NaN')
# # # print('alpha_dropout', alpha_dropout)
# # # nalpha = F.softmax(alpha, 0).view(-1, 1, 1, 1, 1)
# # # nalpha_dropout = F.softmax(alpha_dropout, 0).view(-1, 1, 1, 1, 1)
# # # # print(F.softmax(torch.Tensor([1,0,0,0,0])), 0)
# # # print('nalpha', nalpha)
# # # print('nalpha dropout', nalpha_dropout)
# #
# #
# # ### MANUAL implemetation
# # # p = 0.5
# # # xx = F.feature_alpha_dropout(alpha, p=0.5)
# # # print(xx)
# #
# #
# # # i = [[0, 2], [1, 0], [1, 2]]
# # # v =  [3,      4,      5    ]
# # #
# # # s = torch.sparse_coo_tensor(list(zip(*i)), v, (2, 3))
# # #
# # # i = np.array(i)
# # # transpose = i.T
# # #
# # # i = transpose.tolist()
# # # y = torch.sparse_coo_tensor(i, v, torch.Size([2,3])).to_dense()
# # # print(y)
# #
# #
# #
# # i = [[0, 2], [0, 0], [0, 4]]
# # v =  [1,      1,      1    ]
# # v = nn.Parameter(torch.FloatTensor(v))
# # s = torch.sparse_coo_tensor(list(zip(*i)), v, (1, 5))
# # i = np.array(i)
# # transpose = i.T
# #
# # i = transpose.tolist()
# # y = torch.sparse_coo_tensor(i, v, torch.Size([1,5])).to_dense()
# # # s = s.cuda()
# # print(s)
# # print(y)
# # print('values',s._values())
# # # torch.sparse_coo_tensor(i.t(), v, torch.Size([2,3])).to_dense()
# # print(s.is_coalesced())
# # ss = F.softmax(s._values(), -1)#.view(-1, 1, 1, 1, 1)
# # print(ss)
# # y = torch.sparse_coo_tensor(i, ss, torch.Size([1,5])).to_dense()
# # print(y)
# # xx = [j[1] for j in i]
# # print(xx)
#
# """
# Script to demonstrate the usage of shared dicts using multiple workers.
# In the first epoch the shared dict in the dataset will be filled with
# random values. The next epochs will just use the dict without "loading" the
# data again.
# @author: ptrblck
# """
#
# from multiprocessing import Manager
#
# import torch
# from torch.utils.data import Dataset, DataLoader
# from data_loader import data_loader
# import torch
# import torchvision
# import torchvision.transforms as transforms
# from random import shuffle
# from torch.utils.data import Dataset
# import torch.utils.data
# import torchvision.datasets as datasets
# from torch.utils.data.sampler import SubsetRandomSampler
# import os
# from multiprocessing import Manager
# from torch.utils.data import Dataset
# shared_dict = {}
# global shared_dict
#
#
# class MyDataset(Dataset):
#     def __init__(self, shared_dict, length):
#         self.shared_dict = shared_dict
#         self.length = length
#
#     def __getitem__(self, index):
#         if index not in self.shared_dict:
#             print('Adding {} to shared_dict'.format(index))
#             self.shared_dict[index] = torch.tensor(index)
#         return self.shared_dict[index]
#
#     def __len__(self):
#         return self.length
#
#
# # Init
# def validation_set_indices(num_train, valid_percent, dataset_name):
#     train_size = num_train - int(valid_percent * num_train)  # number of training examples
#     val_size = num_train - train_size  # number of validation examples
#     if dataset_name =='TIN':
#         image_per_class = 500
#         validation_per_class = int(image_per_class * valid_percent)
#         val_index = []
#         train_index = []
#         for i in range(image_per_class, num_train + image_per_class, image_per_class):
#             jj = list(range(i - image_per_class, i))
#             # shuffle(jj)
#             val_index += jj[:validation_per_class]
#             train_index += jj[validation_per_class:]
#             shuffle(val_index)
#             shuffle(train_index)
#     else:
#         indices = list(range(num_train))  # available indices at training set
#         shuffle(indices)
#         indices = indices[:num_train]
#         split = train_size
#         train_index = indices[:split]
#         val_index = indices[split:]
#     indices = [train_index, val_index]
#     return indices
#
#
# def _data_transforms_tinyimagenet():
#     tin_mean = [0.4802, 0.4481, 0.3975]
#     tin_std = [0.2302, 0.2265, 0.2262]
#     train_transform = transforms.Compose([
#         transforms.Pad(4),
#         transforms.RandomCrop(64),
#         transforms.RandomRotation(20),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomResizedCrop(size=64, scale=(0.9, 1.08), ratio=(0.99, 1)),
#         transforms.ToTensor(),
#         transforms.Normalize(tin_mean, tin_std),
#     ])
#     test_transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(tin_mean, tin_std),
#     ])
#     return {'train':train_transform, 'test':test_transform}
#
#
# class Cache(Dataset):
#     def __init__(self, dataset, shared_dict):
#         self.shared_dict = shared_dict
#         self.dataset = dataset
#
#     def __getitem__(self, index):
#         if index not in self.shared_dict:
#              print('Adding {} to shared_dict'.format(index))
#              # self.shared_dict[index] = torch.tensor(index)
#              self.shared_dict[index] = self.dataset[index]
#         return self.shared_dict[index]
#
#     def __len__(self):
#         # return self.length
#         return len(self.dataset)
#
# if __name__ == '__main__':
#     dataset_dir = '~/Desktop/codes/multires/data/tiny-imagenet-200/'
#     valid_percent = 0.34
#     batch_size = 32
#     num_train = 0
#     indices = 0
#     workers = 4
#     image_datasets = {x: datasets.ImageFolder(os.path.join(dataset_dir, x), _data_transforms_tinyimagenet()[x])
#                       for x in ['train', 'test']}
#     trainset = image_datasets['train']
#
#     shared_dict_train = torch.load('train_dict.t7')
#     print(shared_dict_train)
#     i = 0
#     for key, value in shared_dict_train:
#         i += 1
#     print(i)
#
#     # manager = Manager()
#     # shared_dict_train = manager.dict()
#
#     shared_dict_train = dict()
#
#     trainset = Cache(trainset, shared_dict_train)
#
#     if not num_train:
#         num_train = len(trainset)
#
#     if not indices:
#         indices = validation_set_indices(num_train, valid_percent, 'TIN')
#
#     train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=SubsetRandomSampler(indices[0]), num_workers=workers, pin_memory=True, drop_last=True)
#
#     for batch_idx, (train_inputs, train_targets) in enumerate(train_loader):
#         # print(batch_idx)
#         train_inputs.cuda()
#         train_targets.cuda()
#         # train_inputs, train_targets = train_inputs.cuda(), train_targets.cuda()
#         print(batch_idx)
#         if batch_idx == 10:
#             print(shared_dict_train)
#             torch.save(shared_dict_train, 'train_dict.t7')
#             break
#     # for batch_idx, (train_inputs, train_targets) in enumerate(train_loader):
#     #     # print(batch_idx)
#     #     train_inputs, train_targets = train_inputs.cuda(), train_targets.cuda()
#     #     print(batch_idx)
#
# # manager = Manager()
# # shared_dict = manager.dict()
# # print(shared_dict)
# # dataset = MyDataset(shared_dict, length=100)
# #
# # loader = DataLoader(
# #     dataset,
# #     batch_size=10,
# #     num_workers=6,
# #     shuffle=True,
# #     pin_memory=True
# # )
# #
# # # First loop will add data to the shared_dict
# # for x in loader:
# #     print(x)
# #
# # # The second loop will just get the data
# # for x in loader:
# #     print(x)
#
#
