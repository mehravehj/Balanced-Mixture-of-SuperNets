import os
from multiprocessing import Manager
from random import shuffle

import torch
import torch.utils.data
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler


class Cache(Dataset):
    def __init__(self, dataset, shared_dict):
        self.shared_dict = shared_dict
        self.dataset = dataset

    def __getitem__(self, index):
        if index not in self.shared_dict:
             # print('Adding {} to shared_dict'.format(index))
             # self.shared_dict[index] = torch.tensor(index)
             self.shared_dict[index] = self.dataset[index]
        return self.shared_dict[index]

    def __len__(self):
        # return self.length
        return len(self.dataset)



def _data_transforms_cifar10():
    cifar_mean = [0.49139968, 0.48215827, 0.44653124]
    cifar_std = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])
    return train_transform, test_transform


def _data_transforms_stl10():
    stl_mean = [0.5, 0.5, 0.5]
    stl_std = [0.5, 0.5, 0.5]
    train_transform = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomCrop(96),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(stl_mean, stl_std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(stl_mean, stl_std),
    ])
    return train_transform, test_transform


def _data_transforms_tinyimagenet():
    tin_mean = [0.4802, 0.4481, 0.3975]
    tin_std = [0.2302, 0.2265, 0.2262]
    train_transform = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomCrop(64),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(size=64, scale=(0.9, 1.08), ratio=(0.99, 1)),
        transforms.ToTensor(),
        transforms.Normalize(tin_mean, tin_std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(tin_mean, tin_std),
    ])
    return {'train':train_transform, 'test':test_transform}


def validation_set_indices(num_train, valid_percent, dataset_name):
    train_size = num_train - int(valid_percent * num_train)  # number of training examples
    print('training size:', train_size)
    val_size = num_train - train_size  # number of validation examples
    if dataset_name =='TIN':
        image_per_class = 500
        validation_per_class = int(image_per_class * valid_percent)
        val_index = []
        train_index = []
        for i in range(image_per_class, num_train + image_per_class, image_per_class):
            jj = list(range(i - image_per_class, i))
            # shuffle(jj)
            val_index += jj[:validation_per_class]
            train_index += jj[validation_per_class:]
            shuffle(val_index)
            shuffle(train_index)
    else:
        indexes = list(range(num_train))  # available indices at training set
        shuffle(indexes)
        indexes = indexes[:num_train]
        split = train_size
        train_index = indexes[:split]
        val_index = indexes[split:]
        # print('do they have common element?')
        # print(bool(set(train_index) & set(val_index)))
        # print(any(i in train_index for i in val_index))
    indices = [train_index, val_index]
    return indices


def data_loader(dataset_name, valid_percent, batch_size, num_train=0, indices=0, dataset_dir='~/Desktop/codes/multires/data/', workers=0):
    if dataset_name == 'CIFAR10':
        #dataset_dir += 'cifar-10-batches-py/'
        train_transform_CIFAR, test_transform_CIFAR = _data_transforms_cifar10()
        trainset = torchvision.datasets.CIFAR10(root=dataset_dir, train=True, download=True, transform=train_transform_CIFAR)
        valset = torchvision.datasets.CIFAR10(root=dataset_dir, train=True, download=True, transform=test_transform_CIFAR)
        testset = torchvision.datasets.CIFAR10(root=dataset_dir, train=False, download=True, transform=test_transform_CIFAR)
        num_class = 10
    elif dataset_name == 'STL10':
        train_transform_STL, test_transform_STL = _data_transforms_stl10()
        trainset = torchvision.datasets.STL10(root=dataset_dir, split='train', download=True, transform=train_transform_STL)
        valset = torchvision.datasets.STL10(root=dataset_dir, split='train', download=True, transform=test_transform_STL)
        testset = torchvision.datasets.STL10(root=dataset_dir, split='test', download=True, transform=test_transform_STL)
        num_class = 10
    elif dataset_name == 'TIN':
        dataset_dir += 'tiny-imagenet-200/'
        image_datasets = {x: datasets.ImageFolder(os.path.join(dataset_dir, x), _data_transforms_tinyimagenet()[x])
                          for x in ['train', 'test']}
        trainset = image_datasets['train']
        # valset = image_datasets['train']
        testset = image_datasets['test']
        manager = Manager()
        shared_dict_train = manager.dict()
        # shared_dict_test = manager.dict()
        # trainset = Cache(trainset, shared_dict_train)
        valset =trainset
        # valset = Cache(trainset, shared_dict_train)
        # testset = Cache(testset, shared_dict_test)
        num_class = 200
    else:
        raise Exception('dataset' + dataset_name + 'not supported!')

    if not num_train:
        num_train = len(trainset)

    if not indices:
        indices = validation_set_indices(num_train, valid_percent, dataset_name)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=SubsetRandomSampler(indices[0]), num_workers=workers, pin_memory=False, drop_last=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=False, drop_last=True)
    # test_loader = 0
    if valid_percent:
        validation_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, sampler=SubsetRandomSampler(indices[1]), num_workers=workers, pin_memory=False, drop_last=True)
        # validation_loader = 0
    else:
        validation_loader = 0

    return train_loader, validation_loader, test_loader, indices, num_class

