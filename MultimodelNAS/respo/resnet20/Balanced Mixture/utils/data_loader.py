from random import shuffle

import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler


def _data_transforms_cifar10():
    '''
    CIFAR10 data augmentation and normalization
    :return: training set transforms, test set transforms
    '''
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
def validation_set_indices(num_train, valid_percent, dataset_name):
    '''
    separate randomly the training set to training and validation sets
    :param num_train: training size (currently not used)
    :param valid_percent: what portion of training set to be used for validation
    :return: a list containing [training set indices, validation set indices]
    '''
    train_size = num_train - int(valid_percent * num_train)  # number of training examples
    val_size = num_train - train_size  # number of validation examples
    print('training size:', train_size, ', validation size:', val_size)
    indexes = list(range(num_train))  # available indices at training set
    shuffle(indexes) # shuffle
    indexes = indexes[:num_train] # select the first part
    split = train_size
    train_index = indexes[:split]
    val_index = indexes[split:]
    indices = [train_index, val_index]
    return indices


def data_loader(dataset_name, valid_percent, batch_size, num_train=0, indices=0, dataset_dir='~/Desktop/codes/multires/data/', workers=2):
    '''
    Load dataset with augmentation and spliting of training and validation set
    :param dataset_name: Only for CIFAR10
    :param valid_percent: what portion of training set to be used for validation
    :param batch_size: batch_size
    :param indices: use particular indices rather than randomly separate training set
    :param dataset_dir: dataset directory
    :param workers: number of workers
    :return: train, validation, test data loader, indices, number of classes
    '''
    if dataset_name == 'CIFAR10':
        train_transform_CIFAR, test_transform_CIFAR = _data_transforms_cifar10()
        trainset = torchvision.datasets.CIFAR10(root=dataset_dir, train=True, download=True, transform=train_transform_CIFAR)
        valset = torchvision.datasets.CIFAR10(root=dataset_dir, train=True, download=True, transform=test_transform_CIFAR) # no augmentation for validation set
        testset = torchvision.datasets.CIFAR10(root=dataset_dir, train=False, download=True, transform=test_transform_CIFAR)
        num_class = 10
    else:
        raise Exception('dataset' + dataset_name + 'not supported!')

    if not num_train:
        num_train = len(trainset)

    if not indices: # split and create indices for training and validation set
        indices = validation_set_indices(num_train, valid_percent, dataset_name)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=SubsetRandomSampler(indices[0]), num_workers=workers, pin_memory=True, drop_last=True) # load training set
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True, drop_last=True) # load test set
    if valid_percent: # load validation set if used
        validation_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, sampler=SubsetRandomSampler(indices[1]), num_workers=workers, pin_memory=True, drop_last=True)
    else:
        validation_loader = 0

    return train_loader, validation_loader, test_loader, indices, num_class
