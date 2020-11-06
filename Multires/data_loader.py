import torch
import torchvision
import torchvision.transforms as transforms
from random import shuffle
from torch.utils.data import Dataset
import torch.utils.data
from  torch.utils.data.sampler import SubsetRandomSampler


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


def validation_set_indices(num_train, valid_percent):
    train_size = num_train - int(valid_percent * num_train)  # number of training examples
    val_size = num_train - train_size  # number of validation examples
    indices = list(range(num_train))  # available indices at training set
    shuffle(indices)
    indices = indices[:num_train]
    split = train_size
    train_index = indices[:split]
    val_index = indices[split:]
    indices = [train_index, val_index]
    return indices


def data_loader(dataset_name, valid_percent, batch_size, num_train=0, indices=0, dataset_dir='~/Desktop/codes/multires/data/', workers=0):
    if dataset_name == 'CIFAR10':
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
    else:
        raise Exception('dataset' + dataset_name + 'not supported!')

    if not num_train:
        num_train = len(trainset)

    if not indices:
        indices = validation_set_indices(num_train, valid_percent)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=SubsetRandomSampler(indices[0]), num_workers=workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    if valid_percent:
        validation_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, sampler=SubsetRandomSampler(indices[1]), num_workers=workers, pin_memory=True)
    else:
        validation_loader = 0

    return train_loader, validation_loader, test_loader, indices, num_class

