import torch
import torchvision
import torchvision.transforms as transforms
from random import shuffle
from torch.utils.data import Dataset
import torch.utils.data
from torch.utils.data.sampler import SubsetRandomSampler

def _data_transforms_cifar10():
    train_transform = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return train_transform, test_transform


def validation_set_indices(num_train, valid_percent, dataset_name):
    train_size = num_train - int(valid_percent * num_train)  # number of training examples
    print('training size:', train_size)
    val_size = num_train - train_size  # number of validation examples
    indexes = list(range(num_train))  # available indices at training set
    shuffle(indexes)
    indexes = indexes[:num_train]
    split = train_size
    train_index = indexes[:split]
    val_index = indexes[split:]
    indices = [train_index, val_index]
    return indices


def data_loader(dataset_name, valid_percent, batch_size, indices=0, dataset_dir='~/Desktop/codes/multires/data/', workers=0):
    if dataset_name == 'CIFAR10':
        #dataset_dir += 'cifar-10-batches-py/'
        train_transform_CIFAR, test_transform_CIFAR = _data_transforms_cifar10()
        trainset = torchvision.datasets.CIFAR10(root=dataset_dir, train=True, download=True, transform=train_transform_CIFAR)
        valset = torchvision.datasets.CIFAR10(root=dataset_dir, train=True, download=True, transform=test_transform_CIFAR)
        testset = torchvision.datasets.CIFAR10(root=dataset_dir, train=False, download=True, transform=test_transform_CIFAR)
    else:
        raise Exception('dataset' + dataset_name + 'not supported!')
    num_train = len(trainset)

    if not indices:
        indices = validation_set_indices(num_train, valid_percent, dataset_name)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=SubsetRandomSampler(indices[0]), num_workers=workers, pin_memory=False, drop_last=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=False, drop_last=True)
    if valid_percent:
        validation_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, sampler=SubsetRandomSampler(indices[1]), num_workers=workers, pin_memory=False, drop_last=True)
    else:
        validation_loader = 0

    return train_loader, validation_loader, test_loader, indices

