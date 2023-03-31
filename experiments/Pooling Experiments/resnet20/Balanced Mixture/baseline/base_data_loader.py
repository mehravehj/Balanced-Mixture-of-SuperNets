import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms


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


def data_loader(dataset_name, batch_size, dataset_dir='~/Desktop/codes/multires/data/', workers=2):
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
        testset = torchvision.datasets.CIFAR10(root=dataset_dir, train=False, download=True, transform=test_transform_CIFAR)
        num_class = 10
    else:
        raise Exception('dataset' + dataset_name + 'not supported!')

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, drop_last=True) # load training set
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True, drop_last=True) # load test set

    return train_loader, test_loader, num_class
