# from random import shuffle
#
# import torch
# import torch.utils.data
# import torchvision
# import torchvision.transforms as transforms
# from torch.utils.data import Dataset
# from torch.utils.data.sampler import SubsetRandomSampler
# import torch.utils.data
# import torch.utils.data.distributed
import numpy as np
from ffcv.fields.basics import IntDecoder
from ffcv.fields.rgb_image import SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, RandomHorizontalFlip, ToTorchImage, \
    RandomTranslate


def data_loader(gpu, fp32, datset_dir, train_file, val_file, batch_size, workers, np_seed):
    # IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
    # IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
    train_file = datset_dir + '/' + train_file
    val_file = datset_dir + '/' + val_file


    cifar_mean = np.array([0.49139968, 0.48215827, 0.44653124])* 255
    cifar_std = np.array([0.2023, 0.1994, 0.2010])* 255

    train_image_pipeline = [
        SimpleRGBImageDecoder(),
        RandomHorizontalFlip(),
        RandomTranslate(padding=2),
        ToTensor(),
        ToDevice(f'cuda:{gpu}', non_blocking=True),
        ToTorchImage(),
        NormalizeImage(cifar_mean, cifar_std, np.float16 if not fp32 else np.float32),
    ]

    val_image_pipeline = [
        SimpleRGBImageDecoder(),
        ToTensor(),
        ToDevice(f'cuda:{gpu}', non_blocking=True),
        ToTorchImage(),
        NormalizeImage(cifar_mean, cifar_std, np.float16 if not fp32 else np.float32)
    ]

    label_pipeline = [IntDecoder(), ToTensor(), Squeeze(),
                      ToDevice(f'cuda:{gpu}', non_blocking=True)]


    train_loader = Loader(train_file, batch_size=batch_size, num_workers=workers,
                          order=OrderOption.RANDOM, os_cache=True, drop_last=True,
                          pipelines={'image': train_image_pipeline, 'label': label_pipeline},
                          distributed=True, seed=np_seed)


    val_loader = Loader(val_file, batch_size=batch_size, num_workers=workers,
                        order=OrderOption.RANDOM, os_cache=True, drop_last=True,
                        pipelines={'image': val_image_pipeline, 'label': label_pipeline},
                        distributed=True, seed=np_seed)

    return train_loader, val_loader
