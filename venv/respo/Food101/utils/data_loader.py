import numpy as np
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, RandomHorizontalFlip, ToTorchImage
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, RandomResizedCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder


def data_loader(gpu, fp32, datset_dir, train_file, val_file, batch_size, workers, np_seed):
    train_file = datset_dir + '/' + train_file
    val_file = datset_dir + '/' + val_file
    IMAGENET_MEAN = np.array([0.561, 0.440, 0.312]) * 255
    IMAGENET_STD = np.array([0.252, 0.256, 0.259]) * 255

    label_pipeline = [IntDecoder(), ToTensor(), Squeeze(),
                      ToDevice(f'cuda:{gpu}', non_blocking=True)]

    train_image_pipeline = [
        RandomResizedCropRGBImageDecoder((224, 224)),
        RandomHorizontalFlip(),
        ToTensor(),
        ToDevice(f'cuda:{gpu}', non_blocking=True),
        ToTorchImage(),
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16 if not fp32 else np.float32),
    ]


    val_image_pipeline = [
        CenterCropRGBImageDecoder((224, 224), ratio=224 / 256),
        ToTensor(),
        ToDevice(f'cuda:{gpu}', non_blocking=True),
        ToTorchImage(),
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16 if not fp32 else np.float32)
    ]

    train_loader = Loader(train_file, batch_size=batch_size, num_workers=workers,
                          order=OrderOption.RANDOM, os_cache=True, drop_last=True,
                          pipelines={'image': train_image_pipeline, 'label': label_pipeline},
                          distributed=True, seed=np_seed)


    val_loader = Loader(val_file, batch_size=batch_size, num_workers=workers,
                        order=OrderOption.RANDOM, os_cache=True, drop_last=True,
                        pipelines={'image': val_image_pipeline, 'label': label_pipeline},
                        distributed=True)
    return train_loader, val_loader
