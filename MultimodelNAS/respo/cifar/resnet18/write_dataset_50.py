from argparse import ArgumentParser

import torch
from fastargs import get_current_config
from ffcv.fields import IntField, RGBImageField
from ffcv.writer import DatasetWriter
from torchvision.datasets import CIFAR100


def main():
    train_dataset = 'cifar100_train_50.beton'
    val_dataset = 'cifar100_val_50.beton'


    trainset = CIFAR100('.', train=True, download=True)

    nclass = 100

    indices_list = []
    for i in range(nclass):
        print(i)
        indices_list.append((torch.tensor(trainset.targets)[..., None] == i).any(-1).nonzero(as_tuple=True)[0])

    index = torch.stack(indices_list)
    index1 = index[:, :int(index.size(1) / 2)].reshape(-1)
    index2 = index[:, int(index.size(1) / 2):].reshape(-1)

    data_train = torch.utils.data.Subset(trainset, index1)
    data_val = torch.utils.data.Subset(trainset, index2)




    writer = DatasetWriter(train_dataset, {
        'image': RGBImageField(),
        'label': IntField()
    })
    writer.from_indexed_dataset(data_train)



    writer = DatasetWriter(val_dataset, {
        'image': RGBImageField(),
        'label': IntField()
    })
    writer.from_indexed_dataset(data_val)



if __name__ == "__main__":
    config = get_current_config()
    parser = ArgumentParser(description='Fast CIFAR-10 training')
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()

    main()
