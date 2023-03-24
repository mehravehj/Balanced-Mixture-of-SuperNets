# Balanced Mixture of Supernets for Learning CNN Pooling

This repository contains the release of PyTorch code to replicate all main results, figures and tabels presented in the paper: Balanced Mixture of Supernets for Learning the CNN Pooling Architecture

The repository structure is as follows:
  * `figures/`, contains all figures represented in the paper 
  * `cifar/`, contains code for CIFAR10/CIFAR50 experiments on Resnet18/Resnet50 backbone architectures
  * `resnet20/`, contains code for CIFAR10 experiments on Resnet20 backbone architecture 
  * `food101/`, contains code for food101 experiments on Resnet50 backbone architecture 
  

## Getting Started
### Install
It's recommended to use Python3 and create a virtual environment first. 
To run the the experiments, install the following packages:

```bash
$ ./install.sh
```

### Dataset Preperation
For ResNet18 and ResNet50 experiments, datasets need to be converted to [FFCV](https://ffcv.io/) format. For each experiments (assuming in respective dirctory such as `cifar/resnet18/), to download and convert them for CIFAR100 run 

```bash
$ ./write_cifar.sh
```

For NAS methods presented in this paper, dataset are split 50/50. To split CIFAR datsets before conversion, run:

```bash
$ ./write_cifar_50.sh
```











