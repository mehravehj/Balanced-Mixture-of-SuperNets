# Balanced Mixture of Supernets for Learning CNN Pooling

This repository contains the release of PyTorch code to replicate all main results, figures and tabels presented in the paper: Balanced Mixture of Supernets for Learning the CNN Pooling Architecture

The repository structure is as follows:
  * `requirements.txt`, contains all required libraries
  * `figures/`, contains all figures represented in the paper 
  
`experiments/Pooling Experiments/` contains: 
  * `resnet20/`, contains code for CIFAR10 experiments with Resnet20 backbone architecture 
  * `cifar/`, contains code for CIFAR10/CIFAR100 experiments with Resnet18/Resnet50 backbone architectures
  * `food101/`, contains code for food101 experiments on Resnet50 backbone architecture 
  
Each folder contains a ReadMe file for more information.
  

## Getting Started
### Install
It's recommended to use Python3 and create a virtual environment first. Then install libraries from `requirements.txt`.

### Dataset Preperation
Please note that for experiments in  `cifar/` and `food101/` , datasets need to be converted to [FFCV](https://ffcv.io/) format. For experimetns with resnet20 this is not required. Furthermore, for NAS methods presented in this paper, training datasets are split 50/50:

   * For cifar datasets experiments (assuming in respective dirctory such as `cifar/resnet18/`), to download and convert them for CIFAR100 run:

   ```bash
   $ ./write_cifar.sh
   ```
To split CIFAR datsets before conversion, run:

   ```bash
   $ ./write_cifar_50.sh
   ```
   * For Food101 tests the dataset should be downloaded from [here](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) and placed in food101 folder. To convert image folder dataset to FFCV format run:

```bash
$ create_dataset.py
```

For our NAS method, split training set to 50/50 subsets per class before conversion by running: 

```bash
$ python split_dataset.py
```

This will create two folders `train_50/` and `val_50/` that can be converted to FFCV files using create_dataset.py .


### Experiments
Experiment contains code to run Balanced Mixture of models as ```main.py``` and an individual training for retraining architectures. See Readme file on each expermints for more details.

NOTE: The code was originally ran on a cluster with 1 GPU NVIDIA V100 Volta (32G HBM2 memory). However, the files are edited to be ran locally.












