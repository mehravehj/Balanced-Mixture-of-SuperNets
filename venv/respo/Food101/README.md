This folder containscode for Balanced Mixture of Supernets with resnet50 backbone for Food101 dataset.

## Dataset
This dataset consists of 101 food categories, with total of 101,000 images with split. For each class 750/250 training and test data are provided. In our experiments, all images were rescaled to have a maximum side length of 256 pixels when converting to FFCV data files. The detailed information can be found this website https://www.vision.ee.ethz.ch/datasets_extra/food-101/

## Pre-processing Data
To convert image folder dataset to FFCV format run:

```bash
$ create_dataset.py
```

For our NAS method, split training set to 50/50 subsets per class before conversion by running: 

```bash
$ python split_dataset.py
```

This will create two folders `train_50/` and `val_50/` that can be converted to FFCV files.

## Experiments
This direcory contains:

* Balanced Mixture Method (our method): run the following with nm argument for different number of models. ```python train_initial_mm.sh -nm 4``` 

Balanced Mixture Method will output architecture number and their evaluation accuarcy. To study the structure of each architecture, consult ``` rn50_lookup_space.csv  ```

*  Satnd-alone or individaul training: run following command with the path number as an argument:
``` python train_original_baseline.py -pn 360 ``` 


