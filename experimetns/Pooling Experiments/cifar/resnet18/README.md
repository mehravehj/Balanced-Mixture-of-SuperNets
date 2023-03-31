This folder containscode for Balanced Mixture of Supernets with resnet18 backbone for CIFAR10 dataset.

This direcory contains:

* Balanced Mixture Method (our method): run the following with nm argument for different number of models. ```train_balanced_example.sh -nm 4``` 

To study the structure of each architecture, consult ``` rn18_lookup_space.csv  ```

*  Satnd-alone or individaul training: run following command with the path number (pn) as an argument:
``` python train_original_baseline_ffcv.py -pn 360 ``` 
