This folder contains code for Balanced Mixture of Supernets with resnet18 backbone for CIFAR10 dataset.

This direcory contains:

* Balanced Mixture Method (our method): run the following with -nm argument for different number of models, e.g. ```python main.py -nm 4```  for 4 models.

  To study the structure of each architecture, consult ``` rn18_lookup_space.csv  ```

*  Satnd-alone or individaul training: run following command with the path number (-pn) as an argument, example:
  ``` python train_original_baseline_ffcv.py -pn 360 ``` 
  
  
 Use -e to set number of epochs to train.
