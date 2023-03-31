This direcory contains:

* Balanced Mixture Method (our method): run the following with nm argument for different number of models. ```python train_initial_mm.sh -nm 4``` 
  To study the structure of each architecture, consult ``` rn50_lookup_space.csv  ```

*  Satnd-alone or individaul training: run following command with the path number as an argument:
 ``` python train_original_baseline_ffcv100.py -pn 360 ``` 

Use -e tp set number of epochs.
