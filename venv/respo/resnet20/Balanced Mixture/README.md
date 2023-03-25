Code for Balanced Mixture of Supernets with resnet20 backbone. Run:

```bash
$ python main.py -nm 4
```

Change nm argument to test with different number of model (M). The code will output an architecture in terms of poolign locations in layers.
To train obtained architectures individually, run with the output -p as:

```bash
$ python main_base.py -p 0,0,0,0,1,0,0,1,0,0
```
