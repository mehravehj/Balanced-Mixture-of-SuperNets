Code for Balanced Mixture of Supernets with resnet20 backbone. Run:

```bash
$ python main.py -nm 4
```

Change nm argument to test with different number of model (M). To trained obtained architectures individually, run:

```bash
$ python main_base.py -p 0,0,0,0,1,0,0,1,0,0
```
