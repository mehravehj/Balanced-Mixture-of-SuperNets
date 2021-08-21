import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import csv

test_name  = 7000
checkpoint = torch.load('./checkpoint/ckpt_' + str(test_name) + '.t7')
test_properties = checkpoint['test_properties']
accuracy_progress = checkpoint['accuracy_progress']
best_accuracy = checkpoint['best_accuracy']
epoch = checkpoint['epoch']
train = max(accuracy_progress['train'])
dataset = test_properties['dataset']
type = test_properties['net_type']
block = test_properties['mscale']
depth = test_properties['leng']
channels = test_properties['channels']
vp = test_properties['validation_percent']
print(train)


with open('test_results.csv', mode='w') as csv_file:
    fieldnames = ['test_name', 'dataset', 'type', 'block', 'depth', 'channels', 'epoch', 'train', 'test', 'vp']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    writer.writeheader()
    for i in range(test_name,test_name+100):
        try:
            checkpoint = torch.load('./checkpoint/ckpt_' + str(i) + '.t7')
            test_properties = checkpoint['test_properties']
            accuracy_progress = checkpoint['accuracy_progress']
            test = checkpoint['best_accuracy']
            epoch = checkpoint['epoch']
            train = max(accuracy_progress['train'])
            dataset = test_properties['dataset']
            type = test_properties['net_type']
            block = test_properties['mscale']
            depth = test_properties['leng']
            channels = test_properties['channels']
            vp = test_properties['validation_percent']

            writer.writerow({'test_name': i, 'dataset': dataset, 'type': type, 'block': block, 'depth': depth, 'channels': channels,
                             'epoch': epoch, 'train': train, 'test': test, 'vp': vp})
        except:
            pass
