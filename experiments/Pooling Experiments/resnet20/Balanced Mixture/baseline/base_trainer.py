import torch
import torch.nn as nn
import torch.optim as optim
from baseline.create_model_baseline import ResNet20, ResBasicBlock
from utils import *
import torch.utils.data
import copy

def create_models(layers, channels):
  model = ResNet20(ResBasicBlock, layers, channels)
  return model


def create_optimizers(net, lr, m, wd, epochs, m_lr):
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=m, weight_decay=wd)
    # optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs + 1, eta_min=m_lr)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [80, 120, 160, 180], gamma=0.1, last_epoch=- 1, verbose=False)
    return optimizer, scheduler

def train(model, train_queue, optimizer, criterion=nn.CrossEntropyLoss()):
    # initializing average per epoch metrics
    train_loss = 0 #average train loss
    train_accuracy = 0 #average train accuracy

    for batch_idx, (train_inputs, train_targets) in enumerate(train_queue):
        #print(batch_idx)
        train_acc = 0
        per_class = 0
        model.train()
        train_inputs, train_targets = train_inputs.cuda(), train_targets.cuda()
        optimizer.zero_grad()
        train_outputs = model(train_inputs)
        train_minibatch_loss = criterion(train_outputs, train_targets)
        train_minibatch_loss.backward()
        optimizer.step()
        train_loss += train_minibatch_loss.detach().cpu().item()
        train_acc = calculate_accuracy(train_outputs, train_targets)
        train_acc = copy.deepcopy(train_acc)
        train_accuracy += train_acc
        per_class += accuracy_per_class(train_outputs, train_targets)
    train_acc_epoch_per_class = per_class[0, :] / per_class[1, :]
    train_loss = train_loss / (batch_idx + 1)
    return train_loss, train_accuracy, train_acc_epoch_per_class


def test(model, test_queue, criterion=nn.CrossEntropyLoss()):
    model.eval()
    val_loss = 0
    valid_accuracy = 0
    per_class = 0
    for batch_idx, (validation_inputs, validation_targets) in enumerate(test_queue):
        validation_inputs, validation_targets = validation_inputs.cuda(), validation_targets.cuda()
        validation_outputs = model(validation_inputs)
        valid_minibatch_loss = criterion(validation_outputs, validation_targets)
        val_loss += valid_minibatch_loss.detach().cpu().item()
        valid_acc = calculate_accuracy(validation_outputs, validation_targets)
        valid_acc = copy.deepcopy(valid_acc)
        valid_accuracy += valid_acc
        per_class += accuracy_per_class(validation_outputs, validation_targets)
    valid_acc_epoch_per_class = per_class[0, :] / per_class[1, :]
    val_loss = val_loss / (batch_idx + 1)
    return val_loss, valid_accuracy, valid_acc_epoch_per_class

def accuracy_per_class(logits, target):
    correct_class = torch.zeros(10)
    total_class = torch.zeros(10)
    _, test_predicted = logits.max(1)
    for c in range(10):
        total_class[c] = target.eq(c).sum().item()
        correct_class[c] = (test_predicted.eq(target) * target.eq(c)).sum()
    return torch.vstack((correct_class, total_class))

def calculate_accuracy(logits, target):
    _, test_predicted = logits.max(1)
    batch_total = target.size(0)
    batch_correct = test_predicted.eq(target).sum().item()
    return torch.tensor([batch_correct, batch_total])
