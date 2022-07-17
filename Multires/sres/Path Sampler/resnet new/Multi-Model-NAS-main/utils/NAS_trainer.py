import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from create_model import ResNet20, ResBasicBlock
from utils import *
from utils.search_space_design import init_path_logit, sample_uniform, sample_path_prob, exp_moving_avg


def create_models(layers, channels, num_models):
  """
  create n identical models
  """
  models = [ResNet20(ResBasicBlock, layers, channels) for _ in range(num_models)] # create multiple models
  #model2 = copy.deepcopy(model1) # do I make them same initialization?
  return models


def create_optimizers(type, nets, num_models, lr, m, wd, epochs, m_lr):
    optimizers = [optim.SGD(nets[i].parameters(), lr=lr, momentum=m,
                                 weight_decay=wd) for i in range(num_models)]
    schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[i], epochs + 1, eta_min=m_lr) for i in range(num_models)]
    return optimizers, schedulers


def initialize_prob_matrix(num_paths, num_models, init_paths_w = 1, init_models_w = 1):
  init_w_mat = torch.FloatTensor([[init_models_w for i in range(num_models)] for j in range(num_paths)])  # initialize matrix for model prob
  init_w_paths = init_path_logit(num_paths, init_paths_w)
  return init_w_mat, init_w_paths

def train_valid(models, train_queue, optimizers, sample_weights, paths, weight_mat, temperature, validation_queue, decay, counter_matrix,
                criterion=nn.CrossEntropyLoss()):
    # initializing average per epoch metrics
    train_loss = 0 #average train loss
    validation_loss = 0 #average validation loss
    train_accuracy = 0 #average train accuracy
    validation_accuracy = 0 #average validation accuracy
    # iterate over validation set
    validation_iterator = iter(validation_queue)  # validation set iterator

    for batch_idx, (train_inputs, train_targets) in enumerate(train_queue):
        train_acc = 0
        valid_acc = 0
        # sample paths
        path_index, path = sample_uniform(sample_weights, paths)
        #sample model
        model_index = sample_path_prob(weight_mat[path_index,:], temperature)
        counter_matrix[path_index, model_index] += 1
        model = models[model_index]
        optimizer = optimizers[model_index]
        model.train()
        model.set_path(path) # setting path
        train_inputs, train_targets = train_inputs.cuda(), train_targets.cuda()
        optimizer.zero_grad()
        train_outputs = model(train_inputs)
        train_minibatch_loss = criterion(train_outputs, train_targets)
        train_minibatch_loss.backward()
        optimizer.step()
        train_loss += train_minibatch_loss.detach().cpu().item()
        train_acc = calculate_accuracy(train_outputs, train_targets)
        train_accuracy += train_acc
        # validation
        model.eval()
        try:
          validation_inputs, validation_targets = next(validation_iterator)
        except:
          validation_iterator = iter(validation_queue)
          validation_inputs, validation_targets = next(validation_iterator)
        validation_inputs, validation_targets = validation_inputs.cuda(), validation_targets.cuda()
        validation_outputs = model(validation_inputs)
        validation_minibatch_loss = criterion(validation_outputs, validation_targets)

        # update weight matrix
        valid_acc = copy.deepcopy(calculate_accuracy(validation_outputs, validation_targets))
        valid_acc_batch = copy.deepcopy(valid_acc[0] / valid_acc[1])
        weight_mat[path_index, model_index] = exp_moving_avg(weight_mat[path_index, model_index], valid_acc_batch, decay)
        validation_loss += validation_minibatch_loss.detach().cpu().item()
        validation_accuracy += valid_acc

        # #sanity checks
        # print('selected path:', path_index, path)
        # print('selected model:', model_index)
        # print('counter:', counter_matrix)
        # print('weight_mat:', weight_mat)

    return counter_matrix, weight_mat, train_loss, validation_loss, train_accuracy, validation_accuracy


def validate_all(models, num_models, paths, num_paths, validation_queue):
    print('evaluating of all models on all paths....')
    init_acc_mat = torch.zeros((num_paths, num_models)) # initialize matrix for model prob
    valid_accuracy_batch = 0
    for i in range(num_models):
        model = models[i]
        model.eval()
        for j in range(num_paths):
            model.set_path(paths[j])
            for batch_idx, (validation_inputs, validation_targets) in enumerate(validation_queue):
                validation_inputs, validation_targets = validation_inputs.cuda(), validation_targets.cuda()
                validation_outputs = model(validation_inputs)
                valid_acc = calculate_accuracy(validation_outputs, validation_targets)
                valid_accuracy = copy.deepcopy(valid_acc)
                valid_accuracy_batch += valid_accuracy
            valid_acc_epoch = valid_accuracy_batch[0] / valid_accuracy_batch[1]
            init_acc_mat[j,i] = copy.deepcopy(valid_acc_epoch)
    return init_acc_mat


def calculate_accuracy(logits, target):
    _, test_predicted = logits.max(1)
    batch_total = target.size(0)
    batch_correct = test_predicted.eq(target).sum().item()
    return torch.tensor([batch_correct, batch_total])

