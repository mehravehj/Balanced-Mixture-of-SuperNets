import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from create_model import ResNet20, ResBasicBlock
from utils import *
from utils.lr_scheduler import CosineAnnealingWarmupRestarts
from utils.search_space_design import init_path_logit, sample_uniform, model_prob_calculator, exp_moving_avg, \
    sample_path_norm


def create_models(layers, channels, num_models):
  """
  create n identical models
  """
  models = [ResNet20(ResBasicBlock, layers, channels) for _ in range(num_models)] # create multiple models
  #model2 = copy.deepcopy(model1) # do I make them same initialization?
  return models


def create_optimizers(type, nets, num_models, lr, m, wd, epochs, m_lr, first_cycle_steps, cycle_mult, warmup_steps, gamma):
    optimizers = [optim.SGD(nets[i].parameters(), lr=lr, momentum=m,
                                 weight_decay=wd) for i in range(num_models)]
    if type == 'cosine_anneal':
        schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[i], epochs + 1, eta_min=m_lr) for i in range(num_models)]
    elif type == 'cosine_anneal_wr':
        schedulers = [CosineAnnealingWarmupRestarts(optimizers[i],
                                                  first_cycle_steps=first_cycle_steps,
                                                  cycle_mult=cycle_mult,
                                                  max_lr=lr,
                                                  min_lr=m_lr,
                                                  warmup_steps=warmup_steps,
                                                  gamma=gamma) for i in range(num_models)]

    return optimizers, schedulers


def initialize_prob_matrix(num_paths, num_models, init_paths_w = 1, init_models_w = 1):
  init_w_mat = torch.FloatTensor([[init_models_w for i in range(num_models)] for j in range(num_paths)])  # initialize matrix for model prob
  init_w_paths = init_path_logit(num_paths, init_paths_w)
  return init_w_mat, init_w_paths

def train_valid(models, train_queue, optimizers, sample_weights, paths, weight_mat, temperature, validation_queue, decay, counter_matrix, threshold,
                criterion=nn.CrossEntropyLoss()):
    p_m = []
    # print(weight_mat.size())
    prob = torch.nn.functional.softmax(weight_mat * temperature, dim=1)  # conditional prob p(model|arch)
    marginal = torch.sum(prob, 0).view(1,-1) /prob.size(0) #sum / 36 size is (1,2)
    # print(marginal.size())
    num_models = len(models)
    #define uniform probability
    marginal_uniform = torch.nn.functional.softmax(torch.ones(1,num_models), dim=1)
    # print(marginal_uniform.size())
    # calculate KL divergence
    # print(marginal)
    # print(marginal_uniform)
    kl_dis = abs((marginal * (marginal / marginal_uniform).log()).sum())
    print(kl_dis)
    while kl_dis > threshold:
        # print('calculated marginal', marginal)
        p_m.append(marginal)
        weighted_p = prob/marginal #Calculate
        # print(weighted_p)
        # normalize conditional probs over models
        prob = weighted_p / torch.sum(weighted_p, 1).view(weighted_p.size(0),1)
        # print(prob.size())
        marginal = torch.sum(prob, 0).view(1,-1) /prob.size(0) # marginal
        #update KL distance
        kl_dis = abs((marginal * (marginal / marginal_uniform).log()).sum())
        # print(kl_dis)
        # print(marginal)

    # initializing average per epoch metrics
    train_loss = 0 #average train loss
    validation_loss = 0 #average validation loss
    train_accuracy = 0 #average train accuracy
    validation_accuracy = 0 #average validation accuracy
    # iterate over validation set
    validation_iterator = iter(validation_queue)  # validation set iterator

    for batch_idx, (train_inputs, train_targets) in enumerate(train_queue):
        #print(batch_idx)
        train_acc = 0
        valid_acc = 0
        # sample paths
        path_index, pool = sample_uniform(sample_weights, paths)
        #calculate normalized probabilities
        prob_cond = model_prob_calculator(weight_mat, p_m, temperature)
        # sample model
        model_index = sample_path_norm(prob_cond[path_index,:])
        counter_matrix[path_index, model_index] += 1
        model = models[model_index]
        optimizer = optimizers[model_index]
        model.train()
        model.set_path(pool) # setting path
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
        # print('-------------------')
        # print(path_index, model_index)
        # print(valid_acc_batch)
        # print(weight_mat)
        weight_mat[path_index, model_index] = exp_moving_avg(weight_mat[path_index, model_index], valid_acc_batch, decay)
        # print(weight_mat)
        validation_loss += validation_minibatch_loss.detach().cpu().item()
        validation_accuracy += valid_acc

        # #sanity checks
        # print('selected path:', path_index, path)
        # print('selected model:', model_index)
        # print('counter:', counter_matrix)
        # print('weight_mat:', weight_mat)

    return counter_matrix, weight_mat, prob_cond, prob, train_loss, validation_loss, train_accuracy, validation_accuracy


def validate_all(models, num_models, paths, num_paths, validation_queue):
    print('evaluating of all models on all paths....')
    init_acc_mat = torch.zeros((num_paths, num_models)) # initialize matrix for accuracy
    init_acc_mat_per_class = torch.zeros((num_paths, num_models, 10)) # initialize matrix for accuracy
    for i in range(num_models):
        model = models[i]
        model.eval()
        for j in range(num_paths):
            valid_accuracy_batch = 0
            valid_accuracy_epoch = 0
            per_class = 0
            model.set_path(paths[j])

            for batch_idx, (validation_inputs, validation_targets) in enumerate(validation_queue):
                validation_inputs, validation_targets = validation_inputs.cuda(), validation_targets.cuda()
                validation_outputs = model(validation_inputs)
                valid_acc = calculate_accuracy(validation_outputs, validation_targets)
                valid_accuracy = copy.deepcopy(valid_acc)
                valid_accuracy_batch += valid_accuracy
                per_class += accuracy_per_class(validation_outputs, validation_targets)
                # print(per_class.size())
            valid_acc_epoch = valid_accuracy_batch[0] / valid_accuracy_batch[1]
            valid_acc_epoch_per_class = per_class[0,:] / per_class[1,:]
            init_acc_mat[j,i] = copy.deepcopy(valid_acc_epoch)
            init_acc_mat_per_class[j,i,:] = copy.deepcopy(valid_acc_epoch_per_class)
    return init_acc_mat, init_acc_mat_per_class

# def validate_ensemble(models, num_models, paths, num_paths, prob_cond, validation_queue):
#     print('evaluating images accross models and paths....')
#     init_acc_mat = torch.zeros((num_paths, num_models)) # initialize matrix for accuracy
#     init_acc_mat_per_class = torch.zeros((num_paths, num_models, 10)) # initialize matrix for accuracy
#     for i in range(num_models):
#         model = models[i]
#         model.eval()
#         for j in range(num_paths):
#             valid_accuracy_batch = 0
#             valid_accuracy_epoch = 0
#             per_class = 0
#             model.set_path(paths[j])
#
#             for batch_idx, (validation_inputs, validation_targets) in enumerate(validation_queue):
#                 validation_inputs, validation_targets = validation_inputs.cuda(), validation_targets.cuda()
#                 validation_outputs = model(validation_inputs)
#                 valid_acc = calculate_accuracy(validation_outputs, validation_targets)
#                 valid_accuracy = copy.deepcopy(valid_acc)
#                 valid_accuracy_batch += valid_accuracy
#                 per_class += accuracy_per_class(validation_outputs, validation_targets)
#                 # print(per_class.size())
#             valid_acc_epoch = valid_accuracy_batch[0] / valid_accuracy_batch[1]
#             valid_acc_epoch_per_class = per_class[0,:] / per_class[1,:]
#             init_acc_mat[j,i] = copy.deepcopy(valid_acc_epoch)
#             init_acc_mat_per_class[j,i,:] = copy.deepcopy(valid_acc_epoch_per_class)
#     return init_acc_mat, init_acc_mat_per_class


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
