import numpy as np
import torch
from torch.distributions.categorical import Categorical as Categorical


def to_sum_k_rec(n, k):
    if n == 1:
        yield (k,)
    else:
        for x in range(1, k):
            for i in to_sum_k_rec(n - 1, k - x):
                yield (x,) + i


def create_search_space():
    # print(list(to_sum_k_rec(3, 5))
    path_blocks = list(to_sum_k_rec(4, 16))
    number_paths = len(path_blocks)
    print('all %d paths created: ' % (number_paths))
    print(len(list(to_sum_k_rec(4, 16))))
    all_paths = []
    for p in path_blocks:
        cul_new_block = list(np.cumsum(p))
        cul_new_index = [i + 1 for i in cul_new_block]
        binary_new_block = [1 if i in cul_new_index else 0 for i in range(1, 16)]
        binary_new_block = tuple(binary_new_block)
        all_paths.append(binary_new_block)

    if number_paths != len(all_paths):
        print('bug... please fix')

    return tuple(all_paths), number_paths


def init_path_logit(num_paths, initial_logits):
    initial_path_weights = torch.FloatTensor([initial_logits for i in range(num_paths)])  # initialize logits
    return initial_path_weights


def sample_path_prob(sample_weights, temperature):  # sample weight with a probability
    # The probs argument must be non-negative, finite and have a non-zero sum, and it will be normalized to sum to 1 along the last dimension.
    # Sample, categorical or multinomial
    prob = Categorical(logits=sample_weights * temperature)

    # print('probabilities')
    # print(prob.probs)
    path_index = int(prob.sample().data)
    # sampled_path = paths[path_index]
    return path_index


def sample_path_norm(norm_prob):  # sample weight with a probability
    prob = Categorical(probs=norm_prob)
    path_index = int(prob.sample().data)
    return path_index


def intialize_prob_matrix(num_paths, num_models, init_paths_w=1, init_models_w=1):  # sample weight with a probability
    print('Initializing architecture logits with %d and model logtis with %d...' % (init_paths_w, init_paths_w))
    path_w = init_path_logit(num_paths, init_paths_w)
    weight_mat = torch.ones((num_paths, num_models)) * init_models_w
    # weight_mat = torch.FloatTensor([[init_models_w for i in range(num_models)] for j in range(num_paths)])
    return path_w, weight_mat


def sample_uniform(sample_weights, paths):
    prob = Categorical(logits=sample_weights)
    # print('probabilities')
    # print(prob.probs)
    path_index = int(prob.sample().data)
    # sampled_path = paths[path_index]
    return path_index, paths[path_index]


def exp_moving_avg(old_weights, new_weights, decay):  # calculate exponential moving average
    ema_weights = decay * old_weights + (1 - decay) * new_weights
    return ema_weights


def update_path_logits(path_index, path_weights, new_weights, decay):
    path_weights[path_index] = exp_moving_avg(path_weights[path_index], new_weights, decay)
    return path_weights


def update_model_logits(path_index, model_index, weight_matrix, new_weights, decay):
    weight_matrix[path_index, model_index] = exp_moving_avg(weight_matrix[path_index, model_index], new_weights, decay)
    return weight_matrix


def compute_marginal_prob(prob):
    prob_m = torch.sum(prob, 0) / prob.size(0)  # p(model)
    return prob_m


def model_prob_calculator(weight_matrix, marginals, temperature):
    prob = torch.nn.functional.softmax(weight_matrix * temperature, dim=1)  # conditional prob p(model|arch)
    # print(prob)
    for n in range(len(marginals)):
        p = prob / marginals[n]  # so that p is uniform across models
        sum_over_arch = torch.sum(p, 1).view(prob.size(0), 1)
        prob = p / sum_over_arch  # normalize to make sum_m(p(m|a))=1
        # print('normalize: ', n)
        # print('used marginal: ', marginals[n])
        # print('marginals')
        # print(compute_marginal_prob(prob))
        # print('probs')
        # print(prob)
    return prob
