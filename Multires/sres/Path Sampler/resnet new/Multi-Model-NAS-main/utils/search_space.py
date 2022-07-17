import torch
from torch.distributions.categorical import Categorical as Categorical
# import torch.nn as nn
# import torch.nn.functional as F
# # from multiscale_blocks import *
#

# encode the path
# leng = 10 # number of layers
# num_scales = 4 # number of available scales can be automatically selected

class Pooling_search_space():
  
  def __init__(self, num_layers, num_scales, monotonic=True):
    self.num_layers = num_layers
    self.num_scales = num_scales
    self.search_space = None
    self.search_size = 0
    
    
  def create_architectures(self):
    # create intial tuple based on layers and scales
    num_pooling = num_scales - 1 # number of pooling layers to insert
    num_available_layers = num_layers - num_scales # number of availble layers to insert pooling on
    for positions in combinations(range(num_available_layers), num_pooling):
      p = [0] * num_available_layers

      for i in positions:
          p[i] = 1

      yield tuple(p)
    
      paths = tuple([(0,) + i for i in p)])
      number_paths = len(paths)
      self.search_space = paths
      self.search_size = number_paths
      return paths, number_paths

  def init_path_logit(num_paths, initial_logits):
      initial_path_weights = torch.FloatTensor([initial_logits for i in range(num_paths)])  # initialize logits
      return initial_path_weights

  def sample_path(sample_weights, temperature):
      # The probs argument must be non-negative, finite and have a non-zero sum, and it will be normalized to sum to 1 along the last dimension.
      # Sample, categorical or multinomial
      prob = Categorical(logits=sample_weights * temperature)
      # print('probabilities')
      # print(prob.probs)
      path_index = int(prob.sample().data)
      # sampled_path = paths[path_index]
      return path_index

  def exp_moving_avg(old_weights, new_weights, decay):
      ema_weights = decay * new_weights + (1-decay) * old_weights
      return ema_weights

  def update_path_prob(path_index, path_weights, new_weights, decay):
      path_weights[path_index] = exp_moving_avg(path_weights[path_index], new_weights, decay)
      return path_weights
