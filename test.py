import pickle 
from sklearn.cluster import KMeans
from sklearn.exceptions import NotFittedError
import numpy as np 
import math
import torch
import torch.nn as nn
import torch.nn.functional as F 
np.random.default_rng(seed = 42)

means = [1.0, 3.0, 4.0]
cov = [[0.5, 0.8, 2.0], 
       [0.8, 1.0, 1.5],
       [2.0, 1.5, 3.0]
    ]

n_samples = 100
#X = np.random.multivariate_normal(means, cov, size = n_samples)

num_samples = 10
num_classes = 5
input = torch.randn((num_samples, num_classes))
target = torch.randint(0, 2, (num_samples, num_classes)).float()
print(target)