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
X = np.random.multivariate_normal(means, cov, size = n_samples)

x = torch.randn((1, 3, 256 + 8, 256 + 8))
prj = nn.Conv2d(3, 768, kernel_size=16, stride=8, padding=0)
x = prj(x)
print(x.shape)