import torch 
import numpy as np
my_set = {"a", "b", "c", "d"}
save_map = {idx : np.zeros(8) for idx in my_set}
save_map["a"][5] = np.arange(1, 10, 1)
print(save_map["a"[5]])