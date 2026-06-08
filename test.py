import pickle 
from sklearn.cluster import KMeans
from sklearn.exceptions import NotFittedError
import numpy as np 
import math
import torch
import torch.nn as nn
import torch.nn.functional as F 
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
import os

log_path = "logdir/classifier_train_v2"
event_acc = EventAccumulator(log_path)
event_acc.Reload()



# tag_names = ['train/parent loss', 'train/train loss', 'train/child bone loss', 'train/child oc loss']

# with open("train_loss.txt", "w") as f :
#    for tag_name in tag_names :
#       if tag_name in event_acc.Tags()['scalars'] :         
#             events = event_acc.Scalars(tag_name)
#             df = pd.DataFrame([(e.step, e.value) for e in events], columns=['Step', 'Value'])
#             f.write(tag_name)
#             f.write("\n")
#             for i in range(len(df)) :
#                f.write(str(df.loc[i, "Step"]))
#                f.write(" ")
#                f.write(str(df.loc[i, "Value"]))
#                f.write("\n")
               

my_tensor = torch.tensor([[0.8, 0.5, 0.1], [0.55, 0.65, 0.75]])
