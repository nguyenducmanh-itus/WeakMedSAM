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
import json
from PIL import Image, ImageDraw

# log_path = "logdir/classifier_train_v3"
# event_acc = EventAccumulator(log_path)
# event_acc.Reload()



# tag_names = ['train/parent loss', 'train/train loss', 'train/child bone loss', 'train/child occurance loss']

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


# json_path = "data/BTXRD/Annotations/IMG000001.json"
# image_path = "data/BTXRD/images/IMG000001.jpeg"
# with open(json_path, mode="r", encoding="utf-8") as read_file :
#     tumor_inf = json.load(read_file)

# shape_list = []

# for i in range(len(tumor_inf["shapes"])) :
#     shape_list.append(tumor_inf["shapes"][i]["points"])

# image = Image.open(image_path)
# draw = ImageDraw.Draw(image)
# draw.rectangle(shape_list[0], outline="red", fill=None, width=1)
# #image.show()
# print(shape_list[0][0])
#Draw bbx tumor in image
plabs = torch.tensor([[1.], [1.], [1.], [1.], [0.]])
bone_mask = (plabs.squeeze() == 1)
print(bone_mask)
