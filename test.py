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
from tqdm import tqdm
from samus.build_sam_us import samus_model_registry

# log_path = "runs"
# event_acc = EventAccumulator(log_path)
# event_acc.Reload()


##This following code save loss to .txt
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
# img_list = []
# original_img = Image.open(image_path).convert("RGB")
# original_w, original_h = original_img.size #width, heights
# original_img = original_img.resize((512, 512))
# image_bbx = original_img.copy()
# image_mask = image_bbx.copy()
# draw_bbx = ImageDraw.Draw(image_bbx)
# draw_mask = ImageDraw.Draw(image_mask)
# shape_list = []


# for i in range(len(tumor_inf["shapes"])) :
#     shape_list.append(tumor_inf["shapes"][i]["points"])


# ratio_resize_img_w, ratio_resize_img_h = (512 / original_w, 512 / original_h)     

# for i in range(len(shape_list[0])) :
#     shape_list[0][i][0] *= ratio_resize_img_w
#     shape_list[0][i][1] *= ratio_resize_img_h

# for i in range(len(shape_list[1])) :
#     shape_list[1][i][0] *= ratio_resize_img_w
#     shape_list[1][i][1] *= ratio_resize_img_h

# new_img = Image.new("RGB", size=(512 * 3, 512))
# num_img = 3


# #draw.rectangle(shape_list[0], outline = "red", fill=None, width=2)
# draw_bbx.rectangle(shape_list[0], outline = "red", fill=None, width=2)
# draw_mask.polygon(shape_list[1], outline="red", fill="red", width=1)
# img_list = [original_img, image_bbx, image_mask]
# for i in range(num_img) :
#     new_img.paste(img_list[i], (i * 512, 0))

# new_img.save("Mask.jpg")
# new_img.show()


#print(shape_list[0])
pt_dir = "output_patch_image"
pt_file = [os.path.join(pt_dir, file_name) for file_name in os.listdir(pt_dir)]
# for f in pt_file : 
#     data = torch.load(
#                     f,
#                     weights_only=False
#                 )
#     if len(torch.tensor([data["label"]])) == 0:
#         print(f) 
   
t = torch.tensor([])
print(len(t))
