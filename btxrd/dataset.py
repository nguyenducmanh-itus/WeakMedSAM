import torch
from torch.utils.data import Dataset
from PIL import Image, ImageChops
import os
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random
import numpy as np
import pandas as pd
import pickle 
pic_size = 256

class BTRXD_Dataset(Dataset) : 
    def __init__(self, images, parent_classes : int, 
                 images_path = "./data/BTXRD/images", 
                 path_df = "./data/BTXRD/dataset.xlsx") :
        self.images = images
        self.parent_classes = parent_classes
        self.images_path = images_path
        self.df_path = path_df
        self.df = pd.read_excel(self.df_path)
    def __len__(self) :
        return len([name for name in os.listdir(self.images_path) 
                    if os.path.isfile(os.path.join(self.images_path, name))])
    def __getitem__(self, idx : Any) -> Any :
        
        return 

def get_all_datasets(data_path) :
    with open("./split/all.txt") as f :
        file_list = [line.strip() for line in f]
        
    dataset = BTRXD_Dataset(file_list, parent_classes = 9)
    return dataset

    
dataset = BTRXD_Dataset(parent_classes = 8)
num_images = dataset.__len__()
print(num_images)