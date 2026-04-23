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

def get_id_multiple_disea(df, begin_feature, end_feature) :
    subset = df.loc[:, begin_feature : end_feature]
    mask = (subset == 1).sum(axis = 1) >=2
    return df[mask].index



class BTRXD_Dataset(Dataset) : 
    def __init__(
        self,
        image_list,
        cluster_file: str,
        begin_feature: str,
        end_feature: str,
        child_classes: int,
        images_path="./data/BTXRD/images",
        df_path="./data/BTXRD/dataset.xlsx",
    ):
        self.images = image_list
        self.images_path = images_path
        self.child_classes = child_classes
        #Load dataframe 
        df = pd.read_excel(df_path)
        #Remove multiple-labels samples
        drop_indices = get_id_multiple_disea(df, begin_feature, end_feature)
        self.df = df.drop(drop_indices).reset_index(drop = True)
        self.parent_features = self.df.loc[:, begin_feature : end_feature]
        self.parent_classes = len(self.parent_features.columns)
        self.label_map = {name : i + 1 
                          for i, name in enumerate(self.parent_features.columns)}
        if self.child_classes != 0 :
            with open(cluster_file, "rb") as f : 
                self.clabs = pickle.load(f)
    
    #Get length of dataset                        
    def __len__(self) :
        return len(self.images)
    
    #Load image, resize and convert to tensor
    def load_image(self, path) :
        image = Image.open(path).convert("RGB")
        image = image.resize((pic_size, pic_size))
        return TF.to_tensor(image)
    
    def __getitem__(self, index) :
        #Load image converted
        img = self.load_image(self.images[index])
        #Get name idx of image
        idx = os.path.splitext(self.images[index].split("/")[-1])[0]
        #Assign primary label for image
        #Include parent_claases + 1 primary clab with the first element to checks
        #whether or not there is a disea.
        plab = torch.zeros(self.parent_classes + 1).float()
        #Assign label for disea k
        col_plab = self.parent_features.columns[self.parent_features.loc[index] == 1]
        have_pclass = False
        if not col_plab.empty :
            plab[self.label_map[col_plab.item()]] = 1
            pclass =  self.label_map[col_plab.item()]
            have_pclass = True
        else : 
            plab[0] = 1
        if self.child_classes != 0 : 
            clab = torch.zeros(self.child_classes * self.parent_classes + 1).float()
            if have_pclass == True : 
                clab[self.clabs[idx][pclass] + 1] = 1 
            else : 
                clab[0] = 1
            return {
                "img": img,
                "plab": plab,
                "clab": clab,
                "idx": idx,
                "fname": self.images[index],
            }
        else : 
            return {
                "img": img,
                "plab": plab,
                "idx": idx,
                "fname": self.images[index],
            }                        
        
def get_dataset() :
    return 
        

def get_all_dataset(data_path, cluster_file, 
                     begin_features, end_features, 
                     child_classes) :
    with open("./btxrd/splits/all.txt") as f :
        file_list = [line.strip() for line in f]
        
    dataset = BTRXD_Dataset(image_list = file_list, 
                            cluster_file = cluster_file, 
                            begin_feature = begin_features, 
                            end_feature = end_features,
                            child_classes = child_classes)
    return dataset

    

