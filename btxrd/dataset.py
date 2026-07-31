import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image, ImageChops
import torch.nn.functional as F
import random
import os
import pandas as pd
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import pickle

pic_size = 256

def aug(img: Image.Image, seg: Image.Image) :
    rotate_angle = random.randrange(-20, 20)
    img = TF.rotate(img, rotate_angle)
    seg = TF.rotate(img, rotate_angle)
    jitter = T.ColorJitter(brightness=0.5, contrast=0.5)
    img = jitter(img)
    img = TF.to_tensor(img.resize((pic_size, pic_size)))
    seg = TF.to_tensor(seg.resize((pic_size, pic_size)))
    seg[seg > 0.5] = 1
    return img, seg

def no_aug(img: Image.Image, seg: Image.Image) : 
    img = img.resize((pic_size, pic_size))
    seg = seg.resize((pic_size, pic_size))
    img, seg = TF.to_tensor(img), TF.to_tensor(seg)
    return img, seg

class BTXRD(Dataset) :
    def __init__(self, imgs, segs, df_path, train : bool, 
                 child_classes : int, cluster_file : str) :
        super().__init__()
        self.imgs = imgs
        self.segs = segs
        self.df = pd.read_excel(df_path)
        self.train = train
        if child_classes != 0 :
            with open(cluster_file, "rb") as f : 
                self.clab = pickle.load(f)
        self.child_classes = child_classes

    def __len__(self) :
        return len(self.imgs)
    
    def __getitem__(self, index) :
        img = Image.open(self.imgs[index]).convert("RGB")
        if self.segs[index] is not None :
            seg = Image.open(self.segs[index]).convert("F")
        else : 
            seg = Image.new("F", img.size, color = 0.0)
        img, seg = aug(img, seg) if self.train else no_aug(img, seg)
        plab = torch.zeros(1).float()
        plab[0] = 1 if self.df.loc[index, "tumor"] == 1 else 0
        idx = self.imgs[index].split("/")
        idx = os.path.splitext(idx[-1])[0]
        if self.child_classes != 0 :
            clab = torch.zeros(self.child_classes *  + 1).float()
            part_clab[int(self.clab[idx][0])] = 1
            oc_clab = torch.zeros(self.part_child_classes * self.oc_child_classes + 1).float()
            if plab[0] != 0 :
                oc_clab[int(self.oc_clab[idx][0] * (self.part_clab[idx][0] + 1)) + 1] = 1
            else : 
                oc_clab[0] = 1
            return {
                "img" : img,
                "plab" : plab,
                "bone_clab" : part_clab,
                "oc_clab" : oc_clab,  
                "seg" : seg,
                "idx" : idx, 
                "fname" : self.imgs[index]
            }
                
        

def get_dataset(df_path, data_path : str, 
                part_child_classes : int, oc_child_classes : int, 
                part_cluster_file : str, oc_cluster_file : str) :
    def get_file(samples) :
        seg = None
        img = os.path.join(data_path, samples)
        img = img.replace("\\", "/")
        return img, seg
    samples_train = [get_file(sample.strip())
               for sample in list(open("btxrd/splits/train.txt"))]
    img_train, seg_train = zip(*samples_train)
    samples_val = [get_file(sample.strip())
               for sample in list(open("btxrd/splits/train.txt"))]
    img_val, seg_val = zip(*samples_val)
    samples_test = [get_file(sample.strip())
               for sample in list(open("btxrd/splits/train.txt"))]
    img_test, seg_test = zip(*samples_test)
    
    train_dataset = BTXRD(img_train, seg_train, df_path, True, 
                          part_child_classes, oc_child_classes, 
                          part_cluster_file, oc_cluster_file)
    
    val_dataset = BTXRD(img_val, seg_train, df_path, False, 
                          part_child_classes, oc_child_classes, 
                          part_cluster_file, oc_cluster_file)
    
    test_dataset = BTXRD(img_test, seg_train, df_path, False, 
                          part_child_classes, oc_child_classes, 
                          part_cluster_file, oc_cluster_file)

    return train_dataset, val_dataset, test_dataset

def get_all_dataset(df_path, data_path : str, 
                    part_child_classes : int, oc_child_classes : int,  
                    part_cluster_file : str, oc_cluster_file : str) : 
    def get_file(samples) :
        
        img = os.path.join(data_path, samples)
        img = img.replace('\\', '/')
        seg = None
        return img, seg
    samples = [get_file(sample.strip())
               for sample in list(open("btxrd/splits/all_dataset.txt"))]
    img, seg = zip(*samples)
    dataset = BTXRD(img, seg,
                    df_path, 
                    False, 
                    part_child_classes, 
                    oc_child_classes, 
                    part_cluster_file, 
                    oc_cluster_file)
    return dataset

