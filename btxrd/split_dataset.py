#This file used for split to train/valid/test dataset
#Strategy split for classification and U-net stage is difference.
import os
import pandas as pd
from collections import defaultdict
import pickle
import random
from PIL import Image 
from pathlib import Path

random.seed(42)
df_path = "./data/BTXRD/dataset.xlsx"
cluster_path = r"C:/Users/ADMIN/OneDrive - VNU-HCMUS/CNTT-HK8/Thesis/WeakMedSAM/output/btxrd-8.bin"
#Get all dataset and save in all.txt
def get_list_all_images(data_path = "./data/BTXRD/dataset.xlsx") :     
    df = pd.read_excel(data_path)
    images_path = df["image_id"]
    with open("./btxrd/splits/all.txt", "w") as text_file : 
        for i in images_path : 
            text_file.writelines(f"./data/BTXRD/images/{i}" +"/n")        
            

#Get list images is tumor-free
def get_listimg_nontumor(df_path) :
    df = pd.read_excel(df_path)
    images_list = df[df["tumor"] == 0]["image_id"].tolist()
    return images_list

#Get all images in same child class and parent class
def group_same_PC(cluster_file) :
    with open(cluster_file, "rb") as f :
        cluster = pickle.load(f)
    grouped = defaultdict(list)
    for img_name, label in cluster.items() :
        for i, parent in enumerate(label) :
            if parent != 0 :
                grouped[i].append(img_name)
    return grouped 

#Get train/val/test cases without tumors
def split_non_tumors(images_list,
                     train_ratio = 0.8, 
                     val_ratio = 0.1, 
                     test_ratio = 0.1) :
    n = len(images_list)
    train_split = []
    val_split = []
    test_split = []
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)
    images_copy = images_list.copy()
    random.shuffle(images_copy)
    train_split.extend(images_copy[: n_train])
    val_split.extend(images_copy[n_train : n_train + n_val])
    test_split.extend(images_copy[: n_test])
    return train_split, val_split, test_split
#-----Split dataset for classification stage-------#
#Ratio 8-1-1
def split_dataset_classifier(cluster_file, 
                             df_path, 
                             train_ratio = 0.8, 
                             val_ratio = 0.1, 
                             test_ratio = 0.1) :
    train_split = []
    val_split = []
    test_split = []
    grouped = group_same_PC(cluster_file) 
    for parent, list_images in grouped.items() :
        images =  list_images.copy()
        random.shuffle(images)
        n = len(images)
        n_train = int(train_ratio * n)
        n_val = int(val_ratio * n)
        n_test = int(test_ratio * n)
        #add a list to list
        train_split.extend(images[ : n_train])
        val_split.extend(images[n_train : n_train + n_val])
        test_split.extend(images[n_train + n_val : ])    
    return train_split, val_split, test_split
#------Save datasets for classifier---------#
def dictionary_list_split(cluster_path, df_path) :
    train_split, val_split, test_split = split_dataset_classifier(cluster_path, df_path)
    list_non_images = get_listimg_nontumor(df_path)
    non_train_split, non_val_split, non_test_split = split_non_tumors(list_non_images)
    train_split.extend(non_train_split)
    val_split.extend(non_val_split)
    test_split.extend(non_test_split)
    return {"./splits/classifier_split/train.txt" : train_split, 
            "./splits/classifier_split/val.txt" : val_split, 
            "./splits/classifier_split/test.txt" : test_split}

    
def save_path(save_list, data_path = "./data/BTXRD/images") :
    for name, list_imgs in save_list.items() :
        with open(name, "w") as f :
            for i in list_imgs :
                path = os.path.join(data_path, i)
                path = path.replace("\\", "/")
                f.writelines(path + "\n")
                        
#-----Split dataset for U-net stage-------#
save_list  = dictionary_list_split(cluster_path, df_path)
save_path(save_list)

