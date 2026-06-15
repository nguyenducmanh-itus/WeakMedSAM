import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
#This file used for create files .txt save all path to images and split train/val/test
#Beside this file is used for analystic dataset
df = pd.read_excel("data/BTXRD/dataset.xlsx")
#Get all image id and save in folder splits in module btxrd dataset
def Save_all_images() :
    with open("all_dataset.txt", "x", encoding="utf-8") as f :
        for i in range(len(df)) :
            format = os.path.splitext(df.loc[i, "image_id"])[1]
            if format == ".jpg" :
                f.writelines(df.loc[i, "image_id"][:-4] + ".jpeg"+"\n")
            else :    
                f.writelines(df.loc[i, "image_id"]+"\n")

#Analystics the number of images in each body part cluster
def Analystic_bp_cluster(Cluster_file) :
    with open(Cluster_file, 'rb') as f :
        file = pickle.load(f)
    anal_dict = {0 : 0, 
                 1 : 0, 
                 2 : 0, 
                 3 : 0}
    for keys, values in file.items() : 
        anal_dict[int(values.item())] += 1
    
    return anal_dict    
    
#Get labels cluster of all data
def read_labels_cluster(cluster_file) : 
    with open(cluster_file, "rb") as f :
        oc_file = pickle.load(f) 
    return oc_file


#Group data with same cluster in one dict list
def anal_len_each_cluster(objects) :
    copy_object = objects.copy()
    #Get images have tumor
    oc_anal = {}
    non_tumor = {"none" : []}
    for i in range(1,5) :
        oc_anal[i] = {1 : [], 2 : [], 3 : []}

    for keys, values in objects.items() :
        for i in range(1, 5) :
            if values[i-1] != 0 :
                oc_anal[i][int(values[i - 1])].append(keys)
                copy_object.pop(keys)
    for keys, values in copy_object.items() :
        non_tumor["none"].append(keys)  
    return oc_anal, non_tumor            

#This function print len of each child_clusters
def print_len_cluster(objects) :
    for keys, values in objects.items() :
        for i in range(1, 4) :
            print(len(values[i]))
            
#This function split dataset to train/val/test with ratio 8/1/1
#Arguments input is dict grouped data with same cluster into a list
def split_data_train(objects, non_objects, ratio = 0.8) : 
    train = []
    val = []
    test = []
    for keys, values in objects.items() :
        for i in range(1, 4) : 
          if len(values[i]) != 0 :    
            y = [1] * len(values[i])
            x_train, x_test, y_train, y_test = train_test_split(
                values[i], y, test_size= 1 - ratio, random_state=42
            )
            x_val, x_test, y_val, y_test = train_test_split(
                x_test, y_test, test_size=0.5, random_state=42
            )
            train += x_train
            val += x_val
            test += x_test
    for keys, values in non_objects.items() :
        y = [0] * len(values)
        x_train, x_test, y_train, y_test = train_test_split(
            values, y, test_size= 1 - ratio, random_state=42
        )
        x_val, x_test, y_val, y_test = train_test_split(
            x_test, y_test, test_size=0.5, random_state=42
        )
        train += x_train
        val += x_val
        test += x_test
            
    return train, val, test   

def save_split_data(train, val, test) :
    with open("btxrd/splits/train.txt", "w") as f_t :
        for train_data in train :
            path = (train_data + ".jpeg")
            #f_t.write(path.replace("\\", "/"))
            f_t.write(path)
            f_t.write("\n")
        
    with open("btxrd/splits/val.txt", "w") as f_val :
        for val_data in val :
            path = (val_data + ".jpeg")
            f_val.write(path)
            #f_val.write(path.replace("\\", "/"))
            f_val.write("\n")
    with open("btxrd/splits/test.txt", "w") as f_test :
        for test_data in test :
            path = (test_data + ".jpeg")
            f_test.write(path)
            #f_test.write(path.replace("\\", "/"))
            f_test.write("\n")


with open("btxrd/splits/test.txt") as f : 
    train_file = f.readlines()
    
print(len(train_file))