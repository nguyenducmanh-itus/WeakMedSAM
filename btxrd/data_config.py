import os
import pandas as pd
import pickle
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
    
        
analys_part_dict = {0 : [], 
                    1 : [], 
                    2 : [], 
                    3 : []}    
with open("Body_part_classes/btxrd-4.bin", 'rb') as f :
        file = pickle.load(f)
        for keys, values in file.items() :
            analys_part_dict[int(values.item())].append(keys)

for i in range(4) :
    print(f"Length of cluster {i} : {len(analys_part_dict[i])}")
