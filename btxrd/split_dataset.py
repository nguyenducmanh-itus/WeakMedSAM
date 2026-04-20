#This file used for split to train/valid/test dataset
#Strategy split for classification and U-net stage is difference.
import os
import pandas as pd

#Get all dataset and save in all.txt 
data_path = "./data/BTXRD/dataset.xlsx"
df = pd.read_excel(data_path)
images_path = df["image_id"]
with open("./btxrd/split/all.txt", "w") as text_file : 
    for i in images_path : 
        text_file.writelines(f".data/BTXRD/{i}" +"\n")        

#-----Split dataset for classification stage-------#


#-----Split dataset for U-net stage-------#