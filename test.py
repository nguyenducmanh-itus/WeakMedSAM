import torch
import pickle

with open("C:/Users/ADMIN/Downloads/mil_vit_8.0/mil_vit_8.0/data.pkl", "rb") as f :
    files = pickle.load(f)
    
print(files)