#This file used for cluster laten co-occurance in dataset 
#After cluster following to similiar body-parts
#Each cluster in dataset will be clustered one more to extract laten occurance
import torch
from torchvision.models import resnet50
from sklearn.cluster import KMeans
import pickle
import argparse


with open("btxrd/variable_cluster/non_pretrained_medical.pkl", "rb") as f:
    body_parts = pickle.load(f)
    

# if __name__ == "__main__" :
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--data_path", type=str)

