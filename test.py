from tqdm import tqdm
import time
from utils.pytuils import AverageMeter
import pickle

with open("./output/btxrd-8.bin", "rb") as f :
    cluster = pickle.load(f)
    
print(cluster)