from tqdm import tqdm
import time
from utils.pytuils import AverageMeter
import pickle
from btxrd.dataset import BTRXD_Dataset 
list_images = [image.strip() for image in open("./btxrd/splits/group_non/train.txt", "r")]

my_dataset = BTRXD_Dataset(list_images, 
                           "./output/cluster_non_tumor/btxrd-8.bin", 
                           "osteochondroma", 
                           "other mt", 
                           8
                           )
