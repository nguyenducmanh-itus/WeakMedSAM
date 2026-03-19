import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
import torchvision.transforms as T
import random

class FracAtlasDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform or T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.df = pd.read_csv(os.path.join(data_path, 'dataset.csv'))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        subfolder = 'Fractured' if row['fractured'] == 1 else 'Non_fractured'
        img_path = os.path.join(self.data_path, 'images', subfolder, row['image_id'])
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        plab = torch.tensor([row['fractured']], dtype=torch.float)
        return {"img": img, "plab": plab, "idx": idx}
                           
def get_all_dataset(data_path, split, transform):
    return FracAtlasDataset(data_path, transform)

