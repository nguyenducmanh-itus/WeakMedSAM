import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
import torchvision.transforms as T
import random
from sklearn.model_selection import train_test_split

class FracAtlasDataset(Dataset):
    def __init__(self, data_path, df, transform=None, child_classes=0, cluster_file=None):
        self.data_path = data_path
        self.df = df
        self.transform = transform or T.Compose([
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.child_classes = child_classes
        if child_classes != 0 and cluster_file:
            import pickle
            with open(cluster_file, "rb") as f:
                self.clabs = pickle.load(f)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        subfolder = 'Fractured' if row['fractured'] == 1 else 'Non_fractured'
        img_path = os.path.join(self.data_path, 'images', subfolder, row['image_id'])
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        plab = torch.tensor([row['fractured']], dtype=torch.float)
        idx_str = row['image_id']
        
        if self.child_classes != 0:
            clab = torch.zeros(self.child_classes + 1).float()
            if plab[0] != 0:
                clab[int(self.clabs[idx_str][0]) + 1] = 1
            else:
                clab[0] = 1
            return {"img": img, "plab": plab, "clab": clab, "idx": idx_str}
        else:
            return {"img": img, "plab": plab, "idx": idx_str}
                           
def get_dataset(data_path: str, child_classes: int, cluster_file: str):
    df = pd.read_csv(os.path.join(data_path, 'dataset.csv'))
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['fractured'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['fractured'])
    
    train_dataset = FracAtlasDataset(data_path, train_df, child_classes=child_classes, cluster_file=cluster_file)
    val_dataset = FracAtlasDataset(data_path, val_df, child_classes=0, cluster_file=cluster_file)
    test_dataset = FracAtlasDataset(data_path, test_df, child_classes=0, cluster_file=cluster_file)
    
    return train_dataset, val_dataset, test_dataset

def get_all_dataset(data_path, split, transform):
    df = pd.read_csv(os.path.join(data_path, 'dataset.csv'))
    if transform == "":
        transform = None
    return FracAtlasDataset(data_path, df, transform)



