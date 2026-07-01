import os
import cv2 as cv
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from attention_mil import AttentionMIL
import argparse
from tqdm import tqdm

def extract_and_save_bag_patches(image_path, label, save_dir, patch_size=224, stride=112):
    """
    Cut image to patchs by sliding widown with components : 
        - step = 112  
        - size of batch is 224
    """
    os.makedirs(save_dir, exist_ok=True)
    
    

    img = cv.imread(image_path)
    if img is None:
        print(f"Can't read image : {image_path}")
        return
        
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    h, w, _ = img.shape

    
    coords_list = []

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = img[y:y+patch_size, x:x+patch_size]
            
            # Lọc Nền (Bỏ các patch đen vô ích)
            if np.mean(patch) < 15.0: 
                continue
            coords_list.append((x, y))
    if len(coords_list) == 0 : 
        return
    
    file_name = os.path.basename(image_path).replace(".jpeg", ".pt")
    save_path = os.path.join(save_dir, file_name)
    
    torch.save({
        'label' : label, 
        'coords' : coords_list, 
        'image_path' : image_path
    }, save_path)
            


def print_memory(name):
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    print(f"{name}")
    print(f"Allocated : {allocated:.2f} MB")
    print(f"Reserved  : {reserved:.2f} MB")
    print()

#Data Module for Bag dataset
class BagDataset(Dataset):
    def __init__(self,dir_img, pt_file, patch_size=224):
        self.pt_files = pt_file
        self.patch_size = patch_size
        self.dir_img = dir_img
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    def __len__(self):
        return len(self.pt_files)
        
    def __getitem__(self, idx):
        data = torch.load(
                self.pt_files[idx],
                weights_only=False
            )
        img_path = data['image_path']
        coords = data['coords']
        label = data['label']
        
        
        img = cv.imread(os.path.join(self.dir_img, img_path))
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        
        patches = []
        for x, y in coords:
            patch = img[y:y+self.patch_size, x:x+self.patch_size]
            patch_tensor = self.preprocess(patch)
            patches.append(patch_tensor)
            
        bag_tensor = torch.stack(patches)
        
        return bag_tensor, torch.tensor([label], dtype=torch.float32), coords, img_path
            
def collate_fn(batch):
    
    patches, label, coords, img_path = batch[0]
    return patches, label, coords, img_path

def train_and_extract_boxes(dir_img, current_epoch , pt_dir, 
                            save_dir, checkpoint_dir, check_point):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #print("Memory when load model")
    model = AttentionMIL(num_classes=1, num_frozen_blocks=10).to(device)
    if check_point != "" :
        model.load_state_dict(check_point)
    all_pt_files = [os.path.join(pt_dir, f) for f in os.listdir(pt_dir)]
    random.seed(42)
    random.shuffle(all_pt_files)
    train_split = int(0.8 * len(all_pt_files))
    val_split = int(0.1 * len(all_pt_files))
    
    train_files = all_pt_files[:train_split]
    val_files = all_pt_files[train_split:train_split+val_split]
    test_files = all_pt_files[train_split+val_split:]
    
    train_dataset = BagDataset(dir_img, train_files)
    val_dataset = BagDataset(dir_img, val_files)
    test_dataset = BagDataset(dir_img, test_files)
    
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, 
                            collate_fn=collate_fn, num_workers=4) 
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True, 
                            collate_fn=collate_fn, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, 
                            collate_fn=collate_fn, num_workers=4)
    
    #print_memory("Memory after dataloader")
    vit_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if "vit" in name:
            vit_params.append(param)
        else:
            head_params.append(param)
            
    optimizer = torch.optim.AdamW([
        {'params': vit_params, 'lr': 1e-5}, 
        {'params': head_params, 'lr': 1e-4} 
    ], weight_decay=1e-4)
    
    criterion = nn.BCEWithLogitsLoss()
    accumulation_steps = 16 
    MAX_PATCHES = 64
    
    model.train()
    epochs = 10
    max_iters = epochs * len(train_dataloader)
    current_iters = current_epoch * len(train_dataloader)
    train_loader_iter = iter(train_dataloader)
    pbar = tqdm(range(current_iters, max_iters + 1), ncols=100)
    runing_loss = 0.0
    optimizer.zero_grad()
    for n_iter in pbar :
        try : 
            patches, label, _, _ = next(train_loader_iter)
        except :
            train_loader_iter = iter(train_dataloader)
            patches, label, _, _ = next(train_loader_iter)
        
        if patches.size(0) > MAX_PATCHES : 
            indices = torch.randperm(patches.size(0))[:MAX_PATCHES]
            patches = patches[indices]
        
        patches = patches.to(device) 
        label = label.to(device)     
        logits, _ = model(patches, chunk_size=16)
        loss = criterion(logits.squeeze(0), label)
        runing_loss += loss.item()
        loss = loss / accumulation_steps
        loss.backward()
        if n_iter % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            
        if (n_iter) % 100 == 0 :
            avg_loss = runing_loss / 100 
            print(f"Epoch {n_iter // len(train_dataloader) + 1}| iter {n_iter} | Loss {avg_loss}")
            runing_loss = 0.0
            
        if n_iter % (2 * len(train_dataloader)) == 0 :
            checkpoint_path = os.path.join(checkpoint_dir, 
                                           f'mil_vit_{n_iter/len(train_dataloader)}.pth')
            torch.save(model.state_dict(), checkpoint_path)
        if n_iter % len(train_dataloader) == 0 :
            model.eval()
            val_loss = 0.0
            correct = 0.0
            total = 0
            with torch.no_grad() :
                for patches, label, _, _ in val_dataloader :
                    patches = patches.to(device)
                    label = label.to(device)
                    logits, _ = model(patches, chunk_size=32) 
                
                    loss = criterion(logits.squeeze(0), label)
                    val_loss += loss.item()
                    
                    
                    pred = (torch.sigmoid(logits.squeeze(0)) > 0.5).float()
                    if pred.item() == label.item():
                        correct += 1
                    total += 1
                
                avg_val_loss = val_loss / len(val_dataloader)
                val_acc = (correct / total) * 100
                print(f"Valid Loss : {avg_val_loss:.4f} | Valid accuracy : {val_acc:.2f}%" )
            model.train()
            runing_loss = 0.0
    
    print("Create Bounding box")
    model.eval()
    patch_size = 224
    padding = 20
    def extract_bbox(loader) :
        with torch.no_grad():
            for patches, label, coords, image_path in loader:
                if label.item() == 0: 
                    continue
                    
                patches = patches.to(device)
                _, A = model(patches, chunk_size=32) 
                
                best_patch_idx = torch.argmax(A, dim=1).item()
                best_x, best_y = coords[best_patch_idx]
                
                x_min = max(0, best_x - padding)
                y_min = max(0, best_y - padding)
                x_max = best_x + patch_size + padding
                y_max = best_y + patch_size + padding
                img = cv.imread(os.path.join(dir_img, image_path))
                if img is not None : 
                    crop_img = img[y_min : y_max, x_min : x_max]
                    file_name, ext = os.path.splitext(image_path)
                    new_file_name = f"{file_name}_crop{ext}"
                    cv.imwrite(os.path.join(save_dir, new_file_name), crop_img)
    extract_bbox(train_dataloader)
    extract_bbox(val_dataloader)
    extract_bbox(test_dataloader)
          
            
     
    

if __name__ == "__main__":
    # Test chạy thử (Nhớ bỏ comment để chạy thật)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--data_frame", type=str)
    parser.add_argument("--pt_dir", type=str)
    parser.add_argument("--checkpoint_dir", type=str)
    args = parser.parse_args()
    path_img = args.data_path
    
    pseudo_boxes = train_and_extract_boxes(args.data_path, args.pt_dir, args.save_dir, \
        args.checkpoint_dir)
    #pass
                
