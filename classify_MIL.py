import os
import cv2 as cv
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from attention_mil import AttentionMIL

# =====================================================================
# GIAI ĐOẠN 1: TẠO BAG (CẮT ẢNH VÀ LƯU PATCHES TENSOR)
# =====================================================================
def extract_and_save_bag_patches(image_path, label, save_dir, patch_size=224, stride=112):
    """
    Cắt ảnh bằng sliding window, bỏ patch đen, transform thành Tensor 
    [N, 3, 224, 224] và lưu thành file .pt
    """
    os.makedirs(save_dir, exist_ok=True)
    
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = cv.imread(image_path)
    if img is None:
        print(f"Không thể đọc ảnh: {image_path}")
        return
        
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    h, w, _ = img.shape

    patches_list = []
    coords_list = []

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = img[y:y+patch_size, x:x+patch_size]
            
            # Lọc Nền (Bỏ các patch đen vô ích)
            if np.mean(patch) < 15.0: 
                continue
            
            patch_tensor = preprocess(patch) # Shape: [3, 224, 224]
            patches_list.append(patch_tensor)
            coords_list.append((x, y))

    if len(patches_list) == 0:
        return 
        
    
    bag_patches = torch.stack(patches_list) # Shape: [N, 3, 224, 224]
    
    # Lưu xuống ổ cứng
    filename = os.path.basename(image_path).replace('.jpeg', '.pt')
    save_path = os.path.join(save_dir, filename)
    
    torch.save({
        'patches': bag_patches,
        'label': label,
        'coords': coords_list, 
        'image_path': image_path
    }, save_path)


# =====================================================================
# GIAI ĐOẠN 2: DATALOADER CHO TẬP DỮ LIỆU "TÚI"
# =====================================================================
class BagDataset(Dataset):
    def __init__(self, pt_dir):
        self.pt_files = [os.path.join(pt_dir, f) for f in os.listdir(pt_dir) if f.endswith('.pt')]
        
    def __len__(self):
        return len(self.pt_files)
        
    def __getitem__(self, idx):
        data = torch.load(self.pt_files[idx])
        # data['patches'] có kích thước [N, 3, 224, 224]
        return data['patches'], torch.tensor([data['label']], dtype=torch.float32), data['coords'], data['image_path']

def collate_fn(batch):
    # Trả về 1 ảnh duy nhất (với N patches) mỗi bước
    patches, label, coords, img_path = batch[0]
    return patches, label, coords, img_path


# =====================================================================
# GIAI ĐOẠN 3: HUẤN LUYỆN END-TO-END VÀ TRUY XUẤT PSEUDO-BBOX
# =====================================================================
def train_and_extract_boxes(pt_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Khởi tạo mô hình, ví dụ đóng băng 10/12 block đầu tiên của ViT
    model = AttentionMIL(num_classes=1, num_frozen_blocks=10).to(device)

    dataset = BagDataset(pt_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn) 
    
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
    
    # --- HUẤN LUYỆN ---
    model.train()
    epochs = 10
    for epoch in range(epochs):
        optimizer.zero_grad()
        for i, (patches, label, _, _) in enumerate(dataloader):
            patches = patches.to(device) 
            label = label.to(device)     
            
            logits, _ = model(patches, chunk_size=16) 
            
            loss = criterion(logits.squeeze(0), label.squeeze(0))
            loss = loss / accumulation_steps
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0:
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                
        print(f"Epoch {epoch+1} finished.")

    model.eval()
    pseudo_boxes_dict = {}
    patch_size = 224
    padding = 20
    
    with torch.no_grad():
        for patches, label, coords, img_path in dataloader:
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
            
            pseudo_boxes_dict[img_path] = (x_min, y_min, x_max, y_max)
            
    return pseudo_boxes_dict

if __name__ == "__main__":
    # Test chạy thử (Nhớ bỏ comment để chạy thật)
    path_img = "data/BTXRD/images"
    df = pd.read_excel("data/BTXRD/dataset.xlsx")
    for i in range(len(df)) :
        img = df.loc[i, "image_id"]
        label = df.loc[i, "tumor"]
        format = img.split(".")
        if format[1] == "jpg" :
            img_path = f"{format[0]}.jpeg"
        else :
            img_path = img
        
        extract_and_save_bag_patches(os.path.join(path_img, img_path), 
                                     label=label, 
                                     save_dir='./patch_tensors')
    #pseudo_boxes = train_and_extract_boxes('./patch_tensors')
    #pass
                
