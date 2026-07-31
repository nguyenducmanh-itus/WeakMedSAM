import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import copy
from PIL import Image
from sklearn.model_selection import train_test_split
import argparse
# ==========================================
# 1. Định nghĩa Custom Dataset
# ==========================================
class BoneTumorDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        """
        dataframe: DataFrame chứa tên file ảnh và nhãn
        img_dir: Đường dẫn đến thư mục chứa toàn bộ ảnh
        transform: Các phép biến đổi ảnh (augmentation)
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # LƯU Ý: Thay 'filename' và 'label' bằng tên cột thực tế trong file Excel của bạn
        img_name = str(self.dataframe.loc[idx, 'image_id'])
        splt = img_name.split(".")
        if splt[1] == "jpg" :
            img_name = f"{splt[0]}.jpeg"
        label = int(self.dataframe.loc[idx, 'tumor']) # Giả sử nhãn là 0 (Không u) và 1 (Có u)
        
        # Đọc ảnh
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# ==========================================
# 2. Hàm Huấn luyện chính
# ==========================================
def train_model_from_excel(excel_path, save_dir , img_dir, num_epochs=15, batch_size=16, learning_rate=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang huấn luyện trên thiết bị: {device}")

    # Đọc dữ liệu từ file Excel
    print("Đang đọc file Excel và chia tập dữ liệu...")
    df = pd.read_excel(excel_path)
    
    # Chia tập Train (80%) và Val (20%)
    # stratify=df['label'] giúp đảm bảo tỷ lệ Có u / Không u đồng đều ở cả 2 tập
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42,stratify=df['tumor'])
    
    print(f"Số lượng ảnh Train: {len(train_df)} | Val: {len(val_df)}")

    # Cấu hình Augmentation
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Khởi tạo Custom Datasets
    image_datasets = {
        'train': BoneTumorDataset(train_df, img_dir, transform=data_transforms['train']),
        'val': BoneTumorDataset(val_df, img_dir, transform=data_transforms['val'])
    }
    
    # Khởi tạo DataLoaders (batch_size=16 phù hợp cho 4GB VRAM)
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x == 'train'), num_workers=2)
        for x in ['train', 'val']
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    # Setup ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2) # 2 class (0 và 1)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Vòng lặp huấn luyện
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    print(f'\nHuấn luyện hoàn tất. Best Val Acc: {best_acc:4f}')
    model.load_state_dict(best_model_wts)
    save_path = os.path.join(save_dir, "resnet18_classifier.pth")
    
    # 2. Kiểm tra và tạo thư mục nếu chưa tồn tại
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        print(f"Đã tạo thư mục: {save_dir}")

    # 3. Lưu mô hình
    torch.save(model.state_dict(), save_path)
    print(f"Mô hình đã được lưu tại: {save_path}")

    return model

# ==========================================
# CÁCH CHẠY
# ==========================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--excel_path", type=str)
    parser.add_argument("--img_dir", type=str)
    parser.add_argument("--save_dir", type=str)
    args = parser.parse_args()
    # THAY ĐỔI ĐƯỜNG DẪN TẠI ĐÂY
    # EXCEL_PATH = "data/BTXRD/dataset.xlsx" 
    # IMG_DIR = "data/BTXRD/images"   
    trained_model = train_model_from_excel(
        excel_path=args.excel_path,
        save_dir=args.save_dir,  
        img_dir=args.img_dir, 
        num_epochs=15,
        batch_size=16,
        learning_rate=1e-4
    )