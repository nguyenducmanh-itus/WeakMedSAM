import os
import torch

print("====================================")
print("1. Đang tải mã nguồn từ GitHub...")
os.system("git clone https://github.com/nguyenducmanh-itus/WeakMedSAM.git /kaggle/working/my_project")

print("2. Di chuyển vào thư mục dự án...")
# Bắt buộc phải dùng os.chdir để đổi thư mục làm việc thực sự
os.chdir("/kaggle/working/my_project")

print("3. Cài đặt các thư viện cần thiết...")
os.system("pip install nibabel monai segment-anything scikit-learn pandas torchvision")

print("4. Tải SAM checkpoint...")
os.system("mkdir -p /kaggle/working/my_project/ckpt")
os.system("wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -P /kaggle/working/my_project/ckpt/")


print("6. Bắt đầu huấn luyện mô hình...")
cmd_train = (
    "python train.py "
    "--seed 42 "
    "--sam_ckpt '/kaggle/working/my_project/ckpt/sam_vit_b_01ec64.pth' "
    "--lr 1e-4 "
    "--batch_size 8 "  # Giảm batch_size cho GPU Kaggle
    "--max_epochs 10 "
    "--val_iters 3000 "
    "--index fracatlas_kaggle "
    "--data_path '/kaggle/input/fracatlas' "
    "--data_module frac_atlas "
    "--parent_classes 1 "
    "--child_classes 10 "
    "--child_weight 0.5 "
    "--cluster_file '/kaggle/input/fracatlas-dataset' "
    "--logdir '/kaggle/working/logs' "
    "--gpus 0"
)
os.system(cmd_train)

