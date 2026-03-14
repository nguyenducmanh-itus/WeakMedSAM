import os

print("1. Đang tải mã nguồn từ GitHub...")
os.system("git clone https://github.com/nguyenducmanh-itus/WeakMedSAM.git /kaggle/working/my_project")

print("2. Di chuyển vào thư mục dự án...")
# Bắt buộc phải dùng os.chdir để đổi thư mục làm việc thực sự
os.chdir("/kaggle/working/my_project")

print("3. Cài đặt các thư viện cần thiết...")
os.system("pip install nibabel monai segment-anything")

print("4. Bắt đầu quá trình huấn luyện mô hình...")
# Đặt toàn bộ chuỗi lệnh của bạn vào một biến string
cmd = """
python train.py \
  --seed 42 \
  --sam_ckpt /kaggle/input/vit-ckpt-sam/sam_vit_b_01ec64.pth \
  --lr 1e-4 \
  --batch_size 2 \
  --max_epochs 1 \
  --val_iters 3000 \
  --index 0 \
  --data_path /kaggle/input/thesis/Brats_Preprocess \
  --data_module brats \
  --parent_classes 1 \
  --child_classes 8 \
  --child_weight 0.5 \
  --cluster_file /kaggle/input/data-cluster-thesis/brats-8.bin \
  --logdir /kaggle/working/logs \
  --gpus 0
"""

# Thực thi chuỗi lệnh
os.system(cmd)