import os
import torch

print("====================================")
print("1. Đang tải mã nguồn từ GitHub...")
os.system("git clone https://github.com/nguyenducmanh-itus/WeakMedSAM.git /kaggle/working/my_project")

print("2. Di chuyển vào thư mục dự án...")
# Bắt buộc phải dùng os.chdir để đổi thư mục làm việc thực sự
os.chdir("/kaggle/working/my_project")

print("3. Cài đặt các thư viện cần thiết...")
os.system("pip install nibabel monai segment-anything")

print("4. Bắt đầu quá trình huấn luyện mô hình...")
# Đặt toàn bộ chuỗi lệnh của bạn vào một biến string
cmd = (
    "python cluster.py "
    "--batch_size 256 "
    "--data_path '/kaggle/input/datasets/nguyenmanh0404/thesis' "
    "--data_module brats "
    "--parent_classes 1 "
    "--child_classes 8 "
    "--save_path '/kaggle/working/Cluster_dataset/' "
    "--gpus 0"
)
# Thực thi chuỗi lệnh
os.system(cmd)

# import os
# file_path = '/kaggle/input/datasets/nguyenmanh0404/data-cluster-thesis/brats-8.bin'

# print(f"Kích thước file: {os.path.getsize(file_path)} bytes")
# with open(file_path, "rb") as f:
#     print(f"20 bytes đầu tiên: {f.read(20)}")