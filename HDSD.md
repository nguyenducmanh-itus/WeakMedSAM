# HƯỚNG DẪN SỬ DỤNG & CẤU TRÚC MÃ NGUỒN (WEAKMEDSAM)

## Đề tài: Phân đoạn ảnh X-quang xương dựa trên mô hình học giám sát yếu (WeakMedSAM)

Tài liệu này hướng dẫn chi tiết về cấu trúc mã nguồn dự án và quy trình chạy thực nghiệm hệ thống **WeakMedSAM** trên môi trường **Kaggle Notebook** (hoặc qua Terminal Linux/Windows hỗ trợ GPU).

---

##  1. CẤU TRÚC MÃ NGUỒN DỰ ÁN (PROJECT STRUCTURE)

Dưới đây là cấu trúc tổ chức chi tiết của cây thư mục dự án:

```text
WEAKMEDSAM/
├── brats/                        # Thư mục xử lý/cấu hình dữ liệu BraTS
├── btxrd/                        # Thư mục xử lý bộ dữ liệu BTXRD
│   ├── btxrd_preprocess.py       # Script tiền xử lý dữ liệu BTXRD
│   └── dataset.py                # Dataset loader cho BTXRD
├── samus/                        # Kiến trúc mô hình SAM/SAMUS
│   ├── modeling/                 
│   ├── utils/                    
│   ├── __init__.py               
│   ├── automatic_mask_generator.py 
│   ├── build_sam_us.py           
│   └── SamPredictor.py           
├── unet/                         # Kiến trúc mạng phân đoạn UNet
│   ├── __init__.py               
│   ├── unet_model.py             
│   └── unet_parts.py             
├── utils/                        # Các hàm tiện ích dùng chung
│   ├── affinity.py               # Tính toán ma trận độ tương đồng, cải tiến Shannon Entropy
│   ├── metrics.py                # Tính toán các chỉ số đánh giá (Dice, IoU, HD95, ASSD)
│   ├── pytutils.py               
│   └── torchutils.py             
├── cluster.py                    # Script phân cụm đặc trưng định hình child classes
├── eval.py                       # Script đánh giá hiệu năng mô hình trên tập test
├── lab_gen.py                    # Script sinh nhãn giả
├── train_unet.py                 # Script huấn luyện mạng UNet từ nhãn giả
└── train.py                      # Script huấn luyện mô hình WeakMedSAM

```

###  2. MỤC LỤC & QUY TRÌNH THỰC THI

1. [Bước 1: Tải mã nguồn](#bước-1-tải-mã-nguồn)
2. [Bước 2: Chuẩn bị và tiền xử lý dữ liệu](#bước-2-chuẩn-bị-và-tiền-xử-lý-dữ-liệu)
3. [Bước 3: Phân cụm đặc trưng](#bước-3-phân-cụm-đặc-trưng)
4. [Bước 4: Huấn luyện mô hình WeakMedSAM](#bước-4-huấn-luyện-mô-hình-weakmedsam)
5. [Bước 5: Sinh nhãn giả (Pseudo Labels)](#bước-5-sinh-nhãn-giả-pseudo-labels)
6. [Bước 6: Huấn luyện mạng phân đoạn UNet](#bước-6-huấn-luyện-mạng-phân-đoạn-unet)
7. [Bước 7: Đánh giá mô hình](#bước-7-đánh-giá-mô-hình)

---

##  3. HƯỚNG DẪN THỰC THI CHI TIẾT

### BƯỚC 1: TẢI MÃ NGUỒN

Tải mã nguồn thực thi từ nhánh `VinhQuyenv1` của repository:

```bash
!git clone -b VinhQuyenv1 --depth 1 [https://github.com/nguyenducmanh-itus/WeakMedSAM.git](https://github.com/nguyenducmanh-itus/WeakMedSAM.git)
%cd WeakMedSAM
```
### BƯỚC 2: CHUẨN BỊ VÀ TIỀN XỬ LÝ DỮ LIỆU
1. Tải bộ dữ liệu BTXRD từ Kaggle và cấu hình đường dẫn dữ liệu đầu vào.
 - Link bộ dữ liệu: <Kaggle BTXRD Dataset> https://www.google.com/search?q=https://www.kaggle.com/datasets/bhavyasahu/btrxd-original-1/versions/1

2. Tiền xử lý dữ liệu:
```bash
!python btxrd/btxrd_preprocess.py \
   --input-path /kaggle/input/datasets/bhavyasahu/btrxd-original-1/BTXRD \
   --output-path ./btrxd_out \
   --workers 4
```

### BƯỚC 3: PHÂN CỤM ĐẶC TRƯNG
Tiến hành chạy phân cụm đặc trưng để định hình các nhóm lớp con (child classes):
```bash
!python cluster.py \
  --data_path ./btrxd_out \
  --save_path ./btrxd_out \
  --data_module btxrd \
  --batch_size 64 \
  --parent_classes 9 \
  --child_classes 4 \
  --gpus 0
```

### BƯỚC 4: HUẤN LUYỆN MÔ HÌNH WEAKMEDSAM
1. Tải trọng số pre-trained SAM ViT-B từ MetaAI:
```bash
!wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

2. Chạy lệnh huấn luyện WeakMedSAM với thông tin phân cụm đã tạo:
```bash
!python train.py \
  --seed 42 \
  --sam_ckpt ./sam_vit_b_01ec64.pth \
  --lr 1e-4 \
  --batch_size 8 \
  --max_epochs 20 \
  --val_iters 100 \
  --index btrxd_weakmedsam_entropy \
  --data_path ./btrxd_out \
  --data_module btxrd \
  --parent_classes 1 \
  --child_classes 36 \
  --child_weight 0.5 \
  --cluster_file ./btrxd_out/btrxd-4.bin \
  --logdir ./logs \
  --gpus 0
```
### BƯỚC 5: SINH NHÃN GIẢ 
Sử dụng mô hình WeakMedSAM vừa huấn luyện để sinh nhãn giả phục vụ huấn luyện mạng phân đoạn:

```bash
!python lab_gen.py \
  --batch-size 8 \
  --data-path ./btrxd_out \
  --save-path ./btrxd_pseudo_labels \
  --data-module btxrd \
  --parent-classes 1 \
  --child-classes 36 \
  --samus-ckpt ./logs/btrxd_weakmedsam_entropy/btrxd_weakmedsam_entropy_latest.pth \
  --sam-ckpt ./sam_vit_b_01ec64.pth \
  --t 4 \
  --beta 8 \
  --threshold 0.5 \
  --gpus 0
```

### BƯỚC 6: HUẤN LUYỆN MẠNG PHÂN ĐOẠN UNET
Huấn luyện mạng UNet sử dụng các nhãn giả đã được sinh ra:

```bash
!mkdir -p tblog/btrxd_unet
!python train_unet.py \
  --seed 42 \
  --lr 1e-4 \
  --batch_size 16 \
  --max_epochs 50 \
  --val_iters 100 \
  --index btrxd_unet \
  --data_path ./btrxd_out \
  --lab_path ./btrxd_pseudo_labels \
  --data_module btxrd \
  --num_classes 10 \
  --logdir ./tblog \
  --gpus 0
```
### BƯỚC 7: ĐÁNH GIÁ MÔ HÌNH
Chạy file đánh giá để kiểm tra hiệu năng của mô hình trên tập kiểm thử:

```bash
!python eval.py \
  --data_path ./btrxd_out \
  --data_module btxrd \
  --batch_size 16 \
  --num_classes 10 \
  --ckpt ./tblog/btrxd_unet/btrxd_unet.pth \
  --gpus 0
```

### Chỉ số đánh giá đầu ra:
- Dice Coefficient (dice): Độ tương đồng vùng ảnh phân đoạn.
- Jaccard Index (jaccard): Chỉ số giao nhau IoU.
- ASSD (assd): Khoảng cách biên trung bình đối xứng.
- HD95 (hd95): Khoảng cách Hausdorff tại phân vị 95%.