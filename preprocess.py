import os
import sys

# Giải quyết lỗi xung đột OpenMP (OMP: Error #15) thường gặp trên Windows 
# khi dùng chung OpenCV và PyTorch (EasyOCR)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import cv2
import numpy as np
from glob import glob
import easyocr # CẦN CÀI ĐẶT: pip install easyocr
from tqdm import tqdm
# Khởi tạo mô hình OCR một lần duy nhất ở cấp độ toàn cục (Global) 
print("Đang khởi tạo mô hình EasyOCR...")
try:
    # Chỉ định thư mục lưu mô hình nằm ngay trong thư mục đồ án để dễ quản lý
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'easyocr_models')
    os.makedirs(model_dir, exist_ok=True)
    
    # Thử khởi tạo, nếu file chưa có nó sẽ tự động tải về thư mục model_dir
    reader = easyocr.Reader(['en'], gpu=True, model_storage_directory=model_dir) 
    print("Khởi tạo EasyOCR thành công!")
    
except Exception as e:
    print(f"\n[LỖI MẠNG] EasyOCR không thể tự động tải mô hình (Timeout).")
    print(f"Chi tiết lỗi: {e}")
    print("\n" + "="*50)
    print("CÁCH KHẮC PHỤC THỦ CÔNG (OFFLINE MODE):")
    print("="*50)
    print("1. Hãy mở trình duyệt và tải 2 file sau (bằng IDM hoặc Chrome):")
    print("   - Detection model:")
    print("     https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip")
    print("   - Recognition model:")
    print("     https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/english_g2.zip")
    print(f"\n2. Giải nén 2 file .zip vừa tải.")
    print(f"3. Copy 2 file .pth bên trong dán vào thư mục sau của đồ án:\n   {model_dir}")
    print("4. Chạy lại script này, thư viện sẽ tự động nhận diện mô hình offline!")
    print("="*50 + "\n")
    sys.exit(1)

def remove_xray_text(image_path, save_path=None):
    """
    Thuật toán Tối ưu: Dùng AI (EasyOCR) để tìm chính xác chữ cái (L, R, tên bệnh viện, v.v.)
    và dùng Inpainting để 'trám' lại tự nhiên. Không bao giờ xóa nhầm xương.
    """
    # 1. Đọc ảnh
    img = cv2.imread(image_path)
    if img is None:
        print(f"Không thể đọc ảnh: {image_path}")
        return None
        
    h, w = img.shape[:2]
    
    # TẠO MỘT MASK (Mặt nạ): Nền đen, chỗ nào có chữ sẽ tô màu trắng
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # 2. Sử dụng EasyOCR để nhận diện chữ trên ảnh gốc
    # Trả về danh sách các kết quả: (bounding_box, văn_bản, độ_tin_cậy)
    results = reader.readtext(img)
    
    for (bbox, text, prob) in results:
        # Lọc bỏ các kết quả nhận diện sai/độ tin cậy quá thấp
        if prob < 0.2:
            continue
            
        # Bbox từ easyocr có dạng 4 điểm: [Top-Left, Top-Right, Bottom-Right, Bottom-Left]
        (tl, tr, br, bl) = bbox
        
        # Lấy tọa độ x_min, y_min, x_max, y_max
        x_min = int(min(tl[0], bl[0]))
        y_min = int(min(tl[1], tr[1]))
        x_max = int(max(tr[0], br[0]))
        y_max = int(max(bl[1], br[1]))
        
        # Vẽ khối màu TRẮNG lên Mask để đánh dấu vùng chữ cần "trám"
        pad = 8 # Padding để che phủ hoàn toàn vùng ánh sáng bao quanh chữ
        x1, y1 = max(0, x_min - pad), max(0, y_min - pad)
        x2, y2 = min(w, x_max + pad), min(h, y_max + pad)
        
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    # ========================================================
    # INPAINTING (Trám ảnh tự nhiên)
    # Lấy các pixel xung quanh để lấp đầy vùng chữ bị xóa
    # ========================================================
    img_inpainted = cv2.inpaint(img, mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)

    # 3. Phủ CLAHE làm rõ nét khối U xương (tiền xử lý bắt buộc cho X-quang)
    gray_inpainted = cv2.cvtColor(img_inpainted, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_clahe = clahe.apply(gray_inpainted)
    img_final = cv2.cvtColor(gray_clahe, cv2.COLOR_GRAY2BGR)
    if save_path:
        img_save = image_path.split("/")[-1]
        cv2.imwrite(os.path.join(save_path, img_save), img_final)
        
    

# ========================================================
# HƯỚNG DẪN CHẠY THỬ
# ========================================================
if __name__ == "__main__":
    # Test thử trên bức ảnh bị vệt đen ở phiên bản trước
    save_dir = "/kaggle/working/Extract_text"
    img_dir  = "/kaggle/input/datasets/nguyenmanh0404/btxrd-datasets/images"
    os.makedirs(save_dir, exist_ok=True)
    img_list = [os.path.join(img_dir, f) for f in os.listdir(img_dir)]
    pdm = tqdm(range(1, len(img_list) + 1), ncols=100)
    for n_iter in pdm : 
        cleaned_img = remove_xray_text(img_list[n_iter - 1], save_dir)
    
    print("Completed")
    

