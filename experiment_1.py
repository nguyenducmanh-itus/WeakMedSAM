import cv2
import os
import torch
import numpy as np
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import argparse
import pickle 
def get_gradcam_heatmap(model, input_tensor, target_layer):
    """
    Bước 1: Trích xuất Heatmap từ mạng CNN phân loại thô
    """
    # Khởi tạo GradCAM
    cam = GradCAM(model=model, target_layers=[target_layer])
    
    # target=None nghĩa là lấy class có xác suất cao nhất (Dự đoán là Có U)
    targets = [ClassifierOutputTarget(1)] # Giả sử class 1 là "Có u"
    
    # Sinh heatmap (trả về mảng 2D giá trị từ 0 đến 1)
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    
    return grayscale_cam[0, :]

def extract_roi_and_crop(orig_image_path, heatmap, resize_dim=(256, 256), threshold=0.5, padding=30):
    """
    Bước 2: Binarize heatmap, tìm Bounding Box và cắt ảnh gốc
    """
    # 1. Đọc ảnh gốc (độ phân giải cao)
    orig_image = cv2.imread(orig_image_path)
    orig_h, orig_w = orig_image.shape[:2]
    
    # 2. Tiền xử lý heatmap thành ảnh nhị phân (Mask)
    # Những vùng có độ "nóng" > threshold sẽ được giữ lại
    binary_mask = np.uint8(heatmap > threshold) * 255
    
    # 3. Tìm các contours (đường viền) trên mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("Không tìm thấy vùng khả nghi nào thỏa mãn ngưỡng threshold.")
        return None
        
    # Lấy contour lớn nhất (đề phòng nhiễu nhỏ lẻ)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Tìm Bounding Box trên hệ tọa độ của ảnh đã resize (256x256)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # 4. Ánh xạ tọa độ Bounding Box về kích thước ảnh gốc
    scale_x = orig_w / resize_dim[0]
    scale_y = orig_h / resize_dim[1]
    
    x_orig = int(x * scale_x)
    y_orig = int(y * scale_y)
    w_orig = int(w * scale_x)
    h_orig = int(h * scale_y)
    
    # 5. Thêm Padding (mở rộng vùng cắt)
    # Rất quan trọng: Padding giúp WeakMedSAM có thêm ngữ cảnh xung quanh khối u
    x_pad = max(0, x_orig - padding)
    y_pad = max(0, y_orig - padding)
    x_end = min(orig_w, x_orig + w_orig + padding)
    y_end = min(orig_h, y_orig + h_orig + padding)
    
    # 6. Thực hiện cắt (Crop)
    cropped_roi = orig_image[y_pad:y_end, x_pad:x_end]
    
    return cropped_roi, (x_pad, y_pad, x_end, y_end)

# ================= CÁCH SỬ DỤNG =================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type =str)
    parser.add_argument("--dir_path", type = str)
    parser.add_argument("--model_ckpt", type = str)
    parser.add_argument("--save_bbox", type = str)
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.save_bbox, exist_ok=True)
    # 1. Setup mô hình ResNet18 (Thay bằng model bạn đã train để phân loại u/không u)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 2) # 2 class: Bình thường / Có u
    bbx_map = {}
    # Tải weights bạn đã train vào đây
    model.load_state_dict(torch.load(args.model_ckpt))
    model = model.to(device)
    model.eval()
    
    # Target layer cho ResNet18 thường là layer conv cuối cùng
    target_layer = model.layer4[-1]
    
    # 2. Chuẩn bị ảnh đầu vào
   
    list_images = [os.path.join(args.dir_path, f) for f in os.listdir(args.dir_path)]
    
    
    # Transform chuẩn của torchvision
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    for image_path in list_images : 
        # Đọc và biến đổi ảnh thành tensor
        img_cv = cv2.imread(image_path)
        img_cv_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        input_tensor = transform(img_cv_rgb).unsqueeze(0).to(device)
        
        # 3. Thực thi Pipeline
        # Lấy Heatmap
        heatmap = get_gradcam_heatmap(model, input_tensor, target_layer)
        
        # Cắt ROI từ ảnh gốc
        result = extract_roi_and_crop(
            orig_image_path=image_path,
            heatmap=heatmap,
            resize_dim=(256, 256),
            threshold=0.6, 
            padding=50     # Mở rộng 50 pixel mỗi viền
        )
        
        if result is not None:
            cropped_image, bbox = result
            print(f"Cắt thành công ROI tại tọa độ gốc: {bbox}")
            # Lưu lại để đưa vào Bước 3 (WeakMedSAM)
            image_path_split = image_path.split("/")
            img_id = image_path_split[-1].split(".")
            image_name = f"{img_id[0]}_crop.{img_id[1]}"
            bbx_map[img_id[0]] = bbox
            cv2.imwrite(os.path.join(args.save_dir, image_name), cropped_image)
    filehanlder = open(args.save_bbox, 'wb')
    pickle.dump(bbx_map, filehanlder)
    

