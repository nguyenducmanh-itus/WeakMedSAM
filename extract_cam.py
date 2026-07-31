import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# ============================================================
# 1. Khởi tạo ResNet18 giống hệt lúc huấn luyện
# ============================================================
def build_resnet18(num_classes=2):
    # Không cần tải lại trọng số ImageNet vì sẽ load trọng số đã train
    model = models.resnet18(weights=None)

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model


# ============================================================
# 2. Load trọng số mô hình
# ============================================================
def load_model(weights_path, device):
    model = build_resnet18(num_classes=2)

    try:
        checkpoint = torch.load(
            weights_path,
            map_location=device,
            weights_only=True
        )
    except TypeError:
        # Tương thích với các phiên bản PyTorch cũ
        checkpoint = torch.load(
            weights_path,
            map_location=device
        )

    # Hỗ trợ cả trường hợp checkpoint chứa model_state_dict
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

    # Xử lý trường hợp mô hình từng được train bằng DataParallel
    cleaned_state_dict = {}

    for key, value in checkpoint.items():
        new_key = key.replace("module.", "")
        cleaned_state_dict[new_key] = value

    model.load_state_dict(cleaned_state_dict, strict=True)

    model = model.to(device)
    model.eval()

    return model


# ============================================================
# 3. Class trích xuất CAM
# ============================================================
class CAMExtractor:
    def __init__(self, model):
        self.model = model
        self.feature_maps = None

        # Lưu feature map đầu ra của layer4
        self.hook_handle = self.model.layer4.register_forward_hook(
            self._save_feature_maps
        )

    def _save_feature_maps(self, module, inputs, output):
        """
        Với ảnh đầu vào 256x256, output thường có kích thước:
        [batch_size, 512, 8, 8]
        """
        self.feature_maps = output.detach()

    def generate_cam(self, input_tensor, target_class=1):
        """
        input_tensor:
            Tensor có shape [1, 3, H, W]

        target_class:
            0 = không có tổn thương
            1 = có tổn thương
        """

        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)

        if self.feature_maps is None:
            raise RuntimeError(
                "Không lấy được feature map từ layer4."
            )

        # Feature map của ảnh đầu tiên trong batch
        # Shape: [512, H_feature, W_feature]
        feature_map = self.feature_maps[0]

        # Trọng số FC của lớp cần trích CAM
        # Shape: [512]
        class_weights = self.model.fc.weight[target_class].detach()

        # CAM(x,y) = tổng theo channel:
        # weight_k * feature_map_k(x,y)
        cam = torch.einsum(
            "c,chw->hw",
            class_weights,
            feature_map
        )

        # Chỉ giữ những vùng đóng góp dương cho lớp đích
        cam = torch.relu(cam)

        # Chuẩn hóa CAM về [0, 1]
        cam = cam - cam.min()

        max_value = cam.max()

        if max_value > 0:
            cam = cam / max_value

        cam = cam.cpu().numpy()

        predicted_class = int(torch.argmax(probabilities, dim=1).item())
        probabilities = probabilities[0].cpu().numpy()

        return cam, probabilities, predicted_class

    def remove_hook(self):
        self.hook_handle.remove()


# ============================================================
# 4. Tiền xử lý ảnh giống lúc validation
# ============================================================
def get_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


# ============================================================
# 5. Lưu CAM, heatmap và ảnh chồng CAM
# ============================================================
def save_cam_results(
    original_image,
    cam,
    save_dir,
    alpha=0.45,
    threshold=None
):
    os.makedirs(save_dir, exist_ok=True)

    original_array = np.asarray(
        original_image,
        dtype=np.float32
    )

    # Chuyển CAM từ kích thước feature map sang kích thước ảnh gốc
    cam_uint8 = (cam * 255).astype(np.uint8)

    cam_image = Image.fromarray(cam_uint8).resize(
        original_image.size,
        Image.Resampling.BILINEAR
    )

    cam_resized = np.asarray(
        cam_image,
        dtype=np.float32
    ) / 255.0

    # Lưu CAM dạng ảnh xám
    cam_gray_path = os.path.join(
        save_dir,
        "cam_gray.png"
    )
    cam_image.save(cam_gray_path)

    # Chuyển CAM thành heatmap
    colormap = plt.get_cmap("jet")
    heatmap = colormap(cam_resized)[..., :3]
    heatmap = (heatmap * 255).astype(np.uint8)

    heatmap_path = os.path.join(
        save_dir,
        "cam_heatmap.png"
    )
    Image.fromarray(heatmap).save(heatmap_path)

    # Chồng heatmap lên ảnh gốc
    overlay = (
        (1.0 - alpha) * original_array
        + alpha * heatmap.astype(np.float32)
    )

    overlay = np.clip(
        overlay,
        0,
        255
    ).astype(np.uint8)

    overlay_path = os.path.join(
        save_dir,
        "cam_overlay.png"
    )
    Image.fromarray(overlay).save(overlay_path)

    # Lưu ảnh gốc
    original_path = os.path.join(
        save_dir,
        "original.png"
    )
    original_image.save(original_path)

    print(f"Đã lưu ảnh gốc:     {original_path}")
    print(f"Đã lưu CAM xám:     {cam_gray_path}")
    print(f"Đã lưu heatmap:     {heatmap_path}")
    print(f"Đã lưu CAM overlay: {overlay_path}")

    # Có thể tạo mặt nạ nhị phân từ CAM
    if threshold is not None:
        binary_mask = (
            cam_resized >= threshold
        ).astype(np.uint8) * 255

        binary_mask_path = os.path.join(
            save_dir,
            f"cam_mask_threshold_{threshold:.2f}.png"
        )

        Image.fromarray(binary_mask).save(
            binary_mask_path
        )

        print(
            f"Đã lưu mặt nạ CAM:  {binary_mask_path}"
        )


# ============================================================
# 6. Hàm chính
# ============================================================
def main(args):
    if not os.path.isfile(args.weights_path):
        raise FileNotFoundError(
            f"Không tìm thấy file trọng số: {args.weights_path}"
        )

    if not os.path.isfile(args.image_path):
        raise FileNotFoundError(
            f"Không tìm thấy file ảnh: {args.image_path}"
        )

    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError(
            "--alpha phải nằm trong khoảng [0, 1]."
        )

    if args.threshold is not None:
        if not 0.0 <= args.threshold <= 1.0:
            raise ValueError(
                "--threshold phải nằm trong khoảng [0, 1]."
            )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Thiết bị: {device}")
    print(f"Đang load trọng số: {args.weights_path}")

    model = load_model(
        weights_path=args.weights_path,
        device=device
    )

    original_image = Image.open(
        args.image_path
    ).convert("RGB")

    transform = get_transform()

    input_tensor = transform(
        original_image
    ).unsqueeze(0).to(device)

    cam_extractor = CAMExtractor(model)

    try:
        cam, probabilities, predicted_class = (
            cam_extractor.generate_cam(
                input_tensor=input_tensor,
                target_class=args.target_class
            )
        )
    finally:
        cam_extractor.remove_hook()

    class_names = {
        0: "Không có tổn thương",
        1: "Có tổn thương"
    }

    print("\nKết quả phân loại")
    print("-" * 40)
    print(
        f"Xác suất lớp 0 - Không tổn thương: "
        f"{probabilities[0]:.4f}"
    )
    print(
        f"Xác suất lớp 1 - Có tổn thương:    "
        f"{probabilities[1]:.4f}"
    )
    print(
        f"Lớp dự đoán: {predicted_class} - "
        f"{class_names[predicted_class]}"
    )
    print(
        f"CAM đang được tạo cho lớp: "
        f"{args.target_class} - "
        f"{class_names[args.target_class]}"
    )

    save_cam_results(
        original_image=original_image,
        cam=cam,
        save_dir=args.save_dir,
        alpha=args.alpha,
        threshold=args.threshold
    )


# ============================================================
# 7. Cách chạy
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trích xuất CAM từ ResNet18 phân loại tổn thương xương"
    )

    parser.add_argument(
        "--weights_path",
        type=str,
        required=True,
        help="Đường dẫn tới file resnet18_tumor_classifier.pth"
    )

    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Đường dẫn tới ảnh cần trích xuất CAM"
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="cam_results",
        help="Thư mục lưu kết quả"
    )

    parser.add_argument(
        "--target_class",
        type=int,
        default=1,
        choices=[0, 1],
        help="Lớp cần tạo CAM: 0=không tổn thương, 1=có tổn thương"
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Độ trong suốt của heatmap khi chồng lên ảnh"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Ngưỡng tạo mặt nạ nhị phân từ CAM, "
            "ví dụ 0.5. Bỏ trống nếu không cần."
        )
    )

    args = parser.parse_args()
    main(args)