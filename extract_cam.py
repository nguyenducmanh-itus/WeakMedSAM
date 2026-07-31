import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms


# ============================================================
# 1. Khởi tạo ResNet18 giống lúc huấn luyện
# ============================================================
def build_resnet18(num_classes: int = 2) -> nn.Module:
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model


# ============================================================
# 2. Nạp trọng số mô hình
# ============================================================
def load_model(weights_path: str, device: torch.device) -> nn.Module:
    model = build_resnet18(num_classes=2)

    try:
        checkpoint = torch.load(
            weights_path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        # Tương thích với phiên bản PyTorch cũ
        checkpoint = torch.load(
            weights_path,
            map_location=device,
        )

    # Hỗ trợ checkpoint dạng:
    # {"model_state_dict": ...}, {"state_dict": ...}
    # hoặc state_dict được lưu trực tiếp.
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Checkpoint không phải state_dict hợp lệ. "
            "Hãy kiểm tra cách lưu file trọng số."
        )

    # Xử lý trọng số từng được huấn luyện bằng DataParallel.
    cleaned_state_dict = {
        key.replace("module.", ""): value
        for key, value in checkpoint.items()
    }

    model.load_state_dict(cleaned_state_dict, strict=True)
    model = model.to(device)
    model.eval()
    return model


# ============================================================
# 3. Trích xuất CAM từ đầu ra layer4
# ============================================================
class CAMExtractor:
    def __init__(self, model: nn.Module):
        self.model = model
        self.feature_maps = None
        self.hook_handle = self.model.layer4.register_forward_hook(
            self._save_feature_maps
        )

    def _save_feature_maps(self, module, inputs, output):
        self.feature_maps = output.detach()

    def predict_and_generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: int = 1,
    ):
        """
        Trả về:
            cam: CAM chuẩn hóa liên tục trong [0, 1]
            probabilities: xác suất của hai lớp
            predicted_class: lớp được dự đoán
        """
        self.feature_maps = None

        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)

        if self.feature_maps is None:
            raise RuntimeError(
                "Không lấy được feature map từ layer4."
            )

        predicted_class = int(
            torch.argmax(probabilities, dim=1).item()
        )
        probabilities_np = probabilities[0].cpu().numpy()

        feature_map = self.feature_maps[0]
        class_weights = self.model.fc.weight[target_class].detach()

        cam = torch.einsum(
            "c,chw->hw",
            class_weights,
            feature_map,
        )

        cam = torch.relu(cam)
        cam = cam - cam.min()

        max_value = cam.max()
        if max_value > 0:
            cam = cam / max_value

        return (
            cam.cpu().numpy(),
            probabilities_np,
            predicted_class,
        )

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
            std=[0.229, 0.224, 0.225],
        ),
    ])


# ============================================================
# 5. Tìm tất cả ảnh trong thư mục
# ============================================================
SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def find_images(images_dir: str, recursive: bool = False):
    root = Path(images_dir)

    iterator = root.rglob("*") if recursive else root.glob("*")

    image_paths = [
        path
        for path in iterator
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    return sorted(image_paths)


# ============================================================
# 6. Lưu đúng 3 kết quả:
#    - ảnh gốc
#    - CAM chuẩn hóa nhị phân
#    - CAM overlay
# ============================================================
def save_cam_results(
    original_image: Image.Image,
    cam: np.ndarray,
    source_path: Path,
    images_root: Path,
    save_dir: str,
    alpha: float,
    cam_threshold: float,
):
    save_root = Path(save_dir)

    # Giữ lại cấu trúc thư mục con nếu dùng --recursive.
    relative_path = source_path.relative_to(images_root)
    relative_parent = relative_path.parent
    output_name = f"{source_path.stem}.png"

    original_dir = save_root / "original" / relative_parent
    binary_cam_dir = save_root / "binary_cam" / relative_parent
    overlay_dir = save_root / "overlay" / relative_parent

    original_dir.mkdir(parents=True, exist_ok=True)
    binary_cam_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    # Phóng CAM từ kích thước feature map về kích thước ảnh gốc.
    cam_uint8 = np.clip(cam * 255.0, 0, 255).astype(np.uint8)

    cam_image = Image.fromarray(
        cam_uint8,
        mode="L",
    ).resize(
        original_image.size,
        Image.Resampling.BILINEAR,
    )

    # CAM liên tục đã chuẩn hóa về [0, 1].
    cam_normalized = (
        np.asarray(cam_image, dtype=np.float32) / 255.0
    )

    # CAM nhị phân: 0 hoặc 255.
    binary_cam = (
        cam_normalized >= cam_threshold
    ).astype(np.uint8) * 255

    # Tạo heatmap từ CAM liên tục để overlay đẹp hơn.
    colormap = plt.get_cmap("jet")
    heatmap = colormap(cam_normalized)[..., :3]
    heatmap = np.clip(
        heatmap * 255.0,
        0,
        255,
    ).astype(np.uint8)

    original_array = np.asarray(
        original_image,
        dtype=np.float32,
    )

    overlay = (
        (1.0 - alpha) * original_array
        + alpha * heatmap.astype(np.float32)
    )
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    original_path = original_dir / output_name
    binary_cam_path = binary_cam_dir / output_name
    overlay_path = overlay_dir / output_name

    original_image.save(original_path)
    Image.fromarray(binary_cam, mode="L").save(binary_cam_path)
    Image.fromarray(overlay, mode="RGB").save(overlay_path)

    return original_path, binary_cam_path, overlay_path


# ============================================================
# 7. Hàm chính
# ============================================================
def main(args):
    weights_path = Path(args.weights_path)
    images_root = Path(args.images_dir)
    save_root = Path(args.save_dir)

    if not weights_path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy file trọng số: {weights_path}"
        )

    if not images_root.is_dir():
        raise NotADirectoryError(
            f"Không tìm thấy thư mục ảnh: {images_root}"
        )

    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError(
            "--alpha phải nằm trong khoảng [0, 1]."
        )

    if not 0.0 <= args.cam_threshold <= 1.0:
        raise ValueError(
            "--cam_threshold phải nằm trong khoảng [0, 1]."
        )

    if not 0.0 <= args.lesion_prob_threshold <= 1.0:
        raise ValueError(
            "--lesion_prob_threshold phải nằm trong khoảng [0, 1]."
        )

    image_paths = find_images(
        images_dir=str(images_root),
        recursive=args.recursive,
    )

    if not image_paths:
        raise FileNotFoundError(
            f"Không tìm thấy ảnh hợp lệ trong: {images_root}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Thiết bị: {device}")
    print(f"Trọng số: {weights_path}")
    print(f"Thư mục ảnh: {images_root}")
    print(f"Số ảnh tìm thấy: {len(image_paths)}")
    print(
        "Điều kiện lưu: "
        f"P(tổn thương) >= {args.lesion_prob_threshold:.2f}"
    )
    print(
        "Ngưỡng nhị phân CAM: "
        f"{args.cam_threshold:.2f}"
    )
    print("-" * 70)

    model = load_model(
        weights_path=str(weights_path),
        device=device,
    )
    transform = get_transform()
    cam_extractor = CAMExtractor(model)

    total_images = len(image_paths)
    saved_images = 0
    skipped_normal = 0
    failed_images = 0

    try:
        for index, image_path in enumerate(image_paths, start=1):
            try:
                original_image = Image.open(
                    image_path
                ).convert("RGB")

                input_tensor = transform(
                    original_image
                ).unsqueeze(0).to(device)

                cam, probabilities, predicted_class = (
                    cam_extractor.predict_and_generate_cam(
                        input_tensor=input_tensor,
                        target_class=1,
                    )
                )

                lesion_probability = float(probabilities[1])

                # Chỉ lưu ảnh được xem là có tổn thương.
                is_lesion = (
                    predicted_class == 1
                    and lesion_probability
                    >= args.lesion_prob_threshold
                )

                if not is_lesion:
                    skipped_normal += 1
                    print(
                        f"[{index}/{total_images}] Bỏ qua: "
                        f"{image_path.name} | "
                        f"P(tổn thương)={lesion_probability:.4f}"
                    )
                    continue

                original_path, binary_cam_path, overlay_path = (
                    save_cam_results(
                        original_image=original_image,
                        cam=cam,
                        source_path=image_path,
                        images_root=images_root,
                        save_dir=str(save_root),
                        alpha=args.alpha,
                        cam_threshold=args.cam_threshold,
                    )
                )

                saved_images += 1
                print(
                    f"[{index}/{total_images}] Đã lưu: "
                    f"{image_path.name} | "
                    f"P(tổn thương)={lesion_probability:.4f}"
                )
                print(f"  Ảnh gốc:   {original_path}")
                print(f"  CAM nhị phân: {binary_cam_path}")
                print(f"  Overlay:   {overlay_path}")

            except (
                UnidentifiedImageError,
                OSError,
                RuntimeError,
                ValueError,
            ) as error:
                failed_images += 1
                print(
                    f"[{index}/{total_images}] Lỗi ảnh "
                    f"{image_path}: {error}"
                )

    finally:
        cam_extractor.remove_hook()

    print("\n" + "=" * 70)
    print("HOÀN TẤT")
    print(f"Tổng số ảnh tìm thấy:             {total_images}")
    print(f"Số ảnh tổn thương đã lưu:         {saved_images}")
    print(f"Số ảnh không tổn thương bỏ qua:   {skipped_normal}")
    print(f"Số ảnh lỗi:                       {failed_images}")
    print(f"Thư mục kết quả:                  {save_root.resolve()}")
    print("=" * 70)


# ============================================================
# 8. Tham số dòng lệnh
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Phân loại toàn bộ ảnh trong thư mục và chỉ sinh CAM "
            "cho các ảnh được dự đoán có tổn thương."
        )
    )

    parser.add_argument(
        "--weights_path",
        type=str,
        required=True,
        help="Đường dẫn tới file trọng số .pth",
    )

    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Thư mục chứa các ảnh đầu vào",
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="cam_results",
        help="Thư mục lưu kết quả",
    )

    parser.add_argument(
        "--cam_threshold",
        type=float,
        default=0.5,
        help=(
            "Ngưỡng biến CAM chuẩn hóa thành CAM nhị phân. "
            "Mặc định: 0.5"
        ),
    )

    parser.add_argument(
        "--lesion_prob_threshold",
        type=float,
        default=0.5,
        help=(
            "Ngưỡng xác suất lớp tổn thương để lưu ảnh. "
            "Mặc định: 0.5"
        ),
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help=(
            "Độ trong suốt của heatmap khi chồng lên ảnh. "
            "Mặc định: 0.45"
        ),
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Duyệt cả các thư mục con bên trong images_dir",
    )

    args = parser.parse_args()
    main(args)