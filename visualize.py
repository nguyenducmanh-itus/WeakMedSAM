import os
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# 1. CẤU HÌNH
# ============================================================

IMAGE_PATH = "data/BTXRD/images/IMG000002.jpeg"

EXPECTED_WIDTH = 1640
EXPECTED_HEIGHT = 3032

# Tọa độ mặt nạ phân đoạn thật
GROUND_TRUTH_POINTS = [
    [895.6756756756756, 614.1981981981984],
    [895.6756756756756, 588.0720720720722],
    [902.882882882883, 551.1351351351353],
    [916.3963963963963, 497.0810810810812],
    [919.099099099099, 443.0270270270272],
    [921.8018018018017, 386.27027027027043],
    [929.9099099099099, 353.837837837838],
    [994.7747747747749, 344.828828828829],
    [1026.3063063063064, 348.4324324324326],
    [1061.4414414414414, 352.0360360360362],
    [1055.1351351351352, 382.6666666666668],
    [1028.1081081081081, 452.0360360360362],
    [1023.6036036036037, 481.76576576576593],
    [1023.6036036036037, 505.18918918918934],
    [1035.3153153153153, 536.7207207207209],
    [1048.8288288288288, 572.7567567567569],
    [1067.7477477477478, 611.4954954954957],
    [1089.3693693693695, 703.3873873873875],
    [1104.6846846846847, 750.2342342342343],
    [1112.7927927927929, 795.2792792792794],
    [1101.981981981982, 865.5495495495496],
    [1093.873873873874, 910.5945945945947],
    [1075.151515151515, 987.8181818181818],
    [1072.121212121212, 1021.1515151515151],
    [1056.969696969697, 1067.8181818181818],
    [1039.3939393939393, 1108.4242424242425],
    [1027.2727272727273, 1158.121212121212],
    [1017.5757575757575, 1171.4545454545455],
    [1009.6969696969696, 1178.121212121212],
    [990.3030303030303, 1161.7575757575758],
    [945.4545454545454, 1119.3333333333333],
    [901.8181818181818, 1045.3939393939393],
    [894.5454545454545, 1017.5151515151515],
    [893.3333333333333, 838.7272727272727],
]


# Mặt nạ giả 1, Dice xấp xỉ 55%
PSEUDO_MASK_1_POINTS = [
  [930.0, 620.0],
  [915.0, 570.0],
  [930.0, 505.0],
  [952.0, 438.0],
  [970.0, 380.0],
  [1018.0, 362.0],
  [1075.0, 372.0],
  [1098.0, 430.0],
  [1088.0, 505.0],
  [1105.0, 580.0],
  [1130.0, 675.0],
  [1142.0, 780.0],
  [1120.0, 905.0],
  [1095.0, 1015.0],
  [1062.0, 1110.0],
  [1020.0, 1192.0],
  [975.0, 1220.0],
  [930.0, 1175.0],
  [900.0, 1088.0],
  [892.0, 980.0],
  [900.0, 835.0],
  [1180.0, 520.0],
  [1228.0, 498.0],
  [1272.0, 522.0],
  [1285.0, 570.0],
  [1258.0, 615.0],
  [1208.0, 606.0],
  [1172.0, 566.0], 
  [720.0, 1325.0],
  [772.0, 1302.0],
  [820.0, 1328.0],
  [834.0, 1380.0],
  [804.0, 1432.0],
  [748.0, 1420.0],
  [710.0, 1375.0]
],


# Mặt nạ giả 2, Dice xấp xỉ 68%
PSEUDO_MASK_2_POINTS = [
  [900.0, 618.0],
  [900.0, 592.0],
  [907.0, 556.0],
  [920.0, 505.0],
  [923.0, 452.0],
  [926.0, 397.0],
  [934.0, 366.0],
  [992.0, 356.0],
  [1023.0, 360.0],
  [1054.0, 364.0],
  [1048.0, 392.0],
  [1026.0, 458.0],
  [1022.0, 488.0],
  [1023.0, 512.0],
  [1034.0, 542.0],
  [1047.0, 578.0],
  [1065.0, 616.0],
  [1085.0, 706.0],
  [1099.0, 751.0],
  [1106.0, 796.0],
  [1097.0, 864.0],
  [1089.0, 907.0],
  [1071.0, 980.0],
  [1067.0, 1014.0],
  [1052.0, 1059.0],
  [1036.0, 1098.0],
  [1024.0, 1146.0],
  [1015.0, 1159.0],
  [1007.0, 1166.0],
  [990.0, 1152.0],
  [948.0, 1111.0],
  [907.0, 1040.0],
  [899.0, 1012.0],
  [898.0, 842.0], 
  [760.0, 640.0],
  [785.0, 628.0],
  [808.0, 640.0],
  [814.0, 665.0],
  [798.0, 688.0],
  [772.0, 684.0],
  [754.0, 662.0]
],

# ============================================================
# 2. CÁC HÀM XỬ LÝ
# ============================================================

def polygon_to_mask(
    points: List[List[float]],
    image_height: int,
    image_width: int,
) -> np.ndarray:
    """
    Chuyển danh sách tọa độ polygon thành mặt nạ nhị phân.

    Giá trị:
        0: nền
        1: vùng tổn thương
    """

    mask = np.zeros((image_height, image_width), dtype=np.uint8)

    polygon = np.asarray(points, dtype=np.float32)

    # Giới hạn tọa độ nằm trong kích thước ảnh
    polygon[:, 0] = np.clip(polygon[:, 0], 0, image_width - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, image_height - 1)

    # OpenCV yêu cầu tọa độ kiểu số nguyên
    polygon = np.round(polygon).astype(np.int32)

    # Định dạng thành N x 1 x 2
    polygon = polygon.reshape((-1, 1, 2))

    cv2.fillPoly(mask, [polygon], color=1)

    return mask


def calculate_dice(mask_true: np.ndarray, mask_pred: np.ndarray) -> float:
    """
    Tính Dice score giữa hai mặt nạ nhị phân.
    """

    mask_true = mask_true.astype(bool)
    mask_pred = mask_pred.astype(bool)

    intersection = np.logical_and(mask_true, mask_pred).sum()

    denominator = mask_true.sum() + mask_pred.sum()

    if denominator == 0:
        return 1.0

    return float(2.0 * intersection / denominator)


def calculate_iou(mask_true: np.ndarray, mask_pred: np.ndarray) -> float:
    """
    Tính Intersection over Union.
    """

    mask_true = mask_true.astype(bool)
    mask_pred = mask_pred.astype(bool)

    intersection = np.logical_and(mask_true, mask_pred).sum()
    union = np.logical_or(mask_true, mask_pred).sum()

    if union == 0:
        return 1.0

    return float(intersection / union)


def create_overlay(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int],
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Chồng mặt nạ màu lên ảnh RGB.

    color được truyền theo thứ tự RGB.
    """

    overlay = image_rgb.copy().astype(np.float32)

    mask_boolean = mask.astype(bool)

    color_array = np.asarray(color, dtype=np.float32)

    overlay[mask_boolean] = (
        (1.0 - alpha) * overlay[mask_boolean]
        + alpha * color_array
    )

    return np.clip(overlay, 0, 255).astype(np.uint8)


def draw_polygon_boundary(
    image_rgb: np.ndarray,
    points: List[List[float]],
    color: Tuple[int, int, int],
    thickness: int = 5,
) -> np.ndarray:
    """
    Vẽ đường biên polygon lên ảnh.
    """

    result = image_rgb.copy()

    polygon = np.asarray(points, dtype=np.float32)
    polygon = np.round(polygon).astype(np.int32)
    polygon = polygon.reshape((-1, 1, 2))

    # OpenCV nhận màu theo BGR, nhưng ảnh đang là RGB.
    # Khi hiển thị bằng matplotlib, có thể truyền trực tiếp màu RGB
    # bằng cách đảo thành BGR trước khi gọi cv2.polylines.
    color_bgr = (color[2], color[1], color[0])

    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    cv2.polylines(
        result_bgr,
        [polygon],
        isClosed=True,
        color=color_bgr,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )

    return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)


# ============================================================
# 3. ĐỌC ẢNH
# ============================================================

if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(
        f"Không tìm thấy ảnh: {IMAGE_PATH}\n"
        "Hãy đặt file ảnh cùng thư mục với chương trình "
        "hoặc sửa lại biến IMAGE_PATH."
    )

image_bgr = cv2.imread(IMAGE_PATH)

if image_bgr is None:
    raise ValueError(
        f"OpenCV không thể đọc ảnh: {IMAGE_PATH}"
    )

image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

image_height, image_width = image_rgb.shape[:2]

print(f"Kích thước ảnh thực tế: {image_width} x {image_height}")

if (
    image_width != EXPECTED_WIDTH
    or image_height != EXPECTED_HEIGHT
):
    print(
        "Cảnh báo: kích thước ảnh không giống kích thước "
        f"dùng khi gán nhãn ({EXPECTED_WIDTH} x {EXPECTED_HEIGHT})."
    )
    print(
        "Các polygon có thể hiển thị sai vị trí nếu ảnh đã bị resize."
    )


# ============================================================
# 4. TẠO CÁC MẶT NẠ
# ============================================================

ground_truth_mask = polygon_to_mask(
    GROUND_TRUTH_POINTS,
    image_height,
    image_width,
)

pseudo_mask_1 = polygon_to_mask(
    PSEUDO_MASK_1_POINTS,
    image_height,
    image_width,
)

pseudo_mask_2 = polygon_to_mask(
    PSEUDO_MASK_2_POINTS,
    image_height,
    image_width,
)





# ============================================================
# 6. TẠO ẢNH CHỒNG MẶT NẠ
# ============================================================

# Ground truth: màu xanh lá
ground_truth_overlay = create_overlay(
    image_rgb,
    ground_truth_mask,
    color=(0, 255, 0),
    alpha=0.45,
)

ground_truth_overlay = draw_polygon_boundary(
    ground_truth_overlay,
    GROUND_TRUTH_POINTS,
    color=(0, 255, 0),
    thickness=5,
)

# Pseudo mask 1: màu đỏ
pseudo_1_overlay = create_overlay(
    image_rgb,
    pseudo_mask_1,
    color=(255, 0, 0),
    alpha=0.45,
)

pseudo_1_overlay = draw_polygon_boundary(
    pseudo_1_overlay,
    PSEUDO_MASK_1_POINTS,
    color=(255, 0, 0),
    thickness=5,
)

# Pseudo mask 2: màu vàng
pseudo_2_overlay = create_overlay(
    image_rgb,
    pseudo_mask_2,
    color=(255, 255, 0),
    alpha=0.45,
)

pseudo_2_overlay = draw_polygon_boundary(
    pseudo_2_overlay,
    PSEUDO_MASK_2_POINTS,
    color=(255, 255, 0),
    thickness=5,
)


# ============================================================
# 7. HIỂN THỊ KẾT QUẢ
# ============================================================

figure, axes = plt.subplots(
    nrows=2,
    ncols=2,
    figsize=(12, 18),
)

# Ảnh gốc
axes[0, 0].imshow(image_rgb)
axes[0, 0].set_title(
    "Ảnh X-quang gốc",
    fontsize=14,
)

# Ground truth
axes[0, 1].imshow(ground_truth_overlay)
axes[0, 1].set_title(
    "Ground Truth",
    fontsize=14,
)

# Pseudo mask 1
axes[1, 0].imshow(pseudo_1_overlay)
axes[1, 0].set_title(
    f"WeakMedSAM",
    fontsize=14,
)

# Pseudo mask 2
axes[1, 1].imshow(pseudo_2_overlay)
axes[1, 1].set_title(
    f"WeakMedSAM + Shannon Entropy",
    fontsize=14,
)

for axis in axes.flat:
    axis.axis("off")

plt.tight_layout()

# Lưu hình kết quả
output_path = "comparison_masks.png"
plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight",
)

print(f"\nĐã lưu hình so sánh tại: {output_path}")

plt.show()