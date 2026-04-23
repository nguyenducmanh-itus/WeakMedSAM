import torch
import importlib
from samus.build_sam_us import samus_model_registry

# Thiết lập device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model (giả sử vit_b, parent_classes=1, child_classes=3)
model = samus_model_registry["vit_b"](
    parent_classes=1,
    child_classes=3,
    checkpoint="ckpt/sam_vit_b_01ec64.pth"  # Đường dẫn checkpoint
)
model = model.to(device)
model.eval()

# Tạo input giả (batch_size=2, channels=3, height=256, width=256)
imgs = torch.randn(2, 3, 256, 256).to(device)

# Forward pass
with torch.no_grad():
    parent_x, child_x, cam_output = model(imgs)

# In ra shapes và một số giá trị mẫu
print("Input shape:", imgs.shape)
print("Parent_x shape:", parent_x.shape)
print("Parent_x values (first batch):", parent_x[0])
print("Child_x shape:", child_x.shape)
print("Child_x values (first batch):", child_x[0])
print("Cam_output shape:", cam_output.shape)
print("Cam_output values (first batch, first channel):", cam_output[0, 0, :5, :5])  # In 5x5 góc trên