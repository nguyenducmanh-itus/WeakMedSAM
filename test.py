import torch

pt = torch.load("output_patch_image/IMG000001.pt", 
                weights_only=False)

print(pt["image_path"])