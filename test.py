import torch

file = torch.load("output_patch_image/IMG000001.pt", 
                  weights_only=False)

print(file["image_path"])