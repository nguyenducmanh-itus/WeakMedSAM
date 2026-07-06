from classify_MIL import BagDataset, collate_fn
from attention_mil import ViTAttentionMIL
import argparse
import os
import torch
from torch.utils.data import  DataLoader
from tqdm import tqdm
import cv2 as cv
import pickle
class LocalBagDataset(BagDataset) :
    def __getitem__(self, idx):
        data = torch.load(
                self.pt_files[idx],
                weights_only=False
            )
        img_path = data['image_path']
        coords = data['coords']
        label = data['label']
        img = cv.imread(img_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        patches = []
        for x, y in coords:
            patch = img[y:y+self.patch_size, x:x+self.patch_size]
            patch_tensor = self.preprocess_val(patch)
            patches.append(patch_tensor)
            
        bag_tensor = torch.stack(patches)
        
        return bag_tensor, torch.tensor([label], dtype=torch.float32), coords, img_path 

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_dir", type = str)
    parser.add_argument("--dir_img", type = str)
    parser.add_argument("--model_ckpt", type = str)
    parser.add_argument("--save_dir", type = str)
    parser.add_argument("--save_bbox", type = str)
    args = parser.parse_args()
    print(args)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViTAttentionMIL()
    checkpoint = torch.load(args.model_ckpt)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    bbox_map = {}
    all_pt_files = [os.path.join(args.pt_dir, f) for f in os.listdir(args.pt_dir)]
    print(all_pt_files)
    dataset = LocalBagDataset(args.dir_img, all_pt_files, is_train=False)
    dataloader = DataLoader(dataset, batch_size=1, 
                            collate_fn=collate_fn
                            )
    
    padding = 20
    patch_size = 224
    pdm = tqdm((1, len(dataloader) + 1))
    data_iter = iter(dataloader)
    with torch.no_grad() :
        for n_iter in pdm :
            patches, label, coords, img_path = next(data_iter)
            if label.item() == 0 :
                continue
            print(img_path.split("/")[-1])
            patches = patches.to(device)
            _, A = model(patches, chunk_size=32) 
            
            best_patch_idx = torch.argmax(A, dim=1).item()
            best_x, best_y = coords[best_patch_idx]
            
            x_min = max(0, best_x - padding)
            y_min = max(0, best_y - padding)
            x_max = best_x + patch_size + padding
            y_max = best_y + patch_size + padding
            img_split = img_path.split("/")[-1]
            img = cv.imread(img_path)
            if img is not None : 
                crop_img = img[y_min : y_max, x_min : x_max]
                file_name, ext = os.path.splitext(img_split)
                new_file_name = f"{file_name}_crop{ext}"
                cv.imwrite(os.path.join(args.save_dir, new_file_name), crop_img)
                bbox_map[file_name] = [x_min, y_min, x_max, y_max]
            else : 
                print("None iamge")
    with open(args.save_bbox, 'wb') as f :
        pickle.dump(bbox_map, f)

    