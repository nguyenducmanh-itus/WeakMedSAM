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
        img_path = data['image_path'].split("/")[-1]
        coords = data['coords']
        label = data['label']
        img = cv.imread(os.path.join(self.dir_img, img_path))
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
    dataset = LocalBagDataset(args.dir_img, all_pt_files, is_train=False)
    dataloader = DataLoader(dataset, batch_size=1, 
                            collate_fn=collate_fn
                            )
    padding = 20
    patch_size = 224
    pdm = tqdm(range(1, len(dataloader) + 1), ncols=100)
    print(f"Length of dataloader : {len(dataloader)}")
    data_iter = iter(dataloader)
    with torch.no_grad() :
        for n_iter in pdm :
            patches, label, coords, img_path = next(data_iter)
            print(img_path)
            if label.item() == 0 :
                continue
            
            patches = patches.to(device)
            _, A = model(patches, chunk_size=32) 
            k = min(5, A.size(1))
            
            topk_vals, topk_indices = torch.topk(A, k, dim=1)
            selected_coords = [coords[idx.item()] for idx in topk_indices[0]]
            clusters = [] # Danh sách chứa các cụm (mỗi cụm là 1 list tọa độ)
            max_dist = patch_size * 1.5
            best_patch_idx = torch.argmax(A, dim=1).item()
            best_x, best_y = coords[best_patch_idx]
            
            x_min = max(0, best_x - padding)
            y_min = max(0, best_y - padding)
            x_max = best_x + patch_size + padding
            y_max = best_y + patch_size + padding
            #img_id = img_path.split("/")[-1]
            idx, ext = os.path.splitext(img_path)
            img = cv.imread(os.path.join(args.dir_img, img_path))
            if img is not None : 
                crop_img = img[y_min : y_max, x_min : x_max]
                # file_name, ext = os.path.splitext(img_split)
                new_file_name = f"{idx}_crop{ext}"
                new_img = cv.resize(img.copy(), (512, 512))
                h, w, _ = img.shape
                resize_h = 512/ h
                resize_w = 512 / w
                x_min_resize = int(resize_w * x_min)
                y_min_resize = int(resize_h * y_min)
                x_max_resize = int(resize_w * x_max)
                y_max_resize = int(resize_h * y_max)
                cv.rectangle(new_img, (x_min_resize, y_min_resize), (x_max_resize, y_max_resize), (0, 255, 255), 2)
                cv.imshow("Crop area", new_img)
                cv.waitKey(0)
                #cv.imwrite(os.path.join(args.save_dir, new_file_name), crop_img)
                bbox_map[img_path] = [x_min, y_min, x_max, y_max]
            else : 
                print("None image")
    with open(args.save_bbox, 'wb') as f :
        pickle.dump(bbox_map, f)

    