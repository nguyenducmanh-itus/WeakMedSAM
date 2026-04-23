import torch
from sklearn.cluster import KMeans
import numpy as np
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import argparse
import importlib
import pickle


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--data_module", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--parent_classes", type=int)
    parser.add_argument("--child_classes", type=int)
    parser.add_argument("--gpus", type=str)
    args = parser.parse_args()
    print(args)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    os.makedirs(args.save_path, exist_ok=True)

    # resnet = resnet18(weights="DEFAULT").cuda()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet = resnet18(weights="DEFAULT").to(device)
    resnet.fc = torch.nn.Identity()
    resnet.eval()
    
    data_module = importlib.import_module(f"{args.data_module}.dataset")
    dataset = data_module.get_all_dataset(args.data_path, "", "osteochondroma", "other mt", 0)
    data_loader = DataLoader(
        dataset,
        args.batch_size,
        drop_last=False,
        pin_memory=True,
        num_workers=0,
    )
    #Tạo list để lưu danh sách các feature ảnh thuộc về lớp đó
    class_features = [None] * args.parent_classes
    #Danh sách chỉ mục ảnh thuộc về lớp đó.
    idx_list = []
    for _ in range(args.parent_classes):
        idx_list.append([])
    #biến lưu tất cả các chỉ mục ảnh
    all_idx_list = set()

    with torch.no_grad():
        #Load 1 batch ảnh
        for i, pack in tqdm(enumerate(data_loader), ncols=80, total=len(data_loader)):
            imgs = pack["img"]
            lab = pack["plab"]
            idxs = pack["idx"]
            all_idx_list.update(idxs)
            #Trích xuất feature maps của các ảnh trong batch ảnh
            features = resnet(imgs.to(device)).cpu().numpy()
            #Lấy từng feature map
            for b, f in enumerate(features):
                #Kiểm tra xem feature map thứ b có thuộc về lớp thứ c hay không
                for c in range(1, args.parent_classes + 1):
                    #Nếu có lưu feature map đó vào danh sách thông tin các feature map
                    #của lớp c
                    if lab[b, c] != 0:
                        if class_features[c - 1] is None:
                            class_features[c - 1] = []
                        class_features[c - 1].append(f)
                        #Lưu thông tin chỉ mục của ảnh 
                        idx_list[c - 1].append(idxs[b])

    save_map = {idx: np.zeros(args.parent_classes) for idx in all_idx_list}

    for c in range(args.parent_classes):
        if class_features[c] is None or len(class_features[c]) == 0:
            print(f"No features for class {c}, skipping clustering.")
            continue
        features_array = np.array(class_features[c])
        if np.isnan(features_array).any():
            print(f"NaN features for class {c}, skipping clustering.")
            continue
        kmeans = KMeans(n_clusters=args.child_classes)
        kmeans.fit(features_array)
        lbs = list(kmeans.labels_)

        for i, idx in enumerate(idx_list[c]):
            save_map[idx][c] = lbs[i] + 1
    #Save_map là một method set() chứa 
    with open(
        os.path.join(
            args.save_path, f"{str(args.data_module)}-{args.child_classes}.bin"
        ),
        "wb",
    ) as f:
        pickle.dump(save_map, f)
