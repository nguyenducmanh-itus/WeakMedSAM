import torch
import torch.nn as nn
from sklearn.cluster import KMeans
import numpy as np
from torchvision.models import resnet18, resnet50
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import argparse
import importlib
import pickle


def length_each_cluster(cluster_labels, child_classes, len_all_cluster) :
    cluster_len = {}
    for i in range(child_classes) :
        cluster_len[i] = 0
    for j in range(len_all_cluster) :
        cluster_len[int(cluster_labels[j])] += 1
            
    return cluster_len

def cluster_cause_imbalance(cluster_len) :
    cluster_index = []
    for keys, values in cluster_len.items() :
        if values < 20 :
            cluster_index.append(int(keys))
    return cluster_index
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--data_module", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--df_path", type=str)
    parser.add_argument("--parent_classes", type=int)
    parser.add_argument("--child_classes", type=int)
    parser.add_argument("--gpus", type=str)
    args = parser.parse_args()
    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    os.makedirs(args.save_path, exist_ok=True)
    resnet = resnet18(weights="DEFAULT").cuda()
    resnet.fc = torch.nn.Identity()
    resnet.eval()
    data_module = importlib.import_module(f"{args.data_module}.dataset")
    dataset = data_module.get_all_dataset(args.df_path, args.data_path, 
                                          4, 
                                          0,
                                          "Body_part_classes/btxrd-4.bin", 
                                          "")
    data_loader = DataLoader(
        dataset,
        args.batch_size,
        drop_last=False,
        pin_memory=True,
        num_workers=4,
    )
    class_features = [None] * args.parent_classes
    idx_list = []
    for _ in range(args.parent_classes):
        idx_list.append([])

    all_idx_list = []

    with torch.no_grad():
        for i, pack in tqdm(enumerate(data_loader), ncols=80, total=len(data_loader)):
            imgs = pack["img"]
            p_lab = pack["plab"]
            idxs = pack["idx"]
            all_idx_list += idxs

            features = resnet(imgs.cuda()).cpu().numpy()

            for b, f in enumerate(features):
                #This code for cluster body part
                if args.parent_classes == 1 :
                    if class_features[0] is None :
                        class_features[0] = []
                    class_features[0].append(f)
                    idx_list[0].append(idxs[b])
                #This code for cluster co-occurance latent in dataset
                else : 
                    lab = pack["part_clab"]
                    for c in range(args.parent_classes):
                        if lab[b, c] != 0  and p_lab[b, 0] == 1:
                            if class_features[c] is None:
                                class_features[c] = []
                            class_features[c].append(f)
                            idx_list[c].append(idxs[b])

    save_map = {idx: np.zeros(args.parent_classes) for idx in all_idx_list}
    for c in range(args.parent_classes):
        kmeans = KMeans(n_clusters=args.child_classes, random_state=42)
        kmeans.fit(class_features[c])
        lbs = list(kmeans.labels_)
        l_e_cluster = length_each_cluster(lbs, 3, len(lbs))
        idx_make_imbalance = cluster_cause_imbalance(l_e_cluster)
        orginal_index = np.arange(len(kmeans.cluster_centers_))
        filtered_idx = np.delete(orginal_index, idx_make_imbalance)
        centers = np.delete(kmeans.cluster_centers_, idx_make_imbalance, axis = 0)
        for i, idx in enumerate(idx_list[c]):
            if int(lbs[i]) in idx_make_imbalance :
                sample = class_features[c][i].reshape(
                    (-1, 
                     len(class_features[c][i])
                    )
                )
                dist = np.linalg.norm(centers - sample, axis = 1)
                new_idx = np.argmin(dist)
                save_map[idx][c] = filtered_idx[new_idx] + 1
            else : 
                save_map[idx][c] = lbs[i] + 1


    
    with open(
        os.path.join(
            args.save_path, f"{str(args.data_module)}-fix-{args.child_classes}.bin"
        ),
        "wb",
    ) as f:
        pickle.dump(save_map, f)
