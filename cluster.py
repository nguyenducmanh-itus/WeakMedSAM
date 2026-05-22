import torch
from sklearn.cluster import KMeans
import numpy as np
from torchvision.models import resnet18, resnet50
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
    parser.add_argument("--df_path", type=str)
    parser.add_argument("--parent_classes", type=int)
    parser.add_argument("--child_classes", type=int)
    parser.add_argument("--gpus", type=str)
    args = parser.parse_args()
    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    os.makedirs(args.save_path, exist_ok=True)
    #ResNet pre-trained in ImageNet
    resnet = resnet50(weights="DEFAULT").cuda()
    #ResNet pre-trained in medical images
    # resnet = resnet50(pretrained = True)
    # path = "C:/Users/ADMIN/OneDrive - VNU-HCMUS/CNTT-HK8/ResNet50.pt"
    # resnet.load_state_dict(torch.load(path, map_location=torch.device('cuda')))
    resnet.fc = torch.nn.Identity()
    resnet.eval()

    data_module = importlib.import_module(f"{args.data_module}.dataset")
    dataset = data_module.get_all_dataset(args.df_path, args.data_path, 0, "")
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
            lab = pack["plab"]
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
                    for c in range(args.parent_classes):
                        if lab[b, c] != 0:
                            if class_features[c] is None:
                                class_features[c] = []
                            class_features[c].append(f)
                            idx_list[c].append(idxs[b])

    save_map = {idx: np.zeros(args.parent_classes) for idx in all_idx_list}
    
    for c in range(args.parent_classes):
        kmeans = KMeans(n_clusters=args.child_classes)
        kmeans.fit(class_features[c])
        lbs = list(kmeans.labels_)

        for i, idx in enumerate(idx_list[c]):
            save_map[idx][c] = lbs[i]

    with open(
        os.path.join(
            args.save_path, f"{str(args.data_module)}-{args.child_classes}.bin"
        ),
        "wb",
    ) as f:
        pickle.dump(save_map, f)
