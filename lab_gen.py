from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch
import argparse
import os
from samus.build_sam_us import samus_model_registry
from tqdm import tqdm
import importlib
from utils.torchutils import max_norm
from utils.affinity import get_tran
import numpy as np
from PIL import Image
import torch.multiprocessing as mp
import csv
from utils.pseudo_label_qc import (
    SoftIoUQualityControl,
    normalize_cam,
)

def worker(rank, subsets, gpus, args):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus[rank]
    subset = subsets[rank]
    sub_loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=4,
    )

    model = samus_model_registry["vit_b"](
        parent_classes=args.parent_classes,
        child_classes=args.child_classes,
        checkpoint=args.sam_ckpt,
    )
    if args.samus_ckpt:
        checkpoint = torch.load(args.samus_ckpt)
        model.load_state_dict(checkpoint)
    model = model.cuda()
    model.eval()

    pbar = tqdm(enumerate(sub_loader), total=len(sub_loader), desc=f"Rank: {rank}")
    with torch.no_grad():
        for i, pack in pbar:
            imgs = pack["img"].cuda()
            idxs = pack["idx"]

            x, _, cam = model(imgs)
            pred = (torch.sigmoid(x) > 0.5).float()

            trans_mat, _ = get_tran(imgs, model, beta=args.beta, grid_ratio=4)
            rw_cam = F.interpolate(cam, (64, 64), mode="bilinear")
            for i in range(args.t):
                rw_cam = (
                    torch.bmm(
                        trans_mat,
                        rw_cam.permute(0, 2, 3, 1).reshape(
                            rw_cam.size(0), -1, args.parent_classes
                        ),
                    )
                    .permute(0, 2, 1)
                    .reshape_as(rw_cam)
                )
                rw_cam = max_norm(rw_cam)
            rw_cam = F.interpolate(
                rw_cam, (imgs.size(2), imgs.size(3)), mode="bilinear"
            )
            rw_cam = TF.gaussian_blur(rw_cam, kernel_size=21)
            rw_cam *= pred.view(pred.size(0), pred.size(1), 1, 1).expand_as(rw_cam)

            for i, c in enumerate(rw_cam):
                c = c.cpu().numpy()[0]
                Image.fromarray(
                    (c > args.threshold).astype(np.uint8) * 255, mode="L"
                ).save(os.path.join(args.save_path, f"{idxs[i]}.png"))


def worker_QC(rank, subsets, gpus, args):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus[rank]

    subset = subsets[rank]

    sub_loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=4,
    )

    model = samus_model_registry["vit_b"](
        parent_classes=args.parent_classes,
        child_classes=args.child_classes,
        checkpoint=args.sam_ckpt,
    )

    if args.samus_ckpt:
        checkpoint = torch.load(
            args.samus_ckpt,
            map_location="cpu",
        )
        model.load_state_dict(checkpoint)

    model = model.cuda()
    model.eval()

    csv_path = os.path.join(
        args.save_path,
        f"pseudo_label_quality_rank_{rank}.csv",
    )

    csv_file = open(
        csv_path,
        mode="w",
        newline="",
        encoding="utf-8",
    )

    csv_writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "image_id",
            "quality",
            "classification_prediction",
            "mask_path",
        ],
    )

    csv_writer.writeheader()

    pbar = tqdm(
        enumerate(sub_loader),
        total=len(sub_loader),
        desc=f"Rank: {rank}",
    )

    with torch.no_grad():
        for _, pack in pbar:
            imgs = pack["img"].cuda(non_blocking=True)
            idxs = pack["idx"]

           
            x, _, cam = model(imgs)

            prediction = (
                torch.sigmoid(x) > 0.5
            ).float()

            original_cam = F.interpolate(
                cam,
                size=imgs.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            original_cam = normalize_cam(original_cam)

            
            trans_mat, _ = get_tran(
                imgs,
                model,
                beta=args.beta,
                grid_ratio=4,
            )

            refined_cam = F.interpolate(
                cam,
                size=(64, 64),
                mode="bilinear",
                align_corners=False,
            )

            refined_cam = max_norm(refined_cam)

            for _ in range(args.t):
                flattened_cam = (
                    refined_cam
                    .permute(0, 2, 3, 1)
                    .reshape(
                        refined_cam.size(0),
                        -1,
                        args.parent_classes,
                    )
                )

                refined_cam = torch.bmm(
                    trans_mat,
                    flattened_cam,
                )

                refined_cam = (
                    refined_cam
                    .permute(0, 2, 1)
                    .reshape_as(
                        F.interpolate(
                            cam,
                            size=(64, 64),
                            mode="bilinear",
                            align_corners=False,
                        )
                    )
                )

                refined_cam = max_norm(refined_cam)

            refined_cam = F.interpolate(
                refined_cam,
                size=imgs.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

           
            refined_cam = TF.gaussian_blur(
                refined_cam,
                kernel_size=21,
            )

            refined_cam = normalize_cam(refined_cam)

            
            quality_scores = calculate_pseudo_label_quality(
                cam_before=original_cam,
                cam_after=refined_cam,
                class_prediction=prediction,
            )

            
            pseudo_cam = refined_cam * prediction.view(
                prediction.size(0),
                prediction.size(1),
                1,
                1,
            )

            for batch_index, sample_cam in enumerate(pseudo_cam):
                image_id = idxs[batch_index]

                if torch.is_tensor(image_id):
                    image_id = image_id.item()

                image_id = str(image_id)

                quality_value = float(
                    quality_scores[batch_index].item()
                )

                
                sample_cam = sample_cam[0].cpu().numpy()

                binary_mask = (
                    sample_cam > args.threshold
                ).astype(np.uint8) * 255

                mask_path = os.path.join(
                    args.save_path,
                    f"{image_id}.png",
                )

                Image.fromarray(
                    binary_mask,
                    mode="L",
                ).save(mask_path)

                predicted_class = int(
                    prediction[batch_index]
                    .max()
                    .item()
                )

                csv_writer.writerow(
                    {
                        "image_id": image_id,
                        "quality": f"{quality_value:.8f}",
                        "classification_prediction": predicted_class,
                        "mask_path": mask_path,
                    }
                )

    csv_file.close()

def merge_quality_csv(save_path, number_of_workers):
    output_csv_path = os.path.join(
        save_path,
        "pseudo_label_quality.csv",
    )

    fieldnames = [
        "image_id",
        "quality",
        "classification_prediction",
        "mask_path",
    ]

    with open(
        output_csv_path,
        mode="w",
        newline="",
        encoding="utf-8",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for rank in range(number_of_workers):
            rank_csv_path = os.path.join(
                save_path,
                f"pseudo_label_quality_rank_{rank}.csv",
            )

            if not os.path.exists(rank_csv_path):
                continue

            with open(
                rank_csv_path,
                mode="r",
                newline="",
                encoding="utf-8",
            ) as input_file:
                reader = csv.DictReader(input_file)

                for row in reader:
                    writer.writerow(row)

            os.remove(rank_csv_path)

    print(
        "Đã lưu điểm chất lượng nhãn giả tại: "
        f"{output_csv_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--save-path", type=str)
    parser.add_argument("--data-module", type=str)
    parser.add_argument("--vit-name", type=str)
    parser.add_argument("--sam-ckpt", type=str)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--samus-ckpt", type=str)
    parser.add_argument("--parent-classes", type=int)
    parser.add_argument("--child-classes", type=int)
    parser.add_argument("--t", type=int)
    parser.add_argument("--beta", type=int)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--gpus", type=str)
    parser.add_argument("--qc", type=float)
    args = parser.parse_args()
    print(args)

    os.makedirs(args.save_path, exist_ok=True)

    data_module = importlib.import_module(f"{args.data_module}.dataset")
    dataset = data_module.get_all_dataset(args.data_path, 0, "")
    gpus = args.gpus.split(",")
    subset_size = len(dataset) // len(gpus) + 1
    subsets = []
    start_idx = 0
    for i in range(len(gpus)):
        end_idx = min(start_idx + subset_size, len(dataset))
        subset_indices = list(range(start_idx, end_idx))
        subsets.append(torch.utils.data.Subset(dataset, subset_indices))
        start_idx = end_idx
    assert sum([len(subset) for subset in subsets]) == len(dataset)
    mp.spawn(
        worker,
        args=(subsets, gpus, args),
        nprocs=len(gpus),
        join=True,
    )

    merge_quality_csv(
        save_path=args.save_path,
        number_of_workers=len(gpus),
    )
