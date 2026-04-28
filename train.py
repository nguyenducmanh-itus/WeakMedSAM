from torch.utils.data import DataLoader
import torch
import argparse
import os
from samus.build_sam_us import samus_model_registry
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import importlib
from utils.pytuils import AverageMeter
import torch.nn.functional as F
from utils.metrics import dice



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--data_module", type=str)
    parser.add_argument("--vit_name", type=str)
    parser.add_argument("--sam_ckpt", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--index", type=str)
    parser.add_argument("--samus_ckpt", type=str)
    parser.add_argument("--max_epochs", type=int)
    parser.add_argument("--val_iters", type=int)
    parser.add_argument("--parent_classes", type=int)
    parser.add_argument("--child_classes", type=int)
    parser.add_argument("--child_weight", type=float)
    parser.add_argument("--cluster_file", type=str)
    parser.add_argument("--dataframe_path", type = str)
    parser.add_argument("--logdir", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--gpus", type=str)
    args = parser.parse_args()
    print(args)
    if args.gpus != '-1':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    #Init 
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpus != '-1' else "cpu")

    #Set up unifrom random initialization in CPU
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        #If GPU exist, init unifrom ramdom in GPU
        torch.cuda.manual_seed(args.seed)
    #Turn on deterministic
    torch.backends.cudnn.deterministic = True
    #Init model
    model = samus_model_registry["vit_b"](
        parent_classes=args.parent_classes + 1,
        child_classes=args.child_classes,
        checkpoint=args.sam_ckpt,
    )
    #Set up model training on multiple GPU kernels in parallel
    model = torch.nn.DataParallel(model).cuda()
    
    if args.samus_ckpt:
        #Load checkpoints from disk
        checkpoint = torch.load(args.samus_ckpt)
        #Load model parameters from the checkpoint into the model
        model.load_state_dict(checkpoint)

    #Get data module of file datasets
    data_module = importlib.import_module(f"{args.data_module}.dataset")
    #Get train and valid dataset
    train_dataset, val_dataset, _ = data_module.get_dataset(
        cluster_file = args.cluster_file, 
        begin_feature = "osteochondroma", 
        end_feature = "other mt", 
        child_classes = args.child_classes, 
        #This argument below adapt with dataset module
        df_path = args.dataframe_path
    )
    #Get train loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=4,
    )
    #Get val loader
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=4,
    )
    #Init max iterations in training process
    args.max_iters = args.max_epochs * len(train_loader)
    #Init optimizer AdamW
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
    )
    #Schedule adjustments learning rate
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer, max_lr=args.lr, total_steps=args.max_iters
    )

    #Create tensorboard
    writer = SummaryWriter(os.path.join(args.logdir, args.index))
    #create a tqdm objects with max_iters loop and width of column visualize is 100 pixel 
    pbar = tqdm(range(1, args.max_iters + 1), ncols=100)
    #load iter data
    train_loader_iter = iter(train_loader)
    #Loop max_iters iterations
    for n_iter in pbar:
        #Set model to train
        model.train()
        #Set total accumulated gradients to zero
        optimizer.zero_grad()
        try:
            #Get batch data in train_loader
            datapack = next(train_loader_iter)

        except StopIteration :
            #If complete one epoch reset iterator objects
            train_loader_iter = iter(train_loader)
            #Get batch data in train_loader
            datapack = next(train_loader_iter)

        #Get batch tensor images and save in GPU
        imgs = datapack["img"].to(device)
        #ground truth of parent label and save in GPU
        parent_labs = datapack["plab"].to(device)
        #Get ground truth of children label and save in GPU
        if args.child_classes > 0:
            child_labs = datapack["clab"].to(device)

        #Get batch prediction results of the parent label and children label respectively.
        parent_x, child_x, _ = model(imgs)
        
        #Calculate loss of parent classifier
        parent_loss = F.binary_cross_entropy_with_logits(
            parent_x,
            parent_labs,
        )
        #Calculate loss of children classifier
        if args.child_classes > 0:
            child_loss = F.binary_cross_entropy_with_logits(child_x, child_labs)
            #Aggregate loss of parent and children 
            loss = parent_loss + args.child_weight * child_loss
        else:
            loss = parent_loss
        
        #backpropagaion to calculate gradient
        loss.backward()
        #Update parameters
        optimizer.step()
        #Update learning rate
        scheduler.step()

        #Get score and prediction in parent and child classifier
        parent_pred = (torch.sigmoid(parent_x) > 0.5).float()
        parent_score = torch.eq(parent_pred, parent_labs).sum() / parent_labs.numel()
        child_pred = (torch.sigmoid(child_x) > 0.5).float()
        child_score = torch.eq(child_pred, child_labs).sum() / child_labs.numel()

        #write tensorboard to visualize loss, score, learning rate
        writer.add_scalar("train/train loss", loss.item(), n_iter)
        writer.add_scalar("train/parent loss", parent_loss.item(), n_iter)
        writer.add_scalar("train/child loss", child_loss.item(), n_iter)
        writer.add_scalar("train/parent score", parent_score.item(), n_iter)
        writer.add_scalar("train/child score", child_score.item(), n_iter)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], n_iter)
        #If training has been completed val_iters time 
        #evaluate the generality of model
        if n_iter % args.val_iters == 0:
            #transition model through state evaluate
            #In this evaluation state, model's parameters will not be updated 
            model.eval()
            #Init AverageMeter object to calculate loss and score of validation tests
            val_loss = AverageMeter()
            val_score = AverageMeter()
            with torch.no_grad():
                #Loop over validation dataloader
                for pack in val_loader:
                    #Get batch tensor images
                    imgs = pack["img"].to(device)
                    #Get batch labels
                    labs = pack["plab"].float().to(device)
                    x, _, _ = model(imgs)
                    #parent classifier loss
                    val_loss.add(F.binary_cross_entropy_with_logits(x, labs).item())
                    #prediction
                    pred = (torch.sigmoid(x) > 0.5).float()
                    #torch.eq -> return a boolen tensor where input is equal to other
                    #.numel() -> return total number of elements in labels tensor
                    val_score.add(torch.eq(pred, labs).sum().item(), labs.numel())
            #Transition state of model to train
            model.train()
            writer.add_scalar("val/val loss", val_loss.get(), n_iter)
            writer.add_scalar("val/val score", val_score.get(), n_iter)
        #Visualize total loss, total score and learning rate in tqdm
        pbar.set_postfix(
            {
                "tl": loss.item(),
                "ts": parent_score.item(),
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
    #Save model when training phase completed
    torch.save(
        model.state_dict(),
        os.path.join(args.logdir, args.index, f"{args.index}.pth"),
        _use_new_zipfile_serialization=False,
    )


if __name__ == "__main__":
    main()
