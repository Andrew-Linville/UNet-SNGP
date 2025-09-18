import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from utils import load_checkpoint, save_checkpoint, get_loaders, check_accuracy, save_predictions_as_imgs
from pathlib import Path
from custom_loss import BCEDiceLoss
import matplotlib.pyplot as plt

# from model import UNET
from model_sngp import UNET
LEARNING_RATE = 1E-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
NUM_EPOCHS = 100
NUM_WORKERS = 2
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
PIN_MEMORY = True
LOAD_MODEL = False
# TRAIN_IMG_DIR = r"data/train/imgs"
# TRAIN_MASK_DIR = r"data/train/masks"
# VAL_IMG_DIR = r"data/val/imgs"
# VAL_MASK_DIR = r"data/val/masks"

ROOT = Path(__file__).resolve().parents[1]  
TRAIN_IMG_DIR = ROOT / "data" / "train" / "imgs"
TRAIN_MASK_DIR  = ROOT / "data" / "train" / "masks"
VAL_IMG_DIR = ROOT / "data" / "val" / "imgs"
VAL_MASK_DIR  = ROOT / "data" / "val" / "masks"


# def train_fn(loader, model, optimizer, loss_fn, scaler):
#     loop = tqdm(loader)
    
#     for batch_idx, (data, targets) in enumerate(loop):
#         data = data.to(device=DEVICE)
#         targets = targets.float().unsqueeze(1).to(device=DEVICE)
#         # print(torch.max(targets)) # Tragets are currently binary, 0/1
        
#     # Forward
    
#         with torch.cuda.amp.autocast():
#             predictions = model(data)
#             # print(torch.max(predictions)) logit
#             if targets.ndim == 5 and targets.size(1) == 1:
#                 targets = targets.squeeze(1)             # [B, 2, H, W]
#             targets = targets.float()
#             loss = loss_fn(predictions, targets) # Don't need to worry about applyig sigmoid before, b.c. it should be handled in the loss func
            
#         # backwaed
#         optimizer.zero_grad()
#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()
        
#         # Update tqdm
#         # loop.set_postfix(loss=loss.time())
#         loop.set_postfix(loss=f"{loss.item():.4f}")


#     pass
    
def train_fn(loader, model, optimizer, loss_fn, scaler):
    
    
    # Init training loss 
    train_loss_accum = 0

    for batch_idx, (data, targets) in enumerate(loader):
        
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)
        
    # Forward
    
        with torch.cuda.amp.autocast():
            predictions = model(data)
            if targets.ndim == 5 and targets.size(1) == 1:
                targets = targets.squeeze(1)             # [B, 2, H, W]
            targets = targets.float() 
            loss = loss_fn(predictions, targets)
            train_loss_accum += loss * data.size(0) # Running loss sccaled by batch size
        
        
        print(f"Batch Idx: {batch_idx}")
        print(f"Batch Loss: {loss}")
        # backwaed
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Update tqdm
        # loop.set_postfix(loss=loss.time())
        # loop.set_postfix(loss=f"{loss.item():.4f}")

        
        
    epoch_loss = train_loss_accum/len(loader)
    
    return epoch_loss    

import csv
def save_losses(train_losses, val_losses, filename="losses.csv"):
    
    n_epochs = min(len(train_losses), len(val_losses))
    with open(filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train_loss", "Val_loss"])
        for epoch in range(n_epochs):
            writer.writerow([epoch + 1, train_losses[epoch], val_losses[epoch]])
    
    print(f"Saved losses to {filename}")
    
def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    
    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width = IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    
    pass

    model = UNET(in_channels=3, outchannels=2).to(DEVICE)
    # loss_fn = nn.BCEWithLogitsLoss()
    loss_fn = BCEDiceLoss()
    # loss_fn = nn.CrossEntropy # for multiclass/channel models
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )
    
    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.path.tar"), model)
        
    
    scaler = torch.cuda.amp.GradScaler()
    train_losses = []
    val_losses = []

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch: {epoch}")
        train_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler)
        train_losses.append(train_loss)
        # Only saves every 10 times (speeds up training)
        if (epoch+1) % 10 == 0: 
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint)
            
            val_loss = check_accuracy(val_loader, model, loss_fn, device=DEVICE)
            val_losses.append(val_loss)
        

        
        save_predictions_as_imgs(
            val_loader, model, folder="saved_images/", device=DEVICE
        )

    save_losses(train_losses, val_losses)

if __name__ == "__main__":
    main()