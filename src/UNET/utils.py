import torch
import torchvision
from dataset import fillerDataset
from torch.utils.data import DataLoader
from pathlib import Path

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving Checkpoint")
    torch.save(state, filename)
    
def load_checkpoint(checkpoint, model):
    print("=> Loading Checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
    
):
    
    train_ds = fillerDataset(
        image_dir = train_dir,
        mask_dir = train_maskdir,
        transform = train_transform        
    )
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )
    
    val_ds = fillerDataset(
        image_dir = val_dir,
        mask_dir = val_maskdir,
        transform = val_transform,        
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )
    
    return train_loader, val_loader

# This function is setup for binary only currently
# def check_accuracy(loader, model, device="cuda"):
#     num_correct = 0
#     num_pixels = 0
#     dice_score = 0
#     model.eval()
    
#     with torch.no_grad():
#         for x,y in loader:
#             x = x.to(device)
#             y = y.to(device).unsqueeze(1)
#             preds = torch.sigmoid(model(x))
#             preds = (preds>0.5).float()
#             num_correct += (preds==y).sum()
#             num_pixels += torch.numel(preds)
#             dice_score += (2*(preds*y).sum() / ((preds+y).sum()+1e-8)) 
            
            
#     print(
#         f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100}%"
#     )
#     print(f"Dice score: {dice_score/len(loader)}")
    
#     model.train()

import torch

def check_accuracy(loader, model, loss_fn, device="cuda", threshold=0.5, multilabel=False):
    """
    multilabel=True  -> each output channel is its own binary class (sigmoid+threshold per channel)
    multilabel=False -> binary (1 channel, sigmoid) or multiclass (argmax over C)
    """
    model.eval()
    total_correct = 0
    total_pixels = 0
    dice_sum = 0.0
    eps = 1e-8
    running_loss = 0

    with torch.no_grad():
        for data, targets in loader:
            data = data.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            predictions = model(data)
            targets=targets.float()
            loss = loss_fn(predictions,targets)
            running_loss += loss * data.size(0)
            
            

            
            #? Val Loss
            
            
    # acc = 100.0 * total_correct / max(1, total_pixels)
    # dice = dice_sum / max(1, len(loader))
    # print(f"Pixel Acc: {acc:.2f}%")
    # print(f"Dice: {dice:.4f}")
    epoch_loss = running_loss/len(loader)
    model.train()
    return epoch_loss

# def save_predictions_as_imgs(
#     loader,model,folder="saved_images",device="cuda"
# ):
#     out_dir = Path(folder)
#     out_dir.mkdir(parents=True, exist_ok=True)
#     model.eval()
#     for idx, (x,y) in enumerate(loader):
#         x = x.to(device=device)
#         with torch.no_grad():
#             preds = torch.sigmoid(model(x))
#             preds = (preds>0.5).float()
#         torchvision.utils.save_image(
#             preds,f"{folder}/pred_{idx}.png"
#         )
#         torchvision.utils.save_image(y.unsqueeze(1), f"f")

#     model.train()

# def save_predictions_as_imgs(loader, model, folder="saved_images", device="cuda", threshold=0.5):
#     out_dir = Path(folder)
#     out_dir.mkdir(parents=True, exist_ok=True)

#     model.eval()
#     with torch.no_grad():
#         for idx, (x, y) in enumerate(loader):
#             x = x.to(device)
#             y = y.to(device)

#             logits = model(x)  # (B, C, H, W) or (B, 1, H, W)

#             # --- predictions ---
#             if logits.shape[1] == 1:  # binary
#                 probs = torch.sigmoid(logits)
#                 preds = (probs > threshold).float()               # (B,1,H,W) in {0,1}
#             else:  # multi-class → show class map
#                 class_map = logits.argmax(dim=1, keepdim=True).float()  # (B,1,H,W)
#                 preds = class_map

#             # --- ground truth to (B,1,H,W) for saving ---
#             y_vis = y
#             if y_vis.ndim == 3:                          # (B,H,W) → (B,1,H,W)
#                 y_vis = y_vis.unsqueeze(1).float()
#             elif y_vis.ndim == 4 and y_vis.shape[1] > 1: # one-hot / multi-class
#                 y_vis = y_vis.argmax(dim=1, keepdim=True).float()

#             # torchvision expects float in [0,1]; 0/1 masks are fine.
#             torchvision.utils.save_image(preds.cpu(), str(out_dir / f"pred_{idx:04d}.png"))
#             torchvision.utils.save_image(y_vis.cpu(), str(out_dir / f"gt_{idx:04d}.png"))

#     model.train()

from pathlib import Path
import torch
import torchvision

def save_predictions_as_imgs(
    loader,
    model,
    folder="saved_images",
    device="cuda",
    thresholds=0.5,              # float or iterable of length C
):
    out_dir = Path(folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    was_training = model.training
    model.eval()

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            # allow loader to yield (x) or (x,y)
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                x, y = batch
            else:
                x, y = batch, None

            x = x.to(device, non_blocking=True)
            logits = model(x)                             # (B, C, H, W)
            if logits.ndim != 4:
                raise ValueError(f"Expected logits (B,C,H,W), got {logits.shape}")

            B, C, H, W = logits.shape
            probs = torch.sigmoid(logits)                 # (B, C, H, W)

            # build threshold tensor (1,C,1,1)
            if isinstance(thresholds, (int, float)):
                thr = torch.full((1, C, 1, 1), float(thresholds), device=probs.device, dtype=probs.dtype)
            else:
                thr_vec = torch.as_tensor(list(thresholds), device=probs.device, dtype=probs.dtype)
                assert thr_vec.numel() == C, f"thresholds must have length {C}"
                thr = thr_vec.view(1, C, 1, 1)

            masks = (probs > thr).float()                 # (B, C, H, W) in {0,1}

            # save per-channel predictions (and GT if available)
            for b in range(B):
                for c in range(C):
                    pred_path = out_dir / f"pred_{idx:04d}_b{b}_c{c}.png"
                    # save as single-channel image; save_image expects (1,H,W)
                    torchvision.utils.save_image(masks[b, c:c+1].cpu(), str(pred_path))

                    if y is not None:
                        y_b = y[b].to(device) if y.is_cuda or y.device != device else y[b]
                        # Normalize GT to (1,H,W) float in {0,1}
                        if y_b.ndim == 2:                         # (H,W) binary
                            y_vis = y_b.unsqueeze(0).float()
                        elif y_b.ndim == 3:
                            if y_b.shape[0] == C:                  # (C,H,W) multi-label one-hot
                                y_vis = y_b[c:c+1].float()
                            else:                                  # (1,H,W) or class-index (H,W)
                                if y_b.shape[0] == 1:
                                    y_vis = y_b.float()
                                else:
                                    # assume class-index map → convert to binary for class c
                                    y_vis = (y_b.argmax(dim=0) == c).float().unsqueeze(0)
                        else:
                            continue

                        gt_path = out_dir / f"gt_{idx:04d}_b{b}_c{c}.png"
                        torchvision.utils.save_image(y_vis.detach().cpu().clamp(0, 1), str(gt_path))

    if was_training:
        model.train()
