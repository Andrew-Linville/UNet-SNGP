# train_sngp.py
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.tensorboard import SummaryWriter

from SNGP.model_sngp import UNET_SNGP

from custom_loss import BCEDiceLossSNGP

from SNGP.utils_sngp import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy_multilabel,
    log_val_preds_tb_multilabel,
    uncert_map_TB_multilabel,
)


# Config / Arguments

p = argparse.ArgumentParser()
p.add_argument("run_name", nargs="?", default=None, help="Optional run name (ignored by code; used by SLURM script).")
p.add_argument("--logdir", default=None)
p.add_argument("--epochs", type=int, default=400)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--batch_size", type=int, default=64)
p.add_argument("--img_h", type=int, default=240)
p.add_argument("--img_w", type=int, default=240)
p.add_argument("--ridge", type=float, default=1.0)
p.add_argument("--rff_dim", type=int, default=512)
p.add_argument("--reduction_dim", type=int, default=64)
p.add_argument("--chunk_pixels", type=int, default=8192)
p.add_argument("--load_ckpt", action="store_true")
p.add_argument("--loss_func", type=str, default="BCEDice")
args = p.parse_args()

RUN_DIR = Path(args.logdir or "runs/sngp_default")
RUN_DIR.mkdir(parents=True, exist_ok=True)
writer = SummaryWriter(RUN_DIR)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = True
NUM_WORKERS = 4

TRAIN_IMG_DIR = "data/train/imgs"
TRAIN_MASK_DIR = "data/train/masks"
VAL_IMG_DIR   = "data/val/imgs"
VAL_MASK_DIR  = "data/val/masks"


# Train utils

def _to_multilabel_targets(y):
    """
    Coerce y to float [B,2,H,W] for multi-label BCE.
    Accepts: [B,2,H,W] (one-hot or multi-hot), [B,1,2,H,W], [B,H,W] (binary -> broadcast),
             [B,1,H,W] (single channel -> broadcast or raise).
    """
    if y.ndim == 5 and y.shape[2] == 2:         # [B,1,2,H,W]
        y = y[:, 0, ...]                        # -> [B,2,H,W]
    elif y.ndim == 4 and y.shape[1] == 2:       # [B,2,H,W]
        pass
    elif y.ndim == 4 and y.shape[1] == 1:       # [B,1,H,W] -> duplicate to 2 chans (if both labels share same mask)
        y = y.repeat(1, 2, 1, 1)
    elif y.ndim == 3:                            # [B,H,W] -> duplicate to 2 chans
        y = y.unsqueeze(1).repeat(1, 2, 1, 1)
    else:
        raise ValueError(f"Unexpected target shape: {tuple(y.shape)}")
    return y.float()


def _only_logits(out):
    return out[0] if isinstance(out, (tuple, list)) else out

def set_variance(model, flag):
    if hasattr(model, "classifier"):
        model.classifier.enable_variance(bool(flag))

def train_epoch(loader, model, opt, loss_fn, scaler):
    model.train()
    set_variance(model, False)  # no variance during training
    running = 0.0
    loop = tqdm(loader)
    for x, y in loop:
        x = x.to(DEVICE, non_blocking=True)
        # y = y.float().unsqueeze(1).to(DEVICE, non_blocking=True)
        # y = _to_class_indices(y).to(DEVICE, non_blocking=True)
        y = _to_multilabel_targets(y).to(DEVICE, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
            out = model(x)
            logits = _only_logits(out)
            loss = loss_fn(logits, y)

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        running += float(loss.item())
        loop.set_postfix(loss=float(loss.item()))
    return running / max(1, len(loader))

@torch.no_grad()
def eval_loss(loader, model, loss_fn):
    model.eval()
    set_variance(model, False)  # keep plain logits for loss
    total, n = 0.0, 0
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        # y = y.float().unsqueeze(1).to(DEVICE, non_blocking=True)
        # y = _to_class_indices(y).to(DEVICE, non_blocking=True)
        y = _to_multilabel_targets(y).to(DEVICE, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
            out = model(x)
            logits = _only_logits(out)
            loss = loss_fn(logits, y)
        total += float(loss.item()); n += 1
    model.train()
    return total / max(1, n)

# Main 
def choose_loss(loss_type):
    
    if loss_type=="BCEDice":
        print(f"Using {loss_type} loss")
        loss_fn = BCEDiceLossSNGP()
        return loss_fn
    
    elif loss_type == "BCE":
        print(f"Using {loss_type} loss")
        return nn.BCEWithLogitsLoss() # Use for single class
    
    elif loss_type == "CE":
        print(f"Using {loss_type} loss")
        return nn.CrossEntropyLoss() # use for multi class
    
    else:
        print("Not valid loss function")
        return None
        
    
    

def main():
    train_tf = A.Compose([
        A.Resize(height=args.img_h, width=args.img_w),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(mean=[0.,0.,0.], std=[1.,1.,1.], max_pixel_value=255.0),
        ToTensorV2(),
    ])
    val_tf = A.Compose([
        A.Resize(height=args.img_h, width=args.img_w),
        A.Normalize(mean=[0.,0.,0.], std=[1.,1.,1.], max_pixel_value=255.0),
        ToTensorV2(),
    ])

    model = UNET_SNGP(
        in_channels=3,
        num_classes=2,
        features=(64,128,256,512),
        reduction_dim=args.reduction_dim,
        rff_dim=args.rff_dim,
        kernel_scale=None,
        ridge=args.ridge,
        return_variance_train=False,
        return_variance_eval=True,
        chunk_pixels=args.chunk_pixels,
    ).to(DEVICE)

    # Choose the loss function
    loss_fn = choose_loss(args.loss_func)
    
    opt = optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR, TRAIN_MASK_DIR,
        VAL_IMG_DIR, VAL_MASK_DIR,
        args.batch_size, train_tf, val_tf,
        NUM_WORKERS, PIN_MEMORY,
    )

    if args.load_ckpt:
        ckpt = torch.load(RUN_DIR / "checkpoint.pth.tar", map_location=DEVICE)
        load_checkpoint(ckpt, model)

    # Initial metrics
    epoch0 = 0
    set_variance(model, False)
    dice0, acc0 = check_accuracy_multilabel(val_loader, model, epoch0, device=DEVICE.type)
    writer.add_scalars("accuracies", {"dice": dice0, "accuracy": acc0}, epoch0)
    log_val_preds_tb_multilabel(val_loader, model, writer, epoch0, DEVICE.type, max_batches=2, samples_per_batch=10)

    tr0 = eval_loss(train_loader, model, loss_fn)
    vl0 = eval_loss(val_loader, model, loss_fn)
    writer.add_scalars("losses", {"training_loss": tr0, "val_loss": vl0}, epoch0)

    # Train
    for ep in range(1, args.epochs + 1):
        tr = train_epoch(train_loader, model, opt, loss_fn, scaler)
        vl = eval_loss(val_loader, model, loss_fn)
        writer.add_scalars("losses", {"training_loss": tr, "val_loss": vl}, ep)

        # metrics
        set_variance(model, False)
        dice, acc = check_accuracy_multilabel(val_loader, model, ep, device=DEVICE.type)
        writer.add_scalars("accuracies", {"dice": dice, "accuracy": acc}, ep)

        if ep % 10 == 0:
            save_checkpoint({"state_dict": model.state_dict(), "optimizer": opt.state_dict()}, run_dir=RUN_DIR)

        log_val_preds_tb_multilabel(val_loader, model, writer, ep, DEVICE.type, max_batches=2, samples_per_batch=10)

    # 
    # Final one-pass precision build
    # 
    def get_feats(batch_x):
        with torch.no_grad():
            return model.backbone_to_classifier_feats(batch_x)

    model.classifier.build_precision_exact(
        feat_loader=train_loader,
        device=DEVICE,
        use_amp=(DEVICE.type == "cuda"),
        get_backbone_feats=get_feats,
    )

    # 
    # Eval with variance
    # 
    model.eval()
    set_variance(model, True)  # returns (logits, var_map)

    # visualize uncertainty with your TB helper
    log_val_preds_tb_multilabel(val_loader, model, writer, args.epochs + 1, DEVICE.type, max_batches=8)
    uncert_map_TB_multilabel(val_loader, model, writer, 10, DEVICE)

if __name__ == "__main__":
    main()
