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

from model_sngp import UNET_SNGP


from utils_sngp import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    log_val_preds_tb,
    uncert_map_TB,
)


# Config / Arguments

p = argparse.ArgumentParser()
p.add_argument("--logdir", default=None)
p.add_argument("--epochs", type=int, default=10)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--batch_size", type=int, default=64)
p.add_argument("--img_h", type=int, default=160)
p.add_argument("--img_w", type=int, default=240)
p.add_argument("--ridge", type=float, default=1.0)
p.add_argument("--rff_dim", type=int, default=512)
p.add_argument("--reduction_dim", type=int, default=64)
p.add_argument("--chunk_pixels", type=int, default=8192)
p.add_argument("--load_ckpt", action="store_true")
args = p.parse_args()

RUN_DIR = Path(args.logdir or "runs/sngp_default")
RUN_DIR.mkdir(parents=True, exist_ok=True)
writer = SummaryWriter(RUN_DIR)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = True
NUM_WORKERS = 4

TRAIN_IMG_DIR = "data/train_images/"
TRAIN_MASK_DIR = "data/train_masks/"
VAL_IMG_DIR   = "data/val_images/"
VAL_MASK_DIR  = "data/val_masks/"


# Train utils

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
        y = y.float().unsqueeze(1).to(DEVICE, non_blocking=True)

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
        y = y.float().unsqueeze(1).to(DEVICE, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
            out = model(x)
            logits = _only_logits(out)
            loss = loss_fn(logits, y)
        total += float(loss.item()); n += 1
    model.train()
    return total / max(1, n)

# Main 

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
        num_classes=1,
        features=(64,128,256,512),
        reduction_dim=args.reduction_dim,
        rff_dim=args.rff_dim,
        kernel_scale=None,
        ridge=args.ridge,
        return_variance_train=False,
        return_variance_eval=True,
        chunk_pixels=args.chunk_pixels,
    ).to(DEVICE)

    loss_fn = nn.BCEWithLogitsLoss()
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
    dice0, acc0 = check_accuracy(val_loader, model, epoch0, device=DEVICE.type)
    writer.add_scalars("accuracies", {"dice": dice0, "accuracy": acc0}, epoch0)
    log_val_preds_tb(val_loader, model, writer, epoch0, DEVICE.type)

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
        dice, acc = check_accuracy(val_loader, model, ep, device=DEVICE.type)
        writer.add_scalars("accuracies", {"dice": dice, "accuracy": acc}, ep)

        if ep % 10 == 0:
            save_checkpoint({"state_dict": model.state_dict(), "optimizer": opt.state_dict()}, run_dir=RUN_DIR)

        log_val_preds_tb(val_loader, model, writer, ep, DEVICE.type)

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
    log_val_preds_tb(val_loader, model, writer, args.epochs + 1, DEVICE.type)
    uncert_map_TB(val_loader, model, writer, 10, DEVICE)

if __name__ == "__main__":
    main()
