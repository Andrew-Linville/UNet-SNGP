import torch
import torchvision
from dataset.dataset import fillerDataset
from torch.utils.data import DataLoader
from pathlib import Path

# --- SNGP-safe helper: always get logits even if model returns (logits, extra) ---
def _only_logits(out):
    """Accepts either logits or (logits, extra) and returns logits."""
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


def save_checkpoint(state, run_dir=None, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    ckpt = run_dir / filename
    torch.save(state, ckpt)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
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
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = fillerDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader

# def check_accuracy(loader, model, epoch, device="cuda"):
#     num_correct = 0
#     num_pixels = 0
#     dice_score = 0
#     model.eval()

#     with torch.no_grad():
#         for x, y in loader:
#             x = x.to(device)
#             y = y.to(device).unsqueeze(1)
#             logits = _only_logits(model(x))
#             preds = torch.sigmoid(logits)
#             preds = (preds > 0.5).float()
#             num_correct += (preds == y).sum()
#             num_pixels += torch.numel(preds)
#             dice_score += (2 * (preds * y).sum()) / (
#                 (preds + y).sum() + 1e-8
#             )

#     print(f"Epoch: {epoch}")
#     print(
#         f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
#     )
#     print(f"Dice score: {dice_score/len(loader)}")
    
#     acc = num_correct/num_pixels
#     model.train()
    
#     batches = max(len(loader), 1)
#     return (dice_score / batches).item(), (num_correct.float() / num_pixels).item()



def save_predictions_as_imgs(
    loader, model, run_dir=None ,folder="saved_images/", device="cuda",
    save=True):
    out_dir = Path(run_dir) / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    
    preds_list = []
    
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            logits = _only_logits(model(x))
            preds = torch.sigmoid(logits)
            preds = (preds > 0.5).float()
        if save:
            torchvision.utils.save_image(
                preds, f"{out_dir}/pred_{idx}.png"
            )
            torchvision.utils.save_image(y.unsqueeze(1), f"{out_dir}{idx}.png")
        preds_list.append(preds)
        
    model.train()
    
    return preds_list

def TB_preds_vis(preds_list, writer, epoch):
          
    for i, t in enumerate(preds_list):
        t = t.detach().cpu()
        # pick the first item if it's a batch
        if t.ndim == 4:   # [B,C,H,W]
            t = t[0]
        if t.ndim == 3:   # [C,H,W] ok
            pass
        elif t.ndim == 2: # [H,W] -> add channel
            t = t.unsqueeze(0)
        else:
            raise ValueError("unexpected shape for image tensor")

        writer.add_image(f"val/pred_{i}", t.float(), epoch)

@torch.no_grad()
def log_val_preds_tb(
    loader,
    model,
    writer,
    epoch: int,
    device: str = "cuda",
    threshold: float = 0.5,
    max_batches: int = 2,      # how many val batches to visualize
    samples_per_batch: int = 4,# how many images per batch
    tag_prefix: str = "val/preds"
):
    model.eval()
    for b_idx, (x, y) in enumerate(loader):
        if b_idx >= max_batches:
            break

        x = x.to(device)                              # [B, C, H, W]
        logits = _only_logits(model(x))
        probs  = torch.sigmoid(logits)
        preds  = (probs > threshold).float()          # [B, 1 or C, H, W]

        # ensure single-channel for binary case
        if preds.ndim == 3:
            preds = preds.unsqueeze(1)                # [B,1,H,W]

        k = min(samples_per_batch, preds.size(0))
        # make a grid of k predictions (1ch each). If you want inputs/GT too, see note below.
        grid = torchvision.utils.make_grid(
            preds[:k].detach().cpu(), nrow=k, normalize=True
        )
        writer.add_image(f"{tag_prefix}/batch{b_idx}", grid, epoch)

    model.train()

@torch.no_grad()
def uncert_map_TB(
    loader,
    model,
    writer,
    epoch: int,
    device: str = "cuda",
    max_batches: int = 1,      # visualize one or a few batches
    tag_prefix: str = "val/uncertainty",
):
    """
    Logs per-pixel variance maps if the model returns (logits, var_map).
    Our SNGP head returns var_map with shape (B,1,H,W) when enabled.
    """
    model.eval()
    b_shown = 0
    for x, _ in loader:
        if b_shown >= max_batches:
            break
        x = x.to(device)
        out = model(x)

        if isinstance(out, (tuple, list)) and len(out) >= 2 and out[1] is not None:
            logits, var_map = out[0], out[1]          # var_map: (B,1,H,W)
            # Normalize per image for visualization
            vm = var_map.detach().float().cpu()
            B = vm.shape[0]
            for i in range(min(B, 4)):  # show up to 4
                v = vm[i]               # (1,H,W)
                v_norm = (v - v.min()) / (v.max() - v.min() + 1e-8)
                writer.add_image(f"{tag_prefix}/sample{i}", v_norm, epoch)
        else:
            # Model not returning variance (probably disabled) — nothing to log
            pass

        b_shown += 1
    model.train()



import torch
import torchvision

def _to_multilabel_targets(y, C: int):
    """
    Coerce y to float [B,C,H,W] for multi-label BCE.
    Accepts: [B,C,H,W], [B,1,C,H,W], [B,1,H,W], [B,H,W].
    If single-channel/binary is provided, it is duplicated across C channels.
    """
    if y.ndim == 5 and y.shape[2] == C:      # [B,1,C,H,W]
        y = y[:, 0, ...]                     # -> [B,C,H,W]
    elif y.ndim == 4:
        if y.shape[1] == C:                  # [B,C,H,W]
            pass
        elif y.shape[1] == 1:                # [B,1,H,W] -> duplicate
            y = y.repeat(1, C, 1, 1)
        else:
            raise ValueError(f"Unexpected target shape {tuple(y.shape)}")
    elif y.ndim == 3:                         # [B,H,W] -> duplicate
        y = y.unsqueeze(1).repeat(1, C, 1, 1)
    else:
        raise ValueError(f"Unexpected target shape {tuple(y.shape)}")
    return y.float()

@torch.no_grad()
def check_accuracy_multilabel(loader, model, epoch, device="cuda", thresholds=None):
    """
    Returns (mean_dice, mean_accuracy, dice_per_class(list), acc_per_class(list)).
    Thresholds: None or list/1D tensor length C with values in [0,1].
    """
    model.eval()

    # Accumulators across all batches
    tp = None; fp = None; fn = None; tn = None

    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        logits = logits[0] if isinstance(logits, (tuple, list)) else logits  # [B,C,H,W]
        B, C, H, W = logits.shape

        y = _to_multilabel_targets(y, C).to(device)       # [B,C,H,W]
        probs = torch.sigmoid(logits)

        if thresholds is None:
            thr = torch.full((C,), 0.5, device=device)
        else:
            thr = torch.tensor(thresholds, dtype=protorch.float32, device=device)  # type: ignore
            if thr.numel() != C:
                raise ValueError(f"thresholds length {thr.numel()} != C={C}")
        thr = thr.view(1, C, 1, 1)

        preds = (probs > thr).float()

        # Confusion terms per class
        _tp = (preds * y).sum(dim=(0, 2, 3))
        _fp = (preds * (1 - y)).sum(dim=(0, 2, 3))
        _fn = ((1 - preds) * y).sum(dim=(0, 2, 3))
        _tn = ((1 - preds) * (1 - y)).sum(dim=(0, 2, 3))

        if tp is None:
            tp, fp, fn, tn = _tp, _fp, _fn, _tn
        else:
            tp += _tp; fp += _fp; fn += _fn; tn += _tn

    # Per-class metrics
    eps = 1e-8
    dice_c = (2 * tp) / (2 * tp + fp + fn + eps)
    acc_c  = (tp + tn) / (tp + fp + fn + tn + eps)

    mean_dice = dice_c.mean().item()
    mean_acc  = acc_c.mean().item()

    print(f"Epoch: {epoch}")
    print(f"Acc per-class: {[float(a) for a in acc_c.tolist()]}, mean={mean_acc:.4f}")
    print(f"Dice per-class: {[float(d) for d in dice_c.tolist()]}, mean={mean_dice:.4f}")

    model.train()
    return mean_dice, mean_acc


@torch.no_grad()
def log_val_preds_tb_multilabel(
    loader,
    model,
    writer,
    epoch: int,
    device: str = "cuda",
    thresholds=None,           # None or list/1D tensor length C
    max_batches: int = 2,
    samples_per_batch: int = 4,
    class_names=None,          # None or list of length C
    tag_prefix: str = "val/preds",
):
    model.eval()
    batches_shown = 0

    for x, y in loader:
        if batches_shown >= max_batches:
            break

        x = x.to(device)
        out = model(x)
        logits = out[0] if isinstance(out, (tuple, list)) else out   # [B,C,H,W]
        B, C, H, W = logits.shape

        probs = torch.sigmoid(logits)

        if thresholds is None:
            thr = torch.full((C,), 0.5, device=device)
        else:
            thr = torch.tensor(thresholds, dtype=torch.float32, device=device)
            if thr.numel() != C:
                raise ValueError(f"thresholds length {thr.numel()} != C={C}")
        thr = thr.view(1, C, 1, 1)

        preds = (probs > thr).float()   # [B,C,H,W]

        k = min(samples_per_batch, B)
        for c in range(C):
            name = class_names[c] if (class_names and c < len(class_names)) else f"class{c}"
            grid = torchvision.utils.make_grid(
                preds[:k, c:c+1].detach().cpu(), nrow=k, normalize=True
            )
            writer.add_image(f"{tag_prefix}/{name}", grid, epoch)

        batches_shown += 1

    model.train()


@torch.no_grad()
def uncert_map_TB_multilabel(
    loader,
    model,
    writer,
    epoch: int,
    device: str = "cuda",
    max_batches: int = 1,
    tag_prefix: str = "val/uncertainty",
):
    """
    Logs per-pixel variance maps if the model returns (logits, var_map).
    Expects var_map shape [B,1,H,W] from your SNGP head.
    """
    model.eval()
    shown = 0
    for x, _ in loader:
        if shown >= max_batches:
            break
        x = x.to(device)
        out = model(x)
        if isinstance(out, (tuple, list)) and len(out) >= 2 and out[1] is not None:
            var_map = out[1]  # [B,1,H,W]
            vm = var_map.detach().float().cpu()
            for i in range(min(vm.shape[0], 4)):
                v = vm[i]  # [1,H,W]
                v_norm = (v - v.min()) / (v.max() - v.min() + 1e-8)
                writer.add_image(f"{tag_prefix}/sample{i}", v_norm, epoch)
        shown += 1
    model.train()
