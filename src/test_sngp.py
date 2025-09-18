# save_as: run_multilabel_infer_and_save.py
# Inference on **validation set only** for UNet + SNGP (multilabel by default)
# - Loads SNGP-enabled UNet + checkpoint
# - (Optional) builds SNGP precision Σ_inv from a **separate calibration folder** (NOT validation)
# - Runs tiled inference on validation images only
# - Saves per-channel probability PNGs, binary masks, and variance (uncertainty) heatmaps
# - Each saved overlay figure shows: [val_image, mask_overlays, class_1_variance, class_2_variance]
#
# Usage:
#   python run_multilabel_infer_and_save.py
#   (adjust paths in the CONFIG block below)

from pathlib import Path
import os
import glob
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import OrderedDict

# ==== IMPORT YOUR MODEL ====
# Ensure this imports the **SNGP-enabled** UNET (sngp=True supported, gp_head present).
from model_sngp import UNET  # change if your class lives elsewhere


# ----------------------------
# CONFIG
# ----------------------------
class CONFIG:
    # Project root
    ROOT = Path(__file__).resolve().parent.parent

    # === Validation images ONLY ===
    VAL_IMAGE_DIR = ROOT / "data" / "val" / "imgs"

    # === Optional: separate calibration images for building Σ_inv ===
    CALIB_IMAGE_DIR = ROOT / "data" / "images"

    # Model checkpoint
    CKPT_PATH = ROOT / "my_checkpoint.pth.tar"

    # Outputs
    PLT_OUTPUT_DIR = ROOT / "val_mask_overlays"
    CHANNEL_MASK_DIR = ROOT / "val_channel_masks"
    VARIANCE_DIR = ROOT / "val_variance_maps"

    # Task / model
    MULTILABEL = True           # True: sigmoid per channel; False: multiclass softmax
    LAMBDA_MF = np.pi / 8.0     # mean-field correction factor
    TAU = 1e-3                  # Σ_inv prior precision

    # Tiling for validation inference
    TILE = 512
    STRIDE = 512

    # Build precision Σ_inv from CALIB_IMAGE_DIR (recommended if Σ_inv wasn't saved in the checkpoint)
    BUILD_PRECISION = True
    CALIB_MAX_IMAGES = 32       # None for all images in CALIB_IMAGE_DIR
    CALIB_TILE = 384
    CALIB_STRIDE = 384


CFG = CONFIG()


# ----------------------------
# Utilities
# ----------------------------
def load_all_images(folder_path: Path):
    exts = ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp"]
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(str(folder_path), ext)))
    return sorted(files)


def chunk_image(img, tile=512, stride=512):
    """Split HxWxC image into (tile x tile) patches with stride; anchor edges so patches are exact size."""
    H, W = img.shape[:2]
    chunks, coords = [], []
    ys = list(range(0, max(H - tile, 0) + 1, stride)) or [0]
    xs = list(range(0, max(W - tile, 0) + 1, stride)) or [0]
    for y0 in ys:
        for x0 in xs:
            y1, x1 = min(y0 + tile, H), min(x0 + tile, W)
            y0a = y1 - tile if (y1 - y0) < tile else y0
            x0a = x1 - tile if (x1 - x0) < tile else x0
            chunks.append(img[y0a:y0a + tile, x0a:x0a + tile])
            coords.append((y0a, x0a))
    return chunks, coords


def _to_tensor_rgb(patch, device):
    """HxWxC (uint8/float32) -> 1x3xhxw float32 in [0,1]. If gray, tile to 3 channels."""
    if patch.ndim == 2:
        patch = np.stack([patch, patch, patch], axis=-1)
    patch = patch[:, :, :3]
    t = torch.from_numpy(patch.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    return t


# ----------------------------
# SNGP: precision build & per-tile inference
# ----------------------------
@torch.no_grad()
def reset_precision(model, tau: float):
    assert hasattr(model, "gp_head"), "Model missing gp_head; ensure sngp=True."
    model.gp_head.reset_precision(tau=tau)


@torch.no_grad()
def accumulate_precision_from_logitsZ(model, logits, Z, multilabel: bool):
    """Update per-class Σ_inv: Z^T diag(w_k) Z, where w comes from sigmoid/softmax Fisher diag."""
    B, K, H, W = logits.shape
    logits_vec = logits.permute(0, 2, 3, 1).reshape(-1, K)  # (N,K)

    if multilabel or K == 1:
        p = torch.sigmoid(logits_vec)                       # (N,K)
        w = (p * (1 - p)).clamp_min(1e-6)
    else:
        p = torch.softmax(logits_vec, dim=1)                # (N,K)
        w = (p * (1 - p)).clamp_min(1e-6)

    for k in range(K):
        wk = w[:, k]                                        # (N,)
        Zw = Z * wk.sqrt().unsqueeze(1)                     # (N,M)
        model.gp_head.Sigma_inv[k] += Zw.T @ Zw


@torch.no_grad()
def sngp_infer_tile(model, patch_t, lambda_mf: float, multilabel: bool):
    """Return (mu, sigma2, probs) for one tile."""
    logits, Z = model.forward_with_features(patch_t)        # logits: 1xKxhxw
    sigma2_vec = model.gp_head.variance_from_Z(Z)           # (N,K)
    K = logits.shape[1]
    h, w = logits.shape[2], logits.shape[3]
    sigma2 = sigma2_vec.view(1, h, w, K).permute(0, 3, 1, 2).contiguous()  # 1xKxhxw
    mu = logits

    denom = torch.sqrt(1.0 + lambda_mf * sigma2)
    logits_adj = mu / denom
    probs = torch.sigmoid(logits_adj) if (multilabel or K == 1) else F.softmax(logits_adj, dim=1)
    return mu, sigma2, probs


@torch.no_grad()
def apply_model_sngp_tiled(img, model, device, tile=512, stride=512, lambda_mf=np.pi/8.0, multilabel=True):
    """
    Tiled SNGP inference with precision-weighted fusion across overlaps.
    Returns probs, var, logits as HxWxC numpy arrays.
    """
    H, W = img.shape[:2]
    chunks, coords = chunk_image(img, tile=tile, stride=stride)

    K = None
    sum_prec = None      # Σ (1/σ^2)
    sum_mu_prec = None   # Σ (μ/σ^2)

    for patch, (y0, x0) in zip(chunks, coords):
        patch_t = _to_tensor_rgb(patch, device)
        mu, sigma2, _ = sngp_infer_tile(model, patch_t, lambda_mf, multilabel)

        if K is None:
            K = mu.shape[1]
            sum_prec = torch.zeros(1, K, H, W, dtype=torch.float32, device=device)
            sum_mu_prec = torch.zeros(1, K, H, W, dtype=torch.float32, device=device)

        _, _, h, w = mu.shape
        prec = 1.0 / sigma2.clamp_min(1e-8)
        sum_prec[:, :, y0:y0 + h, x0:x0 + w] += prec
        sum_mu_prec[:, :, y0:y0 + h, x0:x0 + w] += (mu * prec)

    mu_post = sum_mu_prec / sum_prec.clamp_min(1e-8)
    var_post = 1.0 / sum_prec.clamp_min(1e-8)

    denom = torch.sqrt(1.0 + lambda_mf * var_post)
    logits_adj = mu_post / denom
    probs = torch.sigmoid(logits_adj) if (multilabel or K == 1) else F.softmax(logits_adj, dim=1)

    probs_np = probs.squeeze(0).permute(1, 2, 0).cpu().numpy()
    var_np = var_post.squeeze(0).permute(1, 2, 0).cpu().numpy()
    logits_np = mu_post.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return probs_np, var_np, logits_np


# ----------------------------
# Visualization + saving
# ----------------------------
def _make_colors(num_classes):
    base = np.array(
        [
            [255,   0,   0],
            [  0, 255,   0],
            [  0,   0, 255],
            [255, 255,   0],
            [255,   0, 255],
            [  0, 255, 255],
            [255, 128,   0],
            [128,   0, 255],
        ],
        dtype=np.uint8,
    )
    out = np.vstack([base for _ in range((num_classes + len(base) - 1) // len(base))])
    return out[:num_classes]


def colorize_mask(mask):
    """Multilabel mask/probabilities HxWxC -> color overlay with union of classes."""
    if mask.ndim == 2:
        mask = mask[..., None]
    H, W, C = mask.shape
    colors = _make_colors(C)
    bin_stack = (mask >= 0.5).astype(np.uint8) if mask.dtype != np.uint8 else mask
    color_mask = np.zeros((H, W, 3), dtype=np.float32)
    for c in range(C):
        cls_layer = bin_stack[..., c].astype(bool)
        color_mask[cls_layer] = np.maximum(color_mask[cls_layer], colors[c].astype(np.float32))
    return np.clip(color_mask, 0, 255).astype(np.uint8)


def overlay_image(img, mask_vis, alpha=0.25):
    img_u8 = img if img.dtype == np.uint8 else np.clip(img, 0, 255).astype(np.uint8)
    mask_present = (mask_vis.sum(axis=-1, keepdims=True) > 0).astype(np.float32)
    blended = img_u8.astype(np.float32) * (1 - alpha * mask_present) + mask_vis.astype(np.float32) * (alpha * mask_present)
    return np.clip(blended, 0, 255).astype(np.uint8)


def save_per_channel_masks(probs_hwc: np.ndarray, out_dir: Path, base_stem: str, thresholds=0.5, save_probs_png=True, save_probs_npy=False):
    out_dir.mkdir(parents=True, exist_ok=True)
    H, W, C = probs_hwc.shape
    thr = np.full((C,), float(thresholds), dtype=np.float32) if np.isscalar(thresholds) else np.asarray(list(thresholds), dtype=np.float32)
    assert thr.size == C, f"thresholds must have length C={C}"
    from PIL import Image as _Image  # avoid shadowing

    for c in range(C):
        prob = probs_hwc[..., c]
        mask = (prob >= thr[c]).astype(np.uint8)

        if save_probs_png:
            prob_u8 = np.clip(np.rint(prob * 255.0), 0, 255).astype(np.uint8)
            _Image.fromarray(prob_u8, mode="L").save(out_dir / f"{base_stem}_c{c}_prob.png")

        _Image.fromarray(mask * 255, mode="L").save(out_dir / f"{base_stem}_c{c}_mask.png")

        if save_probs_npy:
            np.save(out_dir / f"{base_stem}_c{c}_prob.npy", prob)


def save_per_channel_variance(var_hwc: np.ndarray, out_dir: Path, base_stem: str, clip_percentile: float = 99.0):
    out_dir.mkdir(parents=True, exist_ok=True)
    from PIL import Image as _Image
    vmax = max(np.percentile(var_hwc, clip_percentile), 1e-8)
    for c in range(var_hwc.shape[-1]):
        v = var_hwc[..., c]
        v_norm = np.clip(v / vmax, 0.0, 1.0)
        v_u8 = np.rint(v_norm * 255.0).astype(np.uint8)
        _Image.fromarray(v_u8, mode="L").save(out_dir / f"{base_stem}_c{c}_var.png")


def save_overlay_figure(img, probs_hwc, var_hwc, base_stem: str, out_dir: Path, thresholds=0.5):
    """
    Save a single figure with four panels: [val_image, mask_overlays, class_1_variance, class_2_variance].
    If there are more than 2 classes, the first two are shown. If only 1, the last panel repeats class 1.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    K = probs_hwc.shape[-1]
    # panel 1-2: image and mask overlay
    mask_vis = colorize_mask((probs_hwc >= thresholds).astype(np.uint8))
    over = overlay_image(img, mask_vis, alpha=0.25)

    # pick channels
    c0 = 0
    c1 = 1 if K > 1 else 0

    v0 = var_hwc[..., c0]
    v1 = var_hwc[..., c1]
    # robust scaling per class for visualization
    vmax0 = max(np.percentile(v0, 99.0), 1e-8)
    vmax1 = max(np.percentile(v1, 99.0), 1e-8)

    plt.figure(figsize=(22, 6))
    ax = plt.subplot(1, 4, 1); ax.imshow(img); ax.set_title("Validation image"); ax.axis("off")
    ax = plt.subplot(1, 4, 2); ax.imshow(over); ax.set_title("Predicted mask overlay"); ax.axis("off")
    ax = plt.subplot(1, 4, 3); im0 = ax.imshow(v0 / vmax0, cmap="inferno", vmin=0, vmax=1); ax.set_title("Class 1 variance"); ax.axis("off")
    plt.colorbar(im0, fraction=0.046, pad=0.04)
    ax = plt.subplot(1, 4, 4); im1 = ax.imshow(v1 / vmax1, cmap="inferno", vmin=0, vmax=1); ax.set_title("Class 2 variance"); ax.axis("off")
    plt.colorbar(im1, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(out_dir / f"{base_stem}_overlay.png", dpi=150)
    plt.close()


# ----------------------------
# Precision build from CALIB_IMAGE_DIR (NOT validation)
# ----------------------------
@torch.no_grad()
def build_precision_from_images(model, device, image_paths, tau=1e-3, multilabel=True, tile=384, stride=384, max_images=None):
    if not getattr(model, "sngp", False):
        return
    reset_precision(model, tau=tau)
    used = 0
    for p in image_paths:
        if max_images is not None and used >= max_images:
            break
        img = np.array(Image.open(p).convert("RGB"))
        chunks, _ = chunk_image(img, tile=tile, stride=stride)
        for patch in chunks:
            patch_t = _to_tensor_rgb(patch, device)
            logits, Z = model.forward_with_features(patch_t)
            accumulate_precision_from_logitsZ(model, logits, Z, multilabel=multilabel)
        used += 1
    print(f"[SNGP] Built precision from {used} calibration image(s).")


# ----------------------------
# Per-image validation run
# ----------------------------
def run_val_image(model, device, img_path, save_plots=True):
    img = np.array(Image.open(img_path).convert("RGB"))
    base_stem = Path(img_path).stem

    probs, var, logits = apply_model_sngp_tiled(
        img, model, device,
        tile=CFG.TILE, stride=CFG.STRIDE,
        lambda_mf=CFG.LAMBDA_MF, multilabel=CFG.MULTILABEL
    )

    # Save per-channel outputs
    per_img_dir = CFG.CHANNEL_MASK_DIR / base_stem
    save_per_channel_masks(probs_hwc=probs, out_dir=per_img_dir, base_stem=base_stem, thresholds=0.5, save_probs_png=True, save_probs_npy=False)

    # Save variance maps
    var_dir = CFG.VARIANCE_DIR / base_stem
    save_per_channel_variance(var_hwc=var, out_dir=var_dir, base_stem=base_stem, clip_percentile=99.0)

    # Save overlay figure (image, mask overlay, var class 1, var class 2)
    if save_plots:
        CFG.PLT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        save_overlay_figure(img, probs, var, base_stem, CFG.PLT_OUTPUT_DIR, thresholds=0.5)
        print(f"[VAL] Saved overlays for {base_stem}")


# ----------------------------
# Robust checkpoint loader
# ----------------------------
def load_ckpt_safely(path: Path, device, model: torch.nn.Module):
    """
    Loads a checkpoint saved either as:
      - raw state_dict
      - dict with 'state_dict'
      - DataParallel wrapped (keys prefixed with 'module.')
    Uses strict=False and prints missing/unexpected keys.
    """
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)  # newer PyTorch supports weights_only
    except TypeError:
        ckpt = torch.load(path, map_location=device)

    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

    # Strip 'module.' if present
    cleaned = OrderedDict()
    for k, v in state.items():
        new_k = k[7:] if k.startswith("module.") else k
        cleaned[new_k] = v

    incompat = model.load_state_dict(cleaned, strict=False)
    print(
        f"[CKPT] Loaded with {len(incompat.missing_keys)} missing "
        f"and {len(incompat.unexpected_keys)} unexpected keys."
    )
    if incompat.missing_keys:
        print("        Missing:", incompat.missing_keys[:8], "..." if len(incompat.missing_keys) > 8 else "")
    if incompat.unexpected_keys:
        print("        Unexpected:", incompat.unexpected_keys[:8], "..." if len(incompat.unexpected_keys) > 8 else "")


# ----------------------------
# Entry: VALIDATION ONLY
# ----------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Build/load SNGP model (match your trained channels)
    #    e.g., UNET(in_channels=3, outchannels=2, sngp=True, rff_dim=128)
    model = UNET(sngp=True).to(device)

    # 2) Load checkpoint safely (handles DP prefixes and various save formats)
    load_ckpt_safely(CFG.CKPT_PATH, device, model)

    # 3) (Optional) Build Σ_inv from CALIB_IMAGE_DIR (not validation)
    if getattr(model, "sngp", False) and CFG.BUILD_PRECISION:
        calib_images = load_all_images(CFG.CALIB_IMAGE_DIR)
        if len(calib_images) == 0:
            print(f"[WARN] No calibration images found in {CFG.CALIB_IMAGE_DIR}. Skipping Σ_inv build.")
        else:
            build_precision_from_images(
                model, device, calib_images,
                tau=CFG.TAU, multilabel=CFG.MULTILABEL,
                tile=CFG.CALIB_TILE, stride=CFG.CALIB_STRIDE,
                max_images=CFG.CALIB_MAX_IMAGES
            )

    # 4) VALIDATION ONLY: run on VAL_IMAGE_DIR
    val_images = load_all_images(CFG.VAL_IMAGE_DIR)
    if len(val_images) == 0:
        raise SystemExit(f"No validation images found in {CFG.VAL_IMAGE_DIR}. Populate it and retry.")

    for p in val_images:
        run_val_image(model, device, p, save_plots=True)

    print("[VAL] Done.")
