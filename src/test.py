# save_as: run_multilabel_infer_and_save.py

from pathlib import Path
import os
import glob
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from model import UNET  # adjust import if your UNet lives elsewhere


# ----------------------------
# Tiling / stitching utilities
# ----------------------------
def chunk_image(img, tile=512, stride=512):
    """
    Split an HxWxC image into overlapping tiles of size (tile x tile),
    stepping by 'stride'. Anchors to image edges so all patches are tile-sized.
    Returns:
      chunks: list of HxWxC patches
      coords: list of (y0, x0) upper-left coordinates for each patch in the original image
    """
    H, W = img.shape[:2]
    chunks, coords = [], []
    ys = list(range(0, max(H - tile, 0) + 1, stride)) or [0]
    xs = list(range(0, max(W - tile, 0) + 1, stride)) or [0]
    for y0 in ys:
        for x0 in xs:
            y1, x1 = min(y0 + tile, H), min(x0 + tile, W)
            # anchor to ensure exact tile size
            y0_adj = y1 - tile if (y1 - y0) < tile else y0
            x0_adj = x1 - tile if (x1 - x0) < tile else x0
            patch = img[y0_adj:y0_adj + tile, x0_adj:x0_adj + tile]
            chunks.append(patch)
            coords.append((y0_adj, x0_adj))
    return chunks, coords


def apply_model(
    img,
    model,
    device=None,
    mode: str = "multilabel",   # "multilabel" => sigmoid per-channel; "multiclass" => softmax+argmax
    tile: int = 512,
    stride: int = 512,
):
    """
    img: HxWxC uint8/float32 array
    returns:
      - multilabel: HxWxC float probabilities in [0,1] (after sigmoid)
      - multiclass: HxW uint8 class map (softmax+argmax)
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    # chunk with provided tile/stride
    img_chunks, coords = chunk_image(img, tile=tile, stride=stride)

    H, W = img.shape[:2]
    out_accum = None  # will create once we know C
    weight = torch.zeros(1, 1, H, W, dtype=torch.float32, device=device)

    with torch.no_grad():
        for (patch, (y0, x0)) in zip(img_chunks, coords):
            # ensure 3 channels
            if patch.ndim == 2:
                patch = np.stack([patch, patch, patch], axis=-1)
            patch = patch[:, :, :3]

            # to tensor NCHW in [0,1]
            patch_t = (
                torch.from_numpy(patch.astype(np.float32) / 255.0)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(device)
            )

            logits = model(patch_t)  # 1 x C x h x w

            if out_accum is None:
                C = logits.shape[1]
                out_accum = torch.zeros(1, C, H, W, dtype=torch.float32, device=device)

            if mode == "multiclass":
                probs = F.softmax(logits, dim=1)      # 1 x C x h x w
            else:
                probs = torch.sigmoid(logits)         # 1 x C x h x w

            _, _, h, w = probs.shape
            out_accum[:, :, y0:y0 + h, x0:x0 + w] += probs
            weight[:, :, y0:y0 + h, x0:x0 + w] += 1.0

    probs_full = out_accum / weight.clamp_min(1e-6)

    if mode == "multiclass":
        pred = (
            probs_full.argmax(dim=1)
            .squeeze(0)
            .detach()
            .cpu()
            .numpy()
            .astype(np.uint8)
        )  # HxW
        return pred
    else:
        probs_np = (
            probs_full.squeeze(0)
            .permute(1, 2, 0)
            .detach()
            .cpu()
            .numpy()
        )  # HxWxC
        return probs_np


# ----------------------------
# Visualization helpers
# ----------------------------
def _make_colors(num_classes):
    base = np.array(
        [
            [255,   0,   0],  # red
            [  0, 255,   0],  # green
            [  0,   0, 255],  # blue
            [255, 255,   0],  # yellow
            [255,   0, 255],  # magenta
            [  0, 255, 255],  # cyan
            [255, 128,   0],  # orange
            [128,   0, 255],  # violet
        ],
        dtype=np.uint8,
    )
    out = np.vstack([base for _ in range((num_classes + len(base) - 1) // len(base))])
    return out[:num_classes]


def colorize_mask(mask, mode="multilabel"):
    """
    Returns a color HxWx3 visualization of the mask.
    - multilabel: mask can be probabilities HxWxC or binary (0/1). We'll show the union.
    - multiclass: mask is HxW int labels.
    """
    if mode == "multiclass":
        labels = mask.astype(np.int32)  # HxW
        C = int(labels.max()) + 1 if labels.size else 1
        colors = _make_colors(max(C, 1))
        color_mask = colors[np.clip(labels, 0, colors.shape[0] - 1)]
        return color_mask.astype(np.uint8)

    # multilabel
    if mask.ndim == 2:  # HxW (single channel)
        mask = mask[..., None]
    H, W, C = mask.shape
    colors = _make_colors(C)
    # If not binary, threshold at 0.5 just for visualization here
    bin_stack = (mask >= 0.5).astype(np.uint8) if mask.dtype != np.uint8 else mask

    color_mask = np.zeros((H, W, 3), dtype=np.float32)
    for c in range(C):
        cls_layer = bin_stack[..., c].astype(bool)
        color = colors[c].astype(np.float32)
        # Max-blend (multiple classes at same pixel → brightest of them)
        color_mask[cls_layer] = np.maximum(color_mask[cls_layer], color)
    return np.clip(color_mask, 0, 255).astype(np.uint8)


def overlay_image(img, mask_vis, alpha=0.2):
    """
    Blend colorized mask (mask_vis) onto img. Both HxWx3 uint8.
    """
    img_u8 = img if img.dtype == np.uint8 else np.clip(img, 0, 255).astype(np.uint8)
    mask_present = (mask_vis.sum(axis=-1, keepdims=True) > 0).astype(np.float32)
    blended = (
        img_u8.astype(np.float32) * (1 - alpha * mask_present)
        + mask_vis.astype(np.float32) * (alpha * mask_present)
    )
    return np.clip(blended, 0, 255).astype(np.uint8)


def show_image_mask_overlay(img, out, mode="multilabel", thresh=0.5, title=None, show_plot=True):
    """
    img: HxWx3 uint8
    out:
      - multilabel: HxWxC probabilities or binary
      - multiclass: HxW uint8 labels
    """
    if mode == "multiclass":
        mask_for_vis = out  # HxW labels
    else:
        if out.ndim == 3 and out.dtype != np.uint8:
            mask_for_vis = (out >= thresh).astype(np.uint8)
        else:
            mask_for_vis = out

    mask_vis = colorize_mask(mask_for_vis, mode=mode)
    over = overlay_image(img, mask_vis, alpha=0.2)

    # If multilabel with single class (HxWx1), show the single-channel mask in grayscale
    single_channel = (mode == "multilabel" and mask_for_vis.ndim == 3 and mask_for_vis.shape[-1] == 1)
    mask_panel = mask_for_vis.squeeze(-1) if single_channel else mask_vis

    plt.figure(figsize=(24, 8))
    if title:
        plt.suptitle(title)
    # 1) original
    plt.subplot(1, 3, 1); plt.imshow(img); plt.axis("off"); plt.title("Image")
    # 2) mask
    plt.subplot(1, 3, 2)
    if single_channel:
        plt.imshow(mask_panel, cmap="gray", vmin=0, vmax=1)
        plt.title("Mask (binary)")
    elif mode == "multiclass":
        plt.imshow(mask_panel); plt.title("Mask (labels)")
    else:
        plt.imshow(mask_panel); plt.title("Mask (colorized)")
    plt.axis("off")
    # 3) overlay
    plt.subplot(1, 3, 3); plt.imshow(over); plt.axis("off"); plt.title("Overlay")
    plt.tight_layout()
    if show_plot:
        plt.show()


# ----------------------------
# I/O helpers
# ----------------------------
def load_all_images(folder_path):
    """Return sorted list of image file paths in a folder."""
    exts = ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp"]
    image_files = []
    for ext in exts:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
    return sorted(image_files)


def save_per_channel_masks(
    probs_hwc: np.ndarray,          # HxWxC, float [0,1]
    out_dir: Path,
    base_stem: str,
    thresholds=0.5,                 # float or iterable length C
    save_probs_png=True,
    save_probs_npy=False
):
    """
    Save each channel's sigmoid probability and its thresholded binary mask.

    Writes:
      <stem>_c{k}_prob.png    # grayscale PNG of probabilities
      <stem>_c{k}_mask.png    # binary mask (0/255)
      (optional) <stem>_c{k}_prob.npy
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    H, W, C = probs_hwc.shape

    # thresholds -> (C,) array
    if np.isscalar(thresholds):
        thr = np.full((C,), float(thresholds), dtype=np.float32)
    else:
        thr = np.asarray(list(thresholds), dtype=np.float32)
        assert thr.size == C, f"thresholds must have length C={C}"

    for c in range(C):
        prob = probs_hwc[..., c]
        mask = (prob >= thr[c]).astype(np.uint8)

        if save_probs_png:
            prob_u8 = np.clip(np.rint(prob * 255.0), 0, 255).astype(np.uint8)
            Image.fromarray(prob_u8, mode="L").save(out_dir / f"{base_stem}_c{c}_prob.png")

        # binary mask as PNG (0/255)
        Image.fromarray(mask * 255, mode="L").save(out_dir / f"{base_stem}_c{c}_mask.png")

        if save_probs_npy:
            np.save(out_dir / f"{base_stem}_c{c}_prob.npy", prob)


def run_img_overlay(model, device, img_name, save_plt=False):
    img_path = IMAGE_DIR / img_name
    img = np.array(Image.open(img_path).convert("RGB"))

    # run model -> per-channel probabilities (sigmoid)
    probs = apply_model(
        img,
        model,
        device=device,
        mode="multilabel",
        tile=512,
        stride=512
    )  # HxWxC

    # save each channel (prob + binary)
    per_img_dir = CHANNEL_MASK_DIR / Path(img_name).stem
    save_per_channel_masks(
        probs_hwc=probs,
        out_dir=per_img_dir,
        base_stem=Path(img_name).stem,
        thresholds=0.5,             # or list like [0.4, 0.6, ...] per channel
        save_probs_png=True,
        save_probs_npy=False
    )

    # optional overlay figure for quick inspection
    bin_masks = (probs >= 0.5).astype(np.uint8)  # for display only
    show_image_mask_overlay(
        img, bin_masks, mode="multilabel", thresh=0.5,
        title=img_name, show_plot=False
    )

    if save_plt:
        PLT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(PLT_OUTPUT_DIR / img_name)
        print(f"Saved overlay: {img_name}")


# ----------------------------
# Main script
# ----------------------------
ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = ROOT / "data" / "images"
PLT_OUTPUT_DIR = ROOT / "mask_overlays"
CHANNEL_MASK_DIR = ROOT / "channel_masks"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# build/load your model (adjust params to match your training)
model = UNET().to(device)

# load checkpoint (handles dict with 'state_dict' or raw state_dict)
ckpt_path = ROOT / "saved_models" / "UNet_9_11_b16e400.pth.tar"
ckpt = torch.load(ckpt_path, map_location=device)
state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
model.load_state_dict(state, strict=False)

# gather images and run
img_file_names = [Path(p).name for p in load_all_images(IMAGE_DIR)]
for name in img_file_names:
    run_img_overlay(model, device, name, save_plt=True)
