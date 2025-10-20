# UNet + SNGP (distance-aware UNet + Random Fourier Features GP head)
# CHANGE: Do NOT materialize flattened RFFs Z during TRAINING. We now compute Z ONLY
# under torch.no_grad() (precision build / inference). Training forward uses z_map only.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nn_utils


# -----------------------------
# Norm/conv helpers
# -----------------------------
def sn_conv(in_ch, out_ch, k=3, s=1, p=1, bias=False):
    """Spectral-normalized Conv2d for encoder/decoder main convs."""
    return nn_utils.spectral_norm(
        nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=bias)
    )

def gn(num_channels, groups=8):
    """GroupNorm with a sensible default number of groups (safer than BN for small batches)."""
    g = min(groups, num_channels) or 1
    return nn.GroupNorm(g, num_channels)


# -----------------------------
# UNet blocks
# -----------------------------
class DoubleConv(nn.Module):
    """SN Conv -> GN -> ReLU -> SN Conv -> GN -> ReLU"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            sn_conv(in_channels, out_channels, 3, 1, 1, bias=False),
            gn(out_channels),
            nn.ReLU(inplace=True),
            sn_conv(out_channels, out_channels, 3, 1, 1, bias=False),
            gn(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class UpBlock(nn.Module):
    """
    Upsample (bilinear) -> 1x1 conv reduce (no SN) -> concat skip -> DoubleConv
    Avoids checkerboard artifacts vs ConvTranspose2d.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.reduce = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)  # deliberately no SN
        self.reduce_norm = gn(out_ch)
        # IMPORTANT: name must not collide with nn.Module.double()
        self.double_conv = DoubleConv(out_ch * 2, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = self.reduce_norm(self.reduce(x))
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.double_conv(x)


# -----------------------------
# Random Fourier Features (RFF) head
# -----------------------------
class RFF2d(nn.Module):
    """
    Per-pixel random Fourier features.
    TRAINING: use .map(feat) -> z_map (B, M, H, W)  [no gigantic flatten]
    EVAL/PRECISION: use .flatten(z_map) -> Z (BHW, M) under torch.no_grad()
    The projection is a frozen 1x1 conv (NO spectral norm) to avoid over-constraint.
    """
    def __init__(self, in_ch, rff_dim=512, scale=0.25):
        super().__init__()
        self.rff_dim = int(rff_dim)
        self.proj = nn.Conv2d(in_ch, self.rff_dim, kernel_size=1, bias=True)  # NO SN
        for p in self.proj.parameters():
            p.requires_grad_(False)
        with torch.no_grad():
            self.proj.weight.normal_(mean=0.0, std=float(scale))     # W ~ N(0, scale^2)
            self.proj.bias.uniform_(0.0, 2.0 * math.pi)              # b ~ U(0, 2π)
        # √(2/M) scaling for RBF-RFFs
        self.norm = (2.0 / self.rff_dim) ** 0.5

    def map(self, feat):  # -> z_map: (B, M, H, W)   (keeps grad for backbone)
        proj_map = self.proj(feat)                 # (B, M, H, W)
        z_map = self.norm * torch.cos(proj_map)    # (B, M, H, W)
        return z_map

    @staticmethod
    def flatten(z_map):  # -> Z: (BHW, M)   (call under no_grad)
        B, M, H, W = z_map.shape
        return z_map.permute(0, 2, 3, 1).reshape(-1, M)


# -----------------------------
# GP Head (multiclass) + Laplace precision (per-class diagonal-block approx)
# β is a 1x1 conv WITH spectral norm; precision is per-class MxM.
# -----------------------------
class GPHeadMulti(nn.Module):
    """
    Training: optimize β with standard CE/BCE on raw logits (no mean-field in the loss).
    After training:
      reset_precision(tau)
      loop train data (no grad): logits, Z = forward_logits_and_Z_eval(feat); accumulate_precision(Z, logits)
    Inference:
      variance_from_Z(Z) -> per-class logit variance; then mean-field softmax/sigmoid.
    """
    def __init__(self, in_ch, num_classes, rff_dim=512, rff_scale=0.25):
        super().__init__()
        self.num_classes = int(num_classes)
        self.rff = RFF2d(in_ch, rff_dim=rff_dim, scale=rff_scale)

        # Trainable linear map β: M -> K implemented as SN 1x1 conv
        self.beta_conv = nn_utils.spectral_norm(nn.Conv2d(rff_dim, self.num_classes, kernel_size=1, bias=False))

        # Per-class precision Σ_inv[k] (K, M, M)
        self.register_buffer("Sigma_inv", torch.eye(rff_dim).unsqueeze(0).repeat(self.num_classes, 1, 1))

    # ---- TRAINING FORWARD (no Z allocation) ----
    def forward_logits(self, feat):
        z_map = self.rff.map(feat)          # (B, M, H, W)  (grad flows to backbone)
        logits = self.beta_conv(z_map)      # (B, K, H, W)
        return logits

    # ---- EVAL / PRECISION BUILD (compute Z under no_grad) ----
    def forward_logits_and_Z_eval(self, feat):
        with torch.no_grad():
            z_map = self.rff.map(feat)              # (B, M, H, W)
            Z = self.rff.flatten(z_map)             # (BHW, M)
            logits = self.beta_conv(z_map)          # (B, K, H, W)
        return logits, Z

    def reset_precision(self, tau=1e-3):
        K, M, _ = self.Sigma_inv.shape
        eye = torch.eye(M, device=self.beta_conv.weight.device)
        self.Sigma_inv = eye.mul(float(tau)).unsqueeze(0).repeat(K, 1, 1)

    @torch.no_grad()
    def accumulate_precision(self, Z, logits):
        """
        Z: (N,M) with N = B*H*W  (provided from forward_logits_and_Z_eval)
        logits: (B,K,H,W) or (N,K)
        Σ_inv[k] += Z^T diag(p_k (1 - p_k)) Z   (softmax diag-block for K>1)
        """
        if logits.dim() == 4:
            B, K, H, W = logits.shape
            logits_vec = logits.permute(0, 2, 3, 1).reshape(-1, K)  # (N,K)
        else:
            logits_vec = logits  # (N,K)

        if self.num_classes == 1:
            p = torch.sigmoid(logits_vec)                      # (N,1)
            w = (p * (1 - p)).clamp_min(1e-6)                 # (N,1)
            Zw = Z * w.sqrt()                                  # (N,M)
            self.Sigma_inv[0] += Zw.T @ Zw
            return

        p = torch.softmax(logits_vec, dim=1)                   # (N,K)
        for k in range(self.num_classes):
            wk = (p[:, k] * (1.0 - p[:, k])).clamp_min(1e-6)  # (N,)
            Zw = Z * wk.sqrt().unsqueeze(1)                   # (N,M)
            self.Sigma_inv[k] += Zw.T @ Zw

    @torch.no_grad()
    def variance_from_Z(self, Z):
        """
        For each class k, σ_k^2 = diag( Z Σ_k Z^T ), with Σ_k = (Σ_inv[k])^{-1} (via Cholesky solve).
        Returns: sigma2 (N, K)
        """
        N = Z.shape[0]
        K, M, _ = self.Sigma_inv.shape
        sigma2 = Z.new_zeros((N, K))
        for k in range(K):
            Lk = torch.linalg.cholesky(self.Sigma_inv[k])      # (M,M)
            Vk = torch.cholesky_solve(Z.T, Lk)                 # (M,N)
            sigma2[:, k] = (Z * Vk.T).sum(dim=1)               # (N,)
        return sigma2  # (N,K)


# -----------------------------
# UNet with optional SNGP head
# -----------------------------
class UNET(nn.Module):
    """
    If sngp=True: replace final 1x1 classifier with SNGP GP head (N classes).
    If sngp=False: normal UNet with spectral-normalized final 1x1.
    """
    def __init__(self, in_channels=3, outchannels=2, features=(64, 128, 256, 512),
                 sngp=True, rff_dim=512, rff_scale=0.25):
        super().__init__()
        self.sngp = bool(sngp)
        self.outchannels = int(outchannels)

        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        ch = in_channels
        for f in features:
            self.downs.append(DoubleConv(ch, f))
            ch = f

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder with UpBlocks (no transposed convs)
        self.up_blocks = nn.ModuleList()
        dec_ch = features[-1] * 2
        for f in reversed(features):
            self.up_blocks.append(UpBlock(dec_ch, f))
            dec_ch = f

        if self.sngp:
            self.gp_head = GPHeadMulti(in_ch=features[0], num_classes=self.outchannels,
                                       rff_dim=rff_dim, rff_scale=rff_scale)
        else:
            self.final_conv = nn_utils.spectral_norm(nn.Conv2d(features[0], self.outchannels, kernel_size=1))

    # ---- shared encoder-decoder returning (B, features[0], H, W) ----
    def _decode(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skips = skips[::-1]
        for i, up in enumerate(self.up_blocks):
            x = up(x, skips[i])
        return x

    # ---- TRAINING forward: returns logits (no Z allocation) ----
    def forward(self, x):
        x = self._decode(x)
        if self.sngp:
            logits = self.gp_head.forward_logits(x)  # (B,K,H,W)
            return logits
        else:
            return self.final_conv(x)

    # ---- EVAL: logits + flattened RFFs Z for precision/variance ----
    def forward_with_features(self, x):
        x = self._decode(x)
        if not self.sngp:
            raise RuntimeError("forward_with_features requires sngp=True.")
        return self.gp_head.forward_logits_and_Z_eval(x)  # (B,K,H,W), (BHW,M)


# -----------------------------
# Inference helper: calibrated probs + variance (mean-field)
#  - K=1: sigmoid( μ / sqrt(1+λσ²) )
#  - K>1: softmax( μ_k / sqrt(1+λσ_k²) )
# -----------------------------
@torch.no_grad()
def predict_with_uncertainty(model: UNET, img: torch.Tensor, lambda_mf: float = math.pi/8.0):
    """
    Returns:
      probs  : (B,K,H,W) mean-field calibrated probabilities
      sigma2 : (B,K,H,W) predictive variance in logit space (per class)
      logits : (B,K,H,W) raw logits (μ)
    """
    assert model.sngp, "predict_with_uncertainty() requires a UNET with sngp=True."
    model.eval()
    logits, Z = model.forward_with_features(img)              # logits: (B,K,H,W), Z:(BHW,M)
    B, K, H, W = logits.shape
    sigma2_vec = model.gp_head.variance_from_Z(Z)             # (N,K), N=BHW
    sigma2 = sigma2_vec.view(B, H, W, K).permute(0, 3, 1, 2)  # (B,K,H,W)

    denom = torch.sqrt(1.0 + lambda_mf * sigma2)
    logits_adj = logits / denom

    if K == 1:
        probs = torch.sigmoid(logits_adj)
    else:
        probs = torch.softmax(logits_adj, dim=1)

    return probs, sigma2, logits


# -----------------------------
# Loss helpers (pick one in your train loop)
# -----------------------------
def multiclass_ce_dice(
    logits: torch.Tensor,               # (B, K, H, W)
    target: torch.Tensor,               # (B, H, W) in [0..K-1]
    class_weights: torch.Tensor | None = None,
    ignore_index: int | None = 255,
    dice_weight: float = 0.3,
    label_smooth: float = 0.05,
) -> torch.Tensor:
    ce = F.cross_entropy(
        logits, target,
        weight=class_weights,
        ignore_index=ignore_index,
        label_smoothing=label_smooth
    )
    probs = torch.softmax(logits, dim=1)
    B, K, H, W = probs.shape
    one_hot = torch.zeros(B, K, H, W, device=probs.device, dtype=probs.dtype)
    valid = (target != ignore_index) if ignore_index is not None else torch.ones_like(target, dtype=torch.bool)
    target_clamped = torch.where(valid, target, torch.zeros_like(target))
    one_hot.scatter_(1, target_clamped.unsqueeze(1), 1.0)
    one_hot = one_hot * valid.unsqueeze(1)
    inter = (probs * one_hot).sum(dim=(0,2,3))
    denom = (probs.pow(2) + one_hot.pow(2)).sum(dim=(0,2,3)).clamp_min(1e-6)
    dice = (2*inter) / denom
    dice_loss = 1.0 - dice.mean()
    return ce + dice_weight * dice_loss

class BCEDiceLoss(nn.Module):
    """Binary or multi-label: BCEWithLogits + soft Dice (per channel)."""
    def __init__(self, bce_weight=0.7, dice_weight=0.3, pos_weight=None, smooth=1e-6, reduction="mean"):
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.smooth = float(smooth)
        self.reduction = reduction
        if pos_weight is not None:
            pw = torch.as_tensor(pos_weight, dtype=torch.float32)
            self.register_buffer("pos_weight", pw)
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.to(logits.dtype)
        bce = F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=self.pos_weight, reduction=self.reduction
        )
        probs = torch.sigmoid(logits).clamp_(1e-7, 1 - 1e-7)
        inter = (probs * target).sum(dim=(0, 2, 3))
        denom = probs.sum(dim=(0, 2, 3)) + target.sum(dim=(0, 2, 3))
        dice = (2.0 * inter + self.smooth) / (denom + self.smooth)
        dice_loss = 1.0 - dice.mean()
        return self.bce_weight * bce + self.dice_weight * dice_loss


# -----------------------------
# Quick sanity test (K=2)
# -----------------------------
def _test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(2, 1, 160, 160, device=device)  # in_channels=1
    K = 2
    model = UNET(in_channels=1, outchannels=K, sngp=True, rff_dim=256, rff_scale=0.25).to(device)

    # TRAIN forward (no Z allocation)
    with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
        logits = model(x)
    print("logits:", logits.shape)  # (2,K,160,160)

    # Precision build (eval, no grad) — now computes Z safely
    model.eval()
    model.gp_head.reset_precision(tau=1e-3)
    with torch.no_grad():
        logits_eval, Z = model.forward_with_features(x)
        model.gp_head.accumulate_precision(Z, logits_eval)

    # Predict with uncertainty
    with torch.no_grad():
        p, s2, lg = predict_with_uncertainty(model, x)
        print("prob:", p.shape, "sigma2:", s2.shape, "logits:", lg.shape)


if __name__ == "__main__":
    _test()
