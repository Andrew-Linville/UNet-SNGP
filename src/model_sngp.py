# UNet + SNGP (Spectral-normalized features + Random Fourier Features GP head, binary segmentation)
# Drop this file into your project and import UNET, predict_with_uncertainty.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nn_utils


# -----------------------------
# Spectral-normalized conv helpers
# -----------------------------
def sn_conv(in_ch, out_ch, k=3, s=1, p=1, bias=False):
    return nn_utils.spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=bias))

def sn_tconv(in_ch, out_ch, k=2, s=2, bias=False):
    return nn_utils.spectral_norm(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=k, stride=s, bias=bias))


# -----------------------------
# UNet blocks
# -----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            sn_conv(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            sn_conv(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)


# -----------------------------
# Random Fourier Features (per pixel)
# -----------------------------
class RFF2d(nn.Module):
    """
    Maps per-pixel features of shape (B,C,H,W) -> (B*H*W, M) random Fourier features.
    W and b are fixed (buffers), not learned.
    """
    def __init__(self, in_ch, rff_dim=128, scale=1.0):
        super().__init__()
        self.rff_dim = int(rff_dim)
        self.register_buffer("W", torch.randn(in_ch, rff_dim) * float(scale))
        self.register_buffer("b", torch.rand(rff_dim) * 2 * math.pi)
        self.norm = (2.0 / rff_dim) ** 0.5

    def forward(self, feat):  # feat: (B,C,H,W)
        B, C, H, W = feat.shape
        x = feat.permute(0, 2, 3, 1).reshape(-1, C)   # (BHW,C)
        proj = x @ self.W + self.b                    # (BHW,M)
        z = self.norm * torch.cos(proj)               # (BHW,M)
        return z, (B, H, W)


# -----------------------------
# GP Head (binary) + Laplace precision
# -----------------------------
class GPHeadBinary(nn.Module):
    """
    - Trains beta with standard BCEWithLogits.
    - After training, call:
        head.reset_precision(tau)
        then iterate over training data (no grad):
        logits, Z = model.forward_with_features(img)
        head.accumulate_precision_binary(Z, logits.view(-1,1))
    - At inference, use variance_from_Z(Z) for σ^2 per pixel (BHW,1).
    """
    def __init__(self, in_ch, rff_dim=128, rff_scale=1.0):
        super().__init__()
        self.rff = RFF2d(in_ch, rff_dim=rff_dim, scale=rff_scale)
        self.beta = nn.Parameter(torch.zeros(rff_dim, 1))  # learned linear weights
        self.register_buffer("Sigma_inv", torch.eye(rff_dim))  # filled later by reset_precision/accumulate

    def forward_logits_and_z(self, feat):
        Z, (B, H, W) = self.rff(feat)           # Z: (BHW,M)
        logits_vec = Z @ self.beta              # (BHW,1)
        logits = logits_vec.view(B, H, W, 1).permute(0, 3, 1, 2)  # (B,1,H,W)
        return logits, Z

    def reset_precision(self, tau=1e-3):
        M = self.Sigma_inv.shape[0]
        self.Sigma_inv = torch.eye(M, device=self.beta.device) * float(tau)

    @torch.no_grad()
    def accumulate_precision_binary(self, Z, logits):
        """
        Z: (N,M) flattened features for a batch (N = B*H*W)
        logits: (N,1) raw logits
        Σ_inv += Z^T diag(p(1-p)) Z
        """
        p = torch.sigmoid(logits).flatten()                 # (N,)
        w = (p * (1 - p)).clamp_min(1e-6)                  # stability clamp
        Zw = Z * w.sqrt().unsqueeze(1)                     # (N,M)
        self.Sigma_inv += Zw.T @ Zw                        # rank-N update

    @torch.no_grad()
    def variance_from_Z(self, Z):
        """
        σ^2 = diag( Z Σ Z^T ), where Σ = (Σ_inv)^{-1}.
        Compute via Cholesky solves; no explicit inverse.
        Returns (N,1).
        """
        L = torch.linalg.cholesky(self.Sigma_inv)          # (M,M)
        V = torch.cholesky_solve(Z.T, L)                   # (M,N), solves Σ_inv * V = Z^T
        sigma2 = (Z * V.T).sum(dim=1, keepdim=True)        # (N,1)
        return sigma2


# -----------------------------
# UNet with optional SNGP head
# -----------------------------
class UNET(nn.Module):
    """
    If sngp=True: replaces final 1x1 conv with SNGP GP head (binary).
    If sngp=False: behaves like a normal UNet with 1x1 final conv.
    """
    def __init__(self, in_channels=3, outchannels=1, features=(64, 128, 256, 512),
                 sngp=True, rff_dim=128, rff_scale=1.0):
        super().__init__()
        assert outchannels == 1 or not sngp, "This code path implements SNGP for binary (outchannels=1)."

        self.sngp = bool(sngp)
        self.outchannels = int(outchannels)

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        ch = in_channels
        for f in features:
            self.downs.append(DoubleConv(ch, f))
            ch = f

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        for f in reversed(features):
            self.ups.append(sn_tconv(f * 2, f, 2, 2))
            self.ups.append(DoubleConv(f * 2, f))

        if self.sngp:
            self.gp_head = GPHeadBinary(in_ch=features[0], rff_dim=rff_dim, rff_scale=rff_scale)
        else:
            self.final_conv = nn.Conv2d(features[0], outchannels, kernel_size=1)

    # ---- shared decode pass returning (B, features[0], H, W) ----
    def _decode(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skips = skips[::-1]
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            s = skips[i // 2]
            if x.shape[-2:] != s.shape[-2:]:
                x = F.interpolate(x, size=s.shape[-2:], mode="bilinear", align_corners=False)
            x = self.ups[i + 1](torch.cat([s, x], dim=1))
        return x

    # ---- standard forward: returns logits ----
    def forward(self, x):
        x = self._decode(x)
        if self.sngp:
            logits, _ = self.gp_head.forward_logits_and_z(x)  # (B,1,H,W)
            return logits
        else:
            return self.final_conv(x)

    # ---- forward returning logits and flattened RFFs Z (use for precision build/inference) ----
    def forward_with_features(self, x):
        x = self._decode(x)
        if not self.sngp:
            raise RuntimeError("forward_with_features requires sngp=True.")
        return self.gp_head.forward_logits_and_z(x)  # (B,1,H,W), (BHW,M)


# -----------------------------
# Inference helper: prob + variance (binary, mean-field)
# -----------------------------
@torch.no_grad()
def predict_with_uncertainty(model: UNET, img: torch.Tensor):
    """
    Returns:
      probs  : (B,1,H,W) mean-field calibrated probabilities
      sigma2 : (B,1,H,W) predictive variance in logit space
      logits : (B,1,H,W) raw logits (μ)
    """
    assert model.sngp, "predict_with_uncertainty() requires a UNET with sngp=True."
    model.eval()
    logits, Z = model.forward_with_features(img)          # logits: (B,1,H,W), Z:(BHW,M)
    sigma2 = model.gp_head.variance_from_Z(Z)             # (BHW,1)
    B, _, H, W = logits.shape
    sigma2 = sigma2.view(B, 1, H, W)
    probs = torch.sigmoid(logits / torch.sqrt(1.0 + (math.pi / 8.0) * sigma2))
    return probs, sigma2, logits


# -----------------------------
# Quick sanity test
# -----------------------------
def _test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(2, 1, 160, 160, device=device)
    model = UNET(in_channels=1, outchannels=1, sngp=True, rff_dim=128).to(device)

    # Forward (logits only)
    with torch.no_grad():
        logits = model(x)
        print("logits:", logits.shape)  # (2,1,160,160)

    # Build precision with a fake pass over random data
    model.eval()
    model.gp_head.reset_precision(tau=1e-3)
    with torch.no_grad():
        logits, Z = model.forward_with_features(x)
        model.gp_head.accumulate_precision_binary(Z, logits.view(-1, 1))

    # Predict with uncertainty
    with torch.no_grad():
        p, s2, lg = predict_with_uncertainty(model, x)
        print("prob:", p.shape, "sigma2:", s2.shape, "logits:", lg.shape)


if __name__ == "__main__":
    _test()

