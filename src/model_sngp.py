# UNet + SNGP (Spectral-normalized features + Random Fourier Features GP head)
# Supports N output channels (binary or multiclass). Mean-field uncertainty per channel.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nn_utils


# -----------------------------
# Spectral-normalized conv helpers
# -----------------------------
def sn_conv(in_ch, out_ch, k=3, s=1, p=1, bias=False):
    return nn_utils.spectral_norm(
        nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=bias)
    )

def sn_tconv(in_ch, out_ch, k=2, s=2, bias=False):
    return nn_utils.spectral_norm(
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=k, stride=s, bias=bias)
    )


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
# GP Head (multiclass) + Laplace precision (per-class diagonal-block approx)
# -----------------------------
class GPHeadMulti(nn.Module):
    """
    - Trains beta with standard CE/BCE loss (included in model parameters).
    - After training, call:
        head.reset_precision(tau)
        then iterate over training data (no grad):
        logits, Z = model.forward_with_features(img)
        head.accumulate_precision(Z, logits)
      For K>1, this uses a diagonal-block Laplace approx:
        Σ_inv[k] += Z^T diag(p_k(1-p_k)) Z    (ignores cross-class blocks)
    - At inference, use variance_from_Z(Z) to get σ^2 per class (N,K) in logit space.
    """
    def __init__(self, in_ch, num_classes, rff_dim=128, rff_scale=1.0):
        super().__init__()
        self.num_classes = int(num_classes)
        self.rff = RFF2d(in_ch, rff_dim=rff_dim, scale=rff_scale)
        self.beta = nn.Parameter(torch.zeros(rff_dim, self.num_classes))  # (M,K)
        # Per-class precision matrices Σ_inv[k] (K, M, M)
        self.register_buffer("Sigma_inv", torch.eye(rff_dim).unsqueeze(0).repeat(self.num_classes, 1, 1))

    def forward_logits_and_z(self, feat):
        Z, (B, H, W) = self.rff(feat)            # Z: (BHW,M)
        logits_vec = Z @ self.beta               # (BHW,K)
        logits = logits_vec.view(B, H, W, self.num_classes).permute(0, 3, 1, 2)  # (B,K,H,W)
        return logits, Z

    def reset_precision(self, tau=1e-3):
        K, M, _ = self.Sigma_inv.shape
        self.Sigma_inv = torch.eye(M, device=self.beta.device).mul(float(tau)).unsqueeze(0).repeat(K, 1, 1)

    @torch.no_grad()
    def accumulate_precision(self, Z, logits):
        """
        Z: (N,M) flattened features for a batch (N = B*H*W)
        logits: (B,K,H,W) or (N,K) raw logits
        Update per-class precision: Σ_inv[k] += Z^T diag(p_k (1 - p_k)) Z
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

        p = torch.softmax(logits_vec, dim=1)  # (N,K)
        # Diagonal-block approx: ignore off-diagonal Fisher terms
        for k in range(self.num_classes):
            wk = (p[:, k] * (1.0 - p[:, k])).clamp_min(1e-6)  # (N,)
            Zw = Z * wk.sqrt().unsqueeze(1)                   # (N,M)
            self.Sigma_inv[k] += Zw.T @ Zw

    @torch.no_grad()
    def variance_from_Z(self, Z):
        """
        For each class k, σ_k^2 = diag( Z Σ_k Z^T ), Σ_k = (Σ_inv[k])^{-1}
        Returns: sigma2 (N, K)
        """
        N = Z.shape[0]
        K, M, _ = self.Sigma_inv.shape
        sigma2 = Z.new_zeros((N, K))
        for k in range(K):
            Lk = torch.linalg.cholesky(self.Sigma_inv[k])   # (M,M)
            Vk = torch.cholesky_solve(Z.T, Lk)              # (M,N)
            sigma2[:, k] = (Z * Vk.T).sum(dim=1)            # (N,)
        return sigma2  # (N,K)


# -----------------------------
# UNet with optional SNGP head (supports outchannels = K >= 1)
# -----------------------------
class UNET(nn.Module):
    """
    If sngp=True: replaces final 1x1 conv with SNGP GP head (N classes).
    If sngp=False: behaves like a normal UNet with 1x1 final conv.
    """
    def __init__(self, in_channels=3, outchannels=2, features=(64, 128, 256, 512),
                 sngp=True, rff_dim=128, rff_scale=1.0):
        super().__init__()
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
            self.gp_head = GPHeadMulti(in_ch=features[0], num_classes=self.outchannels,
                                       rff_dim=rff_dim, rff_scale=rff_scale)
        else:
            self.final_conv = nn.Conv2d(features[0], self.outchannels, kernel_size=1)

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
            logits, _ = self.gp_head.forward_logits_and_z(x)  # (B,K,H,W)
            return logits
        else:
            return self.final_conv(x)

    # ---- forward returning logits and flattened RFFs Z (use for precision build/inference) ----
    def forward_with_features(self, x):
        x = self._decode(x)
        if not self.sngp:
            raise RuntimeError("forward_with_features requires sngp=True.")
        return self.gp_head.forward_logits_and_z(x)  # (B,K,H,W), (BHW,M)


# -----------------------------
# Inference helper: prob + variance (mean-field)
#  - For K=1: sigmoid( μ / sqrt(1+λσ²) )
#  - For K>1: softmax( μ_k / sqrt(1+λσ_k²) ) across k
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
    logits, Z = model.forward_with_features(img)                  # logits: (B,K,H,W), Z:(BHW,M)
    B, K, H, W = logits.shape
    sigma2_vec = model.gp_head.variance_from_Z(Z)                 # (N,K), N=BHW
    sigma2 = sigma2_vec.view(B, H, W, K).permute(0, 3, 1, 2)      # (B,K,H,W)

    denom = torch.sqrt(1.0 + lambda_mf * sigma2)                  # (B,K,H,W)
    logits_adj = logits / denom

    if K == 1:
        probs = torch.sigmoid(logits_adj)
    else:
        probs = torch.softmax(logits_adj, dim=1)

    return probs, sigma2, logits


# -----------------------------
# Quick sanity test (K=2)
# -----------------------------
def _test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(2, 1, 160, 160, device=device)
    K = 2
    model = UNET(in_channels=1, outchannels=K, sngp=True, rff_dim=128).to(device)

    # Forward (logits only)
    with torch.no_grad():
        logits = model(x)
        print("logits:", logits.shape)  # (2,K,160,160)

    # Build precision with a fake pass over random data
    model.eval()
    model.gp_head.reset_precision(tau=1e-3)
    with torch.no_grad():
        logits, Z = model.forward_with_features(x)
        model.gp_head.accumulate_precision(Z, logits)

    # Predict with uncertainty
    with torch.no_grad():
        p, s2, lg = predict_with_uncertainty(model, x)
        print("prob:", p.shape, "sigma2:", s2.shape, "logits:", lg.shape)


if __name__ == "__main__":
    _test()
