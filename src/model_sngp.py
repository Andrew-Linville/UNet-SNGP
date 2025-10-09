# UNet + SNGP (Spectral-normalized features + Random Fourier Features GP head)
# Supports N output channels (binary or multiclass). Mean-field uncertainty per channel.
# Spectral normalization is applied to ALL conv / transpose-conv layers, including the GP head.

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

def sn_conv1x1(in_ch, out_ch, bias=False):
    return sn_conv(in_ch, out_ch, k=1, s=1, p=0, bias=bias)

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
# Random Fourier Features (per pixel) with spectral-normalized 1x1 projection
# -----------------------------
class RFF2d(nn.Module):
    """
    Maps per-pixel features (B,C,H,W) -> random Fourier features Z (BHW, M).
    The projection to M channels is a *spectral-normalized* 1x1 conv, frozen after init.
    """
    def __init__(self, in_ch, rff_dim=128, scale=1.0):
        super().__init__()
        self.rff_dim = int(rff_dim)
        # SN 1x1 conv used as random projection (frozen weights)
        self.proj = sn_conv1x1(in_ch, self.rff_dim, bias=True)
        for p in self.proj.parameters():
            p.requires_grad_(False)
        with torch.no_grad():
            # Initialize like random features: W ~ N(0, scale^2), b ~ U(0, 2π)
            self.proj.weight.normal_(mean=0.0, std=float(scale))
            self.proj.bias.uniform_(0.0, 2.0 * math.pi)
        # √(2/M) scaling is standard for RFF of RBF kernels
        self.norm = (2.0 / self.rff_dim) ** 0.5

    def forward(self, feat):  # feat: (B,C,H,W)
        B, C, H, W = feat.shape
        proj_map = self.proj(feat)                 # (B, M, H, W), spectral-normalized
        z_map = self.norm * torch.cos(proj_map)    # (B, M, H, W)
        # Flatten to (BHW, M) for GP math
        Z = z_map.permute(0, 2, 3, 1).reshape(-1, self.rff_dim)
        return Z, (B, H, W)


# -----------------------------
# GP Head (multiclass) + Laplace precision (per-class diagonal-block approx)
# All convs in this head are spectral-normalized.
# -----------------------------
class GPHeadMulti(nn.Module):
    """
    - Train beta with standard CE/BCE loss (beta conv is trainable).
    - After training, build precision:
        head.reset_precision(tau)
        for batches (no grad):
            logits, Z = model.forward_with_features(img)
            head.accumulate_precision(Z, logits)
      For K>1, diagonal-block Laplace approx:
        Σ_inv[k] += Z^T diag(p_k(1-p_k)) Z   (ignores off-diagonal Fisher blocks)
    - Inference: variance_from_Z(Z) returns per-class logit variance.
    """
    def __init__(self, in_ch, num_classes, rff_dim=128, rff_scale=1.0):
        super().__init__()
        self.num_classes = int(num_classes)
        self.rff = RFF2d(in_ch, rff_dim=rff_dim, scale=rff_scale)

        # Trainable linear map β: M -> K implemented as SN 1x1 conv
        # We'll feed it feature maps implicitly via shapes; here we operate on flattened Z,
        # so keep β as a parameter matrix but also provide an SN 1x1 conv for the logits route if needed.
        # For simplicity and full SN coverage, we implement β as an SN conv and also keep a weight view.
        self.beta_conv = nn_utils.spectral_norm(nn.Conv2d(rff_dim, self.num_classes, kernel_size=1, bias=False))

        # Maintain a parameter view β (M, K) tied to conv weights for math with flattened Z
        # Conv2d weight shape: (K, M, 1, 1)  -> β = (M, K)
        self._tie_beta_parameter()

        # Per-class precision matrices Σ_inv[k] (K, M, M)
        self.register_buffer("Sigma_inv", torch.eye(rff_dim).unsqueeze(0).repeat(self.num_classes, 1, 1))

    def _tie_beta_parameter(self):
        # Create a property-like view to access β as (M, K) <-> conv weight (K, M, 1, 1)
        self.beta = self.beta_conv.weight  # (K, M, 1, 1)

    def forward_logits_and_z(self, feat):
        """
        Returns:
          logits: (B, K, H, W) using spectral-normalized β conv
          Z     : (BHW, M) flattened RFFs for precision/variance ops
        """
        Z, (B, H, W) = self.rff(feat)            # Z: (BHW, M)
        # Also produce logits via the SN β conv for the (B,K,H,W) path:
        # Reuse the same RFF map but unflattened: recompute quickly to keep code clean
        with torch.no_grad():
            proj_map = self.rff.proj(feat)                      # (B,M,H,W)
            z_map = self.rff.norm * torch.cos(proj_map)         # (B,M,H,W)
        logits = self.beta_conv(z_map)                          # (B,K,H,W)
        return logits, Z

    def reset_precision(self, tau=1e-3):
        K, M, _ = self.Sigma_inv.shape
        eye = torch.eye(M, device=self.beta_conv.weight.device)
        self.Sigma_inv = eye.mul(float(tau)).unsqueeze(0).repeat(K, 1, 1)

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
            # Binary (single-logit)
            p = torch.sigmoid(logits_vec)                      # (N,1)
            w = (p * (1 - p)).clamp_min(1e-6)                 # (N,1)
            Zw = Z * w.sqrt()                                  # (N,M)
            self.Sigma_inv[0] += Zw.T @ Zw
            return

        # Multiclass (softmax): diagonal-block Fisher approx
        p = torch.softmax(logits_vec, dim=1)                   # (N,K)
        for k in range(self.num_classes):
            wk = (p[:, k] * (1.0 - p[:, k])).clamp_min(1e-6)  # (N,)
            Zw = Z * wk.sqrt().unsqueeze(1)                   # (N,M)
            self.Sigma_inv[k] += Zw.T @ Zw

    @torch.no_grad()
    def variance_from_Z(self, Z):
        """
        For each class k, σ_k^2 = diag( Z Σ_k Z^T ), where Σ_k = (Σ_inv[k])^{-1}
        Returns: sigma2 (N, K)
        """
        N = Z.shape[0]
        K, M, _ = self.Sigma_inv.shape
        sigma2 = Z.new_zeros((N, K))
        for k in range(K):
            # Solve Σ_inv[k] * X = Z^T via Cholesky → X = Σ * Z^T  (no explicit inverse)
            Lk = torch.linalg.cholesky(self.Sigma_inv[k])      # (M,M)
            Vk = torch.cholesky_solve(Z.T, Lk)                 # (M,N)
            sigma2[:, k] = (Z * Vk.T).sum(dim=1)               # (N,)
        return sigma2  # (N,K)


# -----------------------------
# UNet with optional SNGP head (supports outchannels = K >= 1)
# All convs are spectral-normalized.
# -----------------------------
class UNET(nn.Module):
    """
    If sngp=True: replaces final 1x1 conv with SNGP GP head (N classes).
    If sngp=False: behaves like a normal UNet with spectral-normalized 1x1 final conv.
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
            self.final_conv = sn_conv1x1(features[0], self.outchannels, bias=True)

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
    x = torch.randn(2, 1, 160, 160, device=device)  # in_channels=1 to match model below
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
