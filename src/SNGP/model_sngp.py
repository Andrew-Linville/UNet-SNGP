# model_sngp.py
import math
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torch.nn.utils import spectral_norm

# 
# Random Fourier Features φ(x)
# 
class RandomFourierFeatures(nn.Module):
    """
    φ(x) = cos(x @ W + b), with W,b fixed (buffers).
    W ~ N(0, I) / ℓ  for Gaussian RBF with length-scale ℓ (kernel_scale).
    """
    def __init__(self, in_features, out_features, kernel_scale=None, seed=None):
        super().__init__()
        assert out_features > 0
        if seed is not None:
            g = torch.Generator()
            g.manual_seed(int(seed))
            weight = torch.randn(in_features, out_features, generator=g)
            bias = torch.rand(out_features, generator=g) * 2 * math.pi
        else:
            weight = torch.randn(in_features, out_features)
            bias = torch.rand(out_features) * 2 * math.pi

        if kernel_scale is None:
            kernel_scale = math.sqrt(in_features / 2.0)
        if not (isinstance(kernel_scale, (int, float)) and kernel_scale > 0):
            raise ValueError("kernel_scale must be a positive float.")
        weight = weight / float(kernel_scale)

        self.register_buffer("weight", weight)
        self.register_buffer("bias", bias)

    def forward(self, x):
        # x: (N, H)
        return torch.cos(x @ self.weight + self.bias)  # (N, D)


# 
# Laplace-style precision/covariance in φ-space
# 
class LaplacePrecision(nn.Module):
    """
    Tracks precision  P = λ I + sum_i α_i φ_i φ_i^T  (D×D),
    and provides predictive variance diag via:
      var_i = λ * φ_i^T Cov φ_i, with Cov = P^{-1}.

    Use only for the final one-pass build; keep disabled during training for speed.
    """
    def __init__(self, dim_features, ridge=1.0, momentum=None, device=None, dtype=None):
        super().__init__()
        self.D = int(dim_features)
        self.ridge = float(ridge)
        self.momentum = momentum  # None or <=0 => exact sum; >0 => EMA

        eye = torch.eye(self.D, device=device, dtype=dtype)
        self.register_buffer("_precision", self.ridge * eye.clone())
        self._cov_cache = None
        self._cov_stale = True

    @property
    def precision(self):
        return self._precision

    def reset(self):
        self._precision[:] = torch.eye(self.D, device=self._precision.device, dtype=self._precision.dtype) * self.ridge
        self._cov_cache = None
        self._cov_stale = True

    def _invalidate_cov(self):
        self._cov_cache = None
        self._cov_stale = True

    def cov(self):
        if self._cov_stale or self._cov_cache is None:
            self._cov_cache = torch.linalg.inv(self._precision)
            self._cov_stale = False
        return self._cov_cache

    @torch.no_grad()
    def accumulate(self, phi, alpha=1.0):
        """
        phi: (B, D)
        alpha: scalar or (B,)
        """
        B, D = phi.shape
        if D != self.D:
            raise ValueError(f"phi has dim {D}, expected {self.D}")
        if isinstance(alpha, torch.Tensor) and alpha.numel() > 1:
            alpha = alpha.reshape(-1, 1)  # (B,1)
            Bt = phi.t() @ (alpha * phi)
        else:
            Bt = (phi.t() @ phi) * float(alpha)

        if (self.momentum is None) or (self.momentum <= 0):
            self._precision += Bt
        else:
            Bt = Bt / float(B)
            m = float(self.momentum)
            self._precision = (m * self._precision) + ((1.0 - m) * Bt)

        self._invalidate_cov()

    @torch.no_grad()
    def predictive_variance_diag(self, phi):
        """
        var_i = λ * φ_i^T Cov φ_i
        phi: (B, D) -> (B,)
        """
        C = self.cov()
        tmp = phi @ C            # (B, D)
        var = self.ridge * torch.sum(tmp * phi, dim=1)
        return var


# 
# Pixel GP Head
# 
class PixelGPHead(nn.Module):
    """
    Drop-in replacement for a 1x1 Conv classifier in UNet.

    Train-time: return_variance=False (fast).
    After training: build_precision_exact(...) once, then enable_variance(True)
    to get per-pixel variance maps.
    """
    def __init__(
        self,
        in_channels,
        num_classes,
        reduction_dim=128,
        rff_dim=1024,
        kernel_scale=None,
        dropout=0.2,
        activation="Tanh",
        ridge=1.0,
        return_variance=False,
        chunk_pixels=8192,
    ):
        super().__init__()
        self.pre = spectral_norm(nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True))
        self.act = getattr(nn, activation)() if hasattr(nn, activation) else nn.Tanh()
        self.drop = nn.Dropout(p=dropout)
        self.reduce = spectral_norm(nn.Conv2d(in_channels, reduction_dim, kernel_size=1, bias=True))

        self.rff = RandomFourierFeatures(reduction_dim, rff_dim, kernel_scale=kernel_scale)
        self.readout = nn.Linear(rff_dim, num_classes, bias=False)

        self.precision = LaplacePrecision(rff_dim, ridge=ridge, momentum=None)
        self.return_variance = bool(return_variance)
        self.chunk_pixels = int(chunk_pixels)

    def enable_variance(self, flag=True):
        self.return_variance = bool(flag)

    @torch.no_grad()
    def reset_precision(self, ridge=None):
        if ridge is not None:
            self.precision.ridge = float(ridge)
        self.precision.reset()

    @torch.no_grad()
    def build_precision_exact(self, feat_loader, device, use_amp=True, get_backbone_feats=None):
        """
        One pass over training data to build exact precision:
            P = λI + Σ φ_i φ_i^T
        `get_backbone_feats(x)` should return the (B,C,H,W) feature map that feeds this head.
        """
        self.eval()
        self.reset_precision()

        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if (use_amp and device.type == "cuda")
            else torch.cuda.amp.autocast(enabled=False)
        )

        for batch in feat_loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device, non_blocking=True)

            with torch.no_grad(), amp_ctx:
                if get_backbone_feats is None:
                    raise RuntimeError("build_precision_exact requires get_backbone_feats callable.")
                feats = get_backbone_feats(x)  # (B,C,H,W)

                B, C, H, W = feats.shape
                reduced = self.reduce(self.drop(self.act(self.pre(feats))))  # (B,R,H,W)
                R = reduced.shape[1]
                flat = reduced.permute(0, 2, 3, 1).reshape(B * H * W, R)

                for i in range(0, flat.shape[0], self.chunk_pixels):
                    sl = slice(i, min(i + self.chunk_pixels, flat.shape[0]))
                    phi = self.rff(flat[sl, :])   # (m,D)
                    self.precision.accumulate(phi, alpha=1.0)

        self.train()

    def forward(self, features):
        """
        features: (B,C,H,W) backbone feature map feeding the head
        Returns logits or (logits, var_map) depending on self.return_variance
        """
        B, C, H, W = features.shape
        x = self.reduce(self.drop(self.act(self.pre(features))))  # (B,R,H,W)
        R = x.shape[1]
        x_flat = x.permute(0, 2, 3, 1).reshape(B * H * W, R)

        logits_out = torch.empty((B * H * W, self.readout.out_features), device=x.device, dtype=x.dtype)
        var_out = None
        if self.return_variance:
            var_out = torch.empty((B * H * W,), device=x.device, dtype=x.dtype)

        for i in range(0, x_flat.shape[0], self.chunk_pixels):
            sl = slice(i, min(i + self.chunk_pixels, x_flat.shape[0]))
            phi = self.rff(x_flat[sl, :])           # (m,D)
            logits = self.readout(phi)              # (m,K)
            logits_out[sl] = logits
            if self.return_variance:
                var_out[sl] = self.precision.predictive_variance_diag(phi)

        logits_map = logits_out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        if self.return_variance:
            var_map = var_out.reshape(B, H, W).unsqueeze(1)
            return logits_map, var_map
        return logits_map


# ---------------------------
# UNet w/ PixelGP head
# ---------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNET_SNGP(nn.Module):
    def __init__(
        self,
        in_channels=3,
        num_classes=1,
        features=(64, 128, 256, 512),
        reduction_dim=128,
        rff_dim=1024,
        kernel_scale=None,
        ridge=1.0,
        return_variance_train=False,
        return_variance_eval=True,
        chunk_pixels=8192,
    ):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        ch = in_channels
        for f in features:
            self.downs.append(DoubleConv(ch, f))
            ch = f

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(f * 2, f))

        # Replace 1x1 conv with GP head
        self.classifier = PixelGPHead(
            in_channels=features[0],
            num_classes=num_classes,
            reduction_dim=reduction_dim,
            rff_dim=rff_dim,
            kernel_scale=kernel_scale,
            ridge=ridge,
            return_variance=return_variance_train,  # off during training for speed
            chunk_pixels=chunk_pixels,
        )
        self._return_variance_eval = bool(return_variance_eval)

    def backbone_to_classifier_feats(self, x):
        """Return the (B,C,H,W) feature map that feeds the classifier head."""
        skips = []
        for d in self.downs:
            x = d(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skips[i // 2]
            if x.shape != skip.shape:
                x = TF.resize(x, size=skip.shape[2:])
            x = self.ups[i + 1](torch.cat((skip, x), dim=1))

        return x  # feeds the head

    def forward(self, x):
        feats = self.backbone_to_classifier_feats(x)
        out = self.classifier(feats)
        if not self.training:
            # honor eval-time preference for variance output
            self.classifier.enable_variance(self._return_variance_eval)
        return out
