"""
model.py  –  3D U-Net for electron density map reconstruction.

Architecture
------------
Input  : (B, 4, D, H, W)  –  2Fo-Fc, Fo-Fc, Fc, cross-Patterson channels
Output : tuple
    mean_map    (B, 1, D, H, W)  –  z-normalised predicted difference map
    log_var_map (B, 1, D, H, W)  –  per-voxel log-variance of mean_map;
                                     exp(0.5 * log_var) is the predicted std
    log_scale   (B,)             –  log(std) of physical diff; multiply mean_map
                                     by exp(log_scale) to recover physical units

Three encoder levels (base_features=32 by default):
    enc1  32  ch   60×60×60
    enc2  64  ch   30×30×30
    enc3  128 ch   15×15×15
    bot   256 ch    7× 7× 7  (floor(15/2))
Decoder mirrors the encoder with skip connections.
F.interpolate(..., size=enc.shape[2:]) handles the 7→15 rounding exactly.

All Conv3d layers use padding_mode='circular' to respect the periodic
boundary conditions of the crystallographic unit cell.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBlock(nn.Module):
    """Conv3d → [BN] → ReLU, repeated twice.

    use_bn=False: [Conv, ReLU, Conv, ReLU]      — keys at .0 .2 (no-BN checkpoints)
    use_bn=True:  [Conv, BN, ReLU, Conv, BN, ReLU] — keys at .0 .1 .3 .4 (BN checkpoints)
    Layer indices are kept compatible with existing checkpoints for both cases.
    """
    def __init__(self, in_ch, out_ch, use_bn=False):
        super().__init__()
        def _half(c_in, c_out):
            layers = [nn.Conv3d(c_in, c_out, 3, padding=1,
                                padding_mode='circular', bias=not use_bn)]
            if use_bn:
                layers.append(nn.BatchNorm3d(c_out))
            layers.append(nn.ReLU(inplace=True))
            return layers
        self.net = nn.Sequential(*_half(in_ch, out_ch), *_half(out_ch, out_ch))

    def forward(self, x):
        return self.net(x)


class SpectralConv3d(nn.Module):
    """Factorised full-band 3D spectral convolution (F-FNO style).

    A dense FNO weight costs O(C² · m³), which forces a low-mode truncation.
    That is wrong for this problem: a nearly-complete model already has
    correct low-resolution phases, so the ghost-peak signal the network must
    produce lives at mid-to-high resolution — exactly the band a low-pass FNO
    discards. This layer instead uses a separate per-axis complex weight,
    cost O(C² · m) per axis, so it covers the *entire* spectrum cheaply. The
    network learns low modes ≈ passthrough; no band has to be guessed.

    rfftn → per-axis per-frequency channel mixing (the three axis terms are
    summed, F-FNO style) → irfftn. Global receptive field, O(N log N),
    translation-equivariant. The crystallographic grid is exactly periodic
    (one unit cell) so the FFT has no boundary artefacts, and its Fourier
    domain *is* structure-factor space — per-frequency mixing is literally
    per-reflection amplitude/phase correction.

    Each per-axis weight is stored at a fixed reference length in monotonic-
    frequency (fftshift) order and linearly interpolated to the runtime axis
    length, so the layer stays grid-agnostic like the rest of the U-Net.

    Weights are stored real with a trailing (real, imag) axis and viewed as
    complex in forward — sidesteps complex-Parameter edge cases under DDP/AdamW.
    """
    def __init__(self, in_ch, out_ch, ref_len=96):
        super().__init__()
        self.in_ch, self.out_ch, self.ref_len = in_ch, out_ch, ref_len
        scale = 1.0 / (in_ch * out_ch)
        # One complex channel-mixing matrix per frequency control point, per axis.
        def _w():
            return nn.Parameter(scale * torch.randn(in_ch, out_ch, ref_len, 2))
        self.w_d, self.w_h, self.w_w = _w(), _w(), _w()

    def _resample(self, w_real, length, shifted):
        """Interpolate a per-axis weight (Ci,Co,ref_len,2) to `length` → complex.

        shifted=True : stored in fftshift (monotonic-frequency) order; after
                       interpolation, ifftshift back to FFT index order — for
                       the full-FFT axes D and H.
        shifted=False: the rfft axis is already monotonic [0 .. N//2].
        """
        ci, co, ref, _ = w_real.shape
        flat = w_real.permute(3, 0, 1, 2).reshape(1, 2 * ci * co, ref)
        flat = F.interpolate(flat, size=length, mode='linear', align_corners=True)
        w    = flat.reshape(2, ci, co, length).permute(1, 2, 3, 0).contiguous()
        w    = torch.view_as_complex(w)
        if shifted:
            w = torch.fft.ifftshift(w, dim=-1)
        return w

    def forward(self, x):
        B, _, D, H, W = x.shape
        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1))   # (B,Ci,D,H,W//2+1)

        wd = self._resample(self.w_d, D,          shifted=True)
        wh = self._resample(self.w_h, H,          shifted=True)
        ww = self._resample(self.w_w, W // 2 + 1, shifted=False)

        # F-FNO: each axis contributes a per-frequency channel mixing; sum them.
        out_ft  = torch.einsum('bidhw,iod->bodhw', x_ft, wd)
        out_ft += torch.einsum('bidhw,ioh->bodhw', x_ft, wh)
        out_ft += torch.einsum('bidhw,iow->bodhw', x_ft, ww)

        return torch.fft.irfftn(out_ft, s=(D, H, W), dim=(-3, -2, -1))


class FNOBlock3d(nn.Module):
    """One FNO layer: global spectral conv + local 1×1×1 conv bypass.

        out = GELU( SpectralConv(x) + W·x )

    Spectral path carries global / reciprocal-space coupling; the bypass
    carries local real-space mixing. Stack a few of these, or run as a branch
    parallel to the U-Net and sum into the head.
    """
    def __init__(self, in_ch, out_ch, ref_len=96):
        super().__init__()
        self.spectral = SpectralConv3d(in_ch, out_ch, ref_len)
        self.bypass   = nn.Conv3d(in_ch, out_ch, kernel_size=1)
        self.act      = nn.GELU()

    def forward(self, x):
        return self.act(self.spectral(x) + self.bypass(x))


class UNet3D(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_features=32, use_bn=False):
        super().__init__()
        f = base_features
        bn = use_bn

        # ── Encoder ───────────────────────────────────────────────────────────
        self.enc1 = _ConvBlock(in_channels, f,           use_bn=bn)  # → f
        self.enc2 = _ConvBlock(f,           f * 2,       use_bn=bn)  # → 2f
        self.enc3 = _ConvBlock(f * 2,       f * 4,       use_bn=bn)  # → 4f
        self.pool = nn.MaxPool3d(2)

        # ── Bottleneck ────────────────────────────────────────────────────────
        self.bottleneck = _ConvBlock(f * 4, f * 8,       use_bn=bn)  # → 8f

        # ── Decoder ───────────────────────────────────────────────────────────
        # after cat with skip: channels are (up + skip)
        self.dec3 = _ConvBlock(f * 8 + f * 4, f * 4,    use_bn=bn)
        self.dec2 = _ConvBlock(f * 4 + f * 2, f * 2,    use_bn=bn)
        self.dec1 = _ConvBlock(f * 2 + f,     f,         use_bn=bn)

        # ── Map + log-variance head (2 channels) ──────────────────────────────
        # channel 0: z-normalised mean prediction
        # channel 1: per-voxel log-variance of that prediction
        self.head = nn.Conv3d(f, 2, kernel_size=1)

        # ── Scale head (bottleneck → scalar log_std) ───────────────────────────
        self.scale_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),   # (B, 8f, 1, 1, 1)
            nn.Flatten(),              # (B, 8f)
            nn.Linear(f * 8, 1),      # (B, 1)
        )

        # ── Global spectral (FNO) branch ──────────────────────────────────────
        # Runs parallel to the U-Net at full resolution. The conv stack is
        # local; the difference-map problem also has non-local / reciprocal-
        # space structure (ghost peaks are cross-vectors). This branch does
        # full-band Fourier-domain mixing — its FFT *is* structure-factor
        # space — and its output is added to the U-Net mean prediction.
        self.fno = nn.Sequential(
            FNOBlock3d(in_channels, f),
            FNOBlock3d(f,           f),
        )
        self.fno_head = nn.Conv3d(f, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Bottleneck
        b = self.bottleneck(self.pool(e3))

        # Scale prediction from bottleneck
        log_scale = self.scale_head(b).squeeze(1)  # (B,)

        # Decoder — interpolate to exact encoder spatial dims to handle odd sizes
        d3 = self.dec3(torch.cat([
            F.interpolate(b,  size=e3.shape[2:], mode='trilinear', align_corners=False),
            e3], dim=1))
        d2 = self.dec2(torch.cat([
            F.interpolate(d3, size=e2.shape[2:], mode='trilinear', align_corners=False),
            e2], dim=1))
        d1 = self.dec1(torch.cat([
            F.interpolate(d2, size=e1.shape[2:], mode='trilinear', align_corners=False),
            e1], dim=1))

        out = self.head(d1)

        # U-Net (local) mean + global spectral branch
        mean_map = out[:, :1] + self.fno_head(self.fno(x))

        log_var = torch.clamp(out[:, 1:], min=-10.0, max=10.0)
        log_scale = torch.clamp(log_scale, min=-10.0, max=10.0)
        return mean_map, log_var, log_scale  # mean_map, log_var_map, log_scale


class TwoStageUNet3D(nn.Module):
    """Two-stage U-Net for iterative electron density refinement.

    Stage 1 (frozen): UNet3D(in_channels=4) → pred1
    Stage 2 (trained): UNet3D(in_channels=5) → pred2
        Input to Stage 2 = [original 4 channels | Stage-1 prediction]

    Load Stage 1 weights from a checkpoint, then freeze them.
    Only Stage 2 parameters are optimised during training.
    """

    def __init__(self, base_features=32):
        super().__init__()
        self.stage1 = UNet3D(in_channels=4, out_channels=1,
                             base_features=base_features)
        self.stage2 = UNet3D(in_channels=5, out_channels=1,
                             base_features=base_features)

    def load_stage1(self, checkpoint_path, device='cpu'):
        """Load Stage-1 weights from a checkpoint and freeze them."""
        ckpt = torch.load(checkpoint_path, map_location=device)
        self.stage1.load_state_dict(ckpt['model'])
        for p in self.stage1.parameters():
            p.requires_grad_(False)
        self.stage1.eval()

    def forward(self, x):
        with torch.no_grad():
            pred1 = self.stage1(x)          # (B, 1, D, H, W)
        inp2  = torch.cat([x, pred1], dim=1)  # (B, 5, D, H, W)
        pred2 = self.stage2(inp2)
        return pred2


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
