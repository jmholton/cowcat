"""
model.py  –  3D U-Net for electron density map reconstruction.

Architecture
------------
Input  : (B, 4, D, H, W)  –  2Fo-Fc, Fo-Fc, Fc, cross-Patterson channels
Output : (B, 1, D, H, W)  –  predicted ground-truth density

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
    """Conv3d → BN → ReLU, repeated twice."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, padding_mode='circular', bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, padding_mode='circular', bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_features=32):
        super().__init__()
        f = base_features

        # ── Encoder ───────────────────────────────────────────────────────────
        self.enc1 = _ConvBlock(in_channels, f)       # → f
        self.enc2 = _ConvBlock(f,           f * 2)   # → 2f
        self.enc3 = _ConvBlock(f * 2,       f * 4)   # → 4f
        self.pool = nn.MaxPool3d(2)

        # ── Bottleneck ────────────────────────────────────────────────────────
        self.bottleneck = _ConvBlock(f * 4, f * 8)   # → 8f

        # ── Decoder ───────────────────────────────────────────────────────────
        # after cat with skip: channels are (up + skip)
        self.dec3 = _ConvBlock(f * 8 + f * 4, f * 4)
        self.dec2 = _ConvBlock(f * 4 + f * 2, f * 2)
        self.dec1 = _ConvBlock(f * 2 + f,     f)

        # ── Head ──────────────────────────────────────────────────────────────
        self.head = nn.Conv3d(f, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Bottleneck
        b = self.bottleneck(self.pool(e3))

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

        return self.head(d1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
