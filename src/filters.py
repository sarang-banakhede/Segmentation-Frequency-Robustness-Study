from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _gaussian_kernel_2d(sigma: float, device: torch.device) -> torch.Tensor:
    radius = int(math.ceil(3 * sigma))
    size   = 2 * radius + 1
    coords = torch.arange(size, dtype=torch.float32, device=device) - radius
    g1d    = torch.exp(-0.5 * (coords / sigma) ** 2)
    g1d    = g1d / g1d.sum()
    kernel = g1d.unsqueeze(0) * g1d.unsqueeze(1)
    return kernel.unsqueeze(0).unsqueeze(0)


def apply_low_pass(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return x
    kernel = _gaussian_kernel_2d(sigma, x.device)
    pad    = kernel.shape[-1] // 2
    C      = x.shape[1]
    k      = kernel.expand(C, 1, kernel.shape[2], kernel.shape[3])
    return F.conv2d(x, k, padding=pad, groups=C).clamp(0.0, 1.0)


def apply_gaussian_noise(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return x
    noise_std = sigma / 100.0
    return (x + torch.randn_like(x) * noise_std).clamp(0.0, 1.0)
