"""3D models for the resolution-transfer experiments.

- ``fnoseg3d``: IBM's FNOSeg (NeuralOperatorSeg, Fourier, shared weights) as published, with its per-voxel 1x1 branch.
- ``e3``: FNOSeg without the 1x1 branch, plus a physically sized local kernel next to every spectral operator.
- ``e3k2``: ``e3`` with a second, larger local kernel (the best 2D model).
- ``unet3d``: a plain 3D U-Net (3x3x3 convolutions, instance norm, LeakyReLU), the CNN control.

Sizes are relative to the input window (the window side = 1), as in 2D: the data loader gives windows of the same
physical size at every voxel size, so the modes and the local kernels cover the same tissue at every resolution.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ibm_nets import NeuralOperatorSeg

MODEL_NAMES = ("fnoseg3d", "e3", "e3k2", "unet3d")


def _fast_len(n: int) -> int:
    """Smallest length >= n whose prime factors are all <= 7 (fast FFT sizes)."""
    while True:
        m = n
        for p in (2, 3, 5, 7):
            while m % p == 0:
                m //= p
        if m == 1:
            return n
        n += 1


def depthwise_conv3d_fft(x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """F.conv3d(x, k, padding=(k.shape[-3:] // 2), groups=C) for a depthwise kernel k (C, 1, kd, kh, kw), with FFTs.
    The input is zero-padded by at least the kernel half-width on each axis, so the circular correlation never wraps
    into the output window and the border sees zeros, as with the direct convolution."""
    c, _, *ks = k.shape
    size = x.shape[-3:]
    m = [s // 2 for s in ks]
    lens = [_fast_len(n + h) for n, h in zip(size, m)]
    g = x.new_zeros((c, *lens), dtype=torch.float32)
    g[:, :ks[0], :ks[1], :ks[2]] = k[:, 0].float()
    g = torch.roll(g, shifts=[-h for h in m], dims=(-3, -2, -1))  # the tap at offset d sits at index d mod L
    spec = torch.fft.rfftn(x.float(), s=lens, dim=(-3, -2, -1)) * torch.fft.rfftn(g, dim=(-3, -2, -1)).conj()
    y = torch.fft.irfftn(spec, s=lens, dim=(-3, -2, -1))
    return y[..., :size[0], :size[1], :size[2]].to(x.dtype)


class PhysLocalConv3d(nn.Module):
    """Depthwise local integral kernel whose support is fixed in window units (so in physical units), not in voxels.

    k_c is the trilinear interpolation of a learned n_ctrl^3 grid of values surrounded by a ring of zeros, spanning
    [-radius, radius]^3 (window side = 1). On a grid of N voxels per axis it is sampled at the voxel offsets inside the
    support and multiplied by the voxel volume relative to a ref_size^3 grid, so every resolution computes the same
    integral. Applied with FFTs (cost independent of the kernel size). Zero-initialised."""

    def __init__(self, channels: int, radius: float, n_ctrl: int = 5, ref_size: int = 48):
        super().__init__()
        self.radius, self.ref_size = radius, ref_size
        self.weight = nn.Parameter(torch.zeros(channels, 1, n_ctrl, n_ctrl, n_ctrl))

    def kernel(self, size) -> torch.Tensor:
        axes = []
        for n in size:
            m = int(self.radius * n)  # voxels on each side of the centre
            axes.append(torch.arange(-m, m + 1, device=self.weight.device, dtype=self.weight.dtype) / n / self.radius)
        gd, gh, gw = torch.meshgrid(*axes, indexing="ij")  # offsets in units of the radius, in [-1, 1]
        grid = torch.stack([gw, gh, gd], dim=-1)[None].expand(self.weight.shape[0], -1, -1, -1, -1)
        k = F.grid_sample(F.pad(self.weight, (1, 1, 1, 1, 1, 1)), grid, mode="bilinear", padding_mode="zeros",
                          align_corners=True)  # trilinear for 5-D input
        vol = size[0] * size[1] * size[2]
        return k * (self.ref_size ** 3 / vol)  # quadrature: voxel volume relative to the reference grid


def _sum_kernels(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Sum of two centred odd-sized kernels, the smaller zero-padded to the size of the larger."""
    tgt = [max(x, y) for x, y in zip(a.shape[-3:], b.shape[-3:])]

    def pad(k):
        p = []
        for t, s in zip(reversed(tgt), reversed(k.shape[-3:])):
            p += [(t - s) // 2, (t - s) // 2]
        return F.pad(k, p)
    return pad(a) + pad(b)


class PhysLocalBranch3d(nn.Module):
    """Spectral operator + one or two physically sized local kernels, summed into one FFT convolution."""

    def __init__(self, spectral: nn.Module, channels: int, radii, ref_size: int):
        super().__init__()
        self.spectral = spectral
        self.locals = nn.ModuleList(PhysLocalConv3d(channels, r, ref_size=ref_size) for r in radii)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-3:]
        k = self.locals[0].kernel(size)
        for loc in self.locals[1:]:
            k = _sum_kernels(k, loc.kernel(size))
        return self.spectral(x) + depthwise_conv3d_fft(x, k)


class _ConvBlock3d(nn.Sequential):
    def __init__(self, cin: int, cout: int):
        super().__init__(nn.Conv3d(cin, cout, 3, padding=1), nn.InstanceNorm3d(cout, affine=True), nn.LeakyReLU(0.01),
                         nn.Conv3d(cout, cout, 3, padding=1), nn.InstanceNorm3d(cout, affine=True), nn.LeakyReLU(0.01))


class UNet3D(nn.Module):
    """Plain 3D U-Net: ``levels`` scales (widths c, 2c, 4c, ...), 2x2x2 max-pooling, transposed convolutions, skips."""

    def __init__(self, in_channels: int = 1, num_classes: int = 2, channels: int = 16, levels: int = 4):
        super().__init__()
        w = [channels * 2 ** i for i in range(levels)]
        self.enc = nn.ModuleList([_ConvBlock3d(in_channels, w[0])] + [_ConvBlock3d(w[i - 1], w[i]) for i in range(1, levels)])
        self.up = nn.ModuleList(nn.ConvTranspose3d(w[i], w[i - 1], 2, 2) for i in range(levels - 1, 0, -1))
        self.dec = nn.ModuleList(_ConvBlock3d(2 * w[i - 1], w[i - 1]) for i in range(levels - 1, 0, -1))
        self.out = nn.Conv3d(w[0], num_classes, 1)
        self.pool = nn.MaxPool3d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for i, enc in enumerate(self.enc):
            x = enc(x)
            if i < len(self.enc) - 1:
                skips.append(x)
                x = self.pool(x)
        for up, dec, skip in zip(self.up, self.dec, reversed(skips)):
            x = dec(torch.cat([up(x), skip], dim=1))
        return self.out(x)


def build_model(name: str, filters: int = 32, blocks: int = 8, modes: int = 8, radii=(4 / 64, 8 / 64),
                train_window: int = 96, unet_channels: int = 16, unet_levels: int = 4) -> nn.Module:
    """``radii`` are in window units; ``train_window`` (voxels per axis at the training level) sets the reference
    grid of the quadrature (the grid after the stride-2 input layer, train_window // 2)."""
    if name not in MODEL_NAMES:
        raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")
    if name == "unet3d":
        return UNet3D(1, 2, unet_channels, unet_levels)
    net = NeuralOperatorSeg(1, 2, filters, blocks, (modes, modes, modes), transform_type="Fourier", use_resize=True,
                            output_activation=nn.Identity(), ndim=5)
    if name == "fnoseg3d":
        return net
    from .ibm_nets.architectures import NeuralOperatorBlock
    use = radii[:1] if name == "e3" else radii
    for blk in net.modules():
        if isinstance(blk, NeuralOperatorBlock):
            blk.conv_branch = None  # no per-voxel 1x1 branch
            blk.op = PhysLocalBranch3d(blk.op, filters, use, ref_size=train_window // 2)
    return net


def n_params(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())
