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


QUADRATURES = ("point", "cell", "auto")


class PhysLocalConv3d(nn.Module):
    """Depthwise local integral kernel whose support is fixed in window units (so in physical units), not in voxels.

    k_c is the trilinear interpolation of a learned n_ctrl^3 grid of values surrounded by a ring of zeros, spanning
    [-radius, radius]^3 (window side = 1). On a grid of N voxels per axis it is discretised and multiplied by the voxel
    volume relative to a ref_size^3 grid, so every resolution computes the same integral. ``quadrature``:
    "point" samples k at the voxel centres inside the support (as trained); "cell" averages k over each voxel
    (sub_samples points per voxel and axis) and includes every voxel that overlaps the support, which keeps the
    integral right when the kernel spans only one or two voxels; "auto" uses "point" along axes at least as finely
    sampled as the reference grid (identical to training there) and "cell" along coarser axes. Applied with FFTs (cost independent of the kernel size).
    Zero-initialised."""

    def __init__(self, channels: int, radius: float, n_ctrl: int = 5, ref_size: int = 48, quadrature: str = "point"):
        super().__init__()
        self.radius, self.ref_size, self.quadrature = radius, ref_size, quadrature
        self.weight = nn.Parameter(torch.zeros(channels, 1, n_ctrl, n_ctrl, n_ctrl))

    def kernel(self, size, sub_samples: int = 4) -> torch.Tensor:
        q = getattr(self, "quadrature", "point")
        dev, dt = self.weight.device, self.weight.dtype
        axes, pool = [], []
        for n in size:
            cell = q == "cell" or (q == "auto" and n < self.ref_size)
            s = sub_samples if cell else 1
            m = int(self.radius * n + 0.5) if cell else int(self.radius * n)  # voxels on each side of the centre
            off = torch.arange(-m, m + 1, device=dev, dtype=dt)
            if cell:  # s sub-sample positions inside each voxel
                off = (off[:, None] + (torch.arange(s, device=dev, dtype=dt) + 0.5) / s - 0.5).reshape(-1)
            axes.append(off / n / self.radius)
            pool.append(s)
        gd, gh, gw = torch.meshgrid(*axes, indexing="ij")  # offsets in units of the radius, in [-1, 1]
        grid = torch.stack([gw, gh, gd], dim=-1)[None].expand(self.weight.shape[0], -1, -1, -1, -1)
        k = F.grid_sample(F.pad(self.weight, (1, 1, 1, 1, 1, 1)), grid, mode="bilinear", padding_mode="zeros",
                          align_corners=True)  # trilinear for 5-D input; zero outside the support
        if max(pool) > 1:
            k = F.avg_pool3d(k, pool)
        vol = size[0] * size[1] * size[2]
        return k * (self.ref_size ** 3 / vol)  # quadrature: voxel volume relative to the reference grid


def set_quadrature(net: nn.Module, quadrature: str) -> nn.Module:
    """Discretisation of every physically sized kernel of a trained model ("point" as trained, or "cell")."""
    if quadrature not in QUADRATURES:
        raise ValueError(f"unknown quadrature {quadrature!r}; choose from {QUADRATURES}")
    for mod in net.modules():
        if isinstance(mod, (PhysLocalConv3d, PhysInputConv3d)):
            mod.quadrature = quadrature
    return net


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


class PhysInputConv3d(nn.Module):
    """Input layer without filters sized in voxels: a dense (in -> out channels) local integral kernel of a fixed
    physical radius (window units), trilinear on a 5^3 control grid, applied with FFTs at the input resolution with
    quadrature weights, then 2x2x2 average pooling (the stride-2 step), bias and SELU."""

    def __init__(self, in_channels: int, out_channels: int, radius: float, ref_size: int, n_ctrl: int = 5):
        super().__init__()
        self.radius, self.ref_size = radius, ref_size
        self.weight = nn.Parameter(torch.randn(out_channels * in_channels, 1, n_ctrl, n_ctrl, n_ctrl) * 0.2)
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1, 1))
        self.cin, self.cout = in_channels, out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-3:]
        k = PhysLocalConv3d.kernel(self, size)  # (cout * cin, 1, kd, kh, kw), same parameterisation as the blocks
        ks = k.shape[-3:]
        m = [t // 2 for t in ks]
        lens = [_fast_len(n + h) for n, h in zip(size, m)]
        xf = torch.fft.rfftn(x.float(), s=lens, dim=(-3, -2, -1))  # (b, cin, ...)
        k = k.view(self.cout, self.cin, *ks)
        out = []
        for c0 in range(0, self.cout, 8):  # a few output channels at a time: full-resolution spectra are large
            kc = k[c0:c0 + 8]
            g = x.new_zeros((kc.shape[0], self.cin, *lens), dtype=torch.float32)
            g[..., :ks[0], :ks[1], :ks[2]] = kc.float()
            g = torch.roll(g, shifts=[-h for h in m], dims=(-3, -2, -1))
            kf = torch.fft.rfftn(g, dim=(-3, -2, -1)).conj()
            y = torch.fft.irfftn(torch.einsum("oi...,bi...->bo...", kf, xf), s=lens, dim=(-3, -2, -1))
            out.append(F.avg_pool3d(y[..., :size[0], :size[1], :size[2]], 2))
        return F.selu(torch.cat(out, 1) + self.bias)


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
                train_window: int = 96, unet_channels: int = 16, unet_levels: int = 4, input_layer: str = "conv",
                input_radius: float = 2 / 64) -> nn.Module:
    """``radii`` are in window units; ``train_window`` (voxels per axis at the training level) sets the reference
    grid of the quadrature (the grid after the stride-2 input layer, train_window // 2). ``input_layer``: "conv"
    (IBM's learned 2x2x2 stride-2 convolution, sized in voxels), "avgpool" (2x2x2 average + 1x1 convolution + SELU)
    or "phys" (PhysInputConv3d: a physically sized kernel, then 2x2x2 average)."""
    if name not in MODEL_NAMES:
        raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")
    if name == "unet3d":
        return UNet3D(1, 2, unet_channels, unet_levels)
    net = NeuralOperatorSeg(1, 2, filters, blocks, (modes, modes, modes), transform_type="Fourier", use_resize=True,
                            output_activation=nn.Identity(), ndim=5)
    if input_layer == "avgpool":
        net.conv_in = nn.Sequential(nn.AvgPool3d(2), nn.Conv3d(1, filters, 1), nn.SELU())
    elif input_layer == "phys":
        net.conv_in = PhysInputConv3d(1, filters, input_radius, ref_size=train_window)
    elif input_layer != "conv":
        raise ValueError(f"unknown input layer {input_layer!r}")
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
