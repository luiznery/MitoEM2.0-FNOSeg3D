"""
FNOSeg3D model implementation (simplified).

This file implements a compact 3D Fourier Neural Operator inspired
by the FNOSeg3D architecture. The implementation is intentionally
lightweight and self-contained so it can be used in this project
without external dependencies.
"""
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.fft

from mitoem2.models.base import BaseModel
from mitoem2.utils.logging import get_logger

logger = get_logger(__name__)


class SpectralConv3d(nn.Module):
    """3D spectral convolution operating on low-frequency Fourier modes."""

    def __init__(self, in_channels: int, out_channels: int, modes: Tuple[int, int, int]):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  # (m_x, m_y, m_z)

        # Weight for real and imaginary parts
        m_x, m_y, m_z = modes
        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, m_x, m_y, m_z, 2)
        )

    def compl_mul3d(self, input_ft, weights):
        # input_ft: (batch, in_ch, x, y, z) complex tensor represented as complex dtype
        # weights: real+imag stacked
        # Convert stored weights into complex tensor
        w_real = weights[..., 0]
        w_imag = weights[..., 1]
        w = torch.complex(w_real, w_imag)
        # Multiply low-frequency modes
        # input_ft: (batch, in_ch, nx, ny, nz)
        # We will perform einsum for the selected modes
        # Perform complex multiplication using real/imag parts to preserve
        # autograd support across devices.
        x_r = input_ft.real
        x_i = input_ft.imag
        w_r = w.real
        w_i = w.imag

        out_r = torch.einsum("bixyz, ioxyz -> boxyz", x_r, w_r) - torch.einsum("bixyz, ioxyz -> boxyz", x_i, w_i)
        out_i = torch.einsum("bixyz, ioxyz -> boxyz", x_r, w_i) + torch.einsum("bixyz, ioxyz -> boxyz", x_i, w_r)
        return torch.complex(out_r, out_i)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_ch, nx, ny, nz)
        batchsize = x.shape[0]
        # Compute Fourier transform
        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1))

        # Extract low-frequency modes
        m_x, m_y, m_z = self.modes
        # rfftn produces last dim size floor(n/2)+1 along z
        x_ft_low = x_ft[..., :m_x, :m_y, :m_z]

        # x_ft_low: (batch, in_ch, m_x, m_y, m_z)
        # Multiply per-channel with weights
        # Rearrange to (batch, in_ch, m_x, m_y, m_z)
        # Use einsum to multiply in->out
        # Multiply using real/imag parts to keep computation in real tensors
        w_r = self.weights[..., 0]
        w_i = self.weights[..., 1]

        x_r = x_ft_low.real
        x_i = x_ft_low.imag

        out_r = torch.einsum("bixyz, ioxyz -> boxyz", x_r, w_r) - torch.einsum("bixyz, ioxyz -> boxyz", x_i, w_i)
        out_i = torch.einsum("bixyz, ioxyz -> boxyz", x_r, w_i) + torch.einsum("bixyz, ioxyz -> boxyz", x_i, w_r)
        out_ft = torch.complex(out_r, out_i)

        # create full frequency tensor with same shape as x_ft but only low modes filled
        full_shape = list(x_ft.shape)
        out_ch = self.out_channels
        out_full = torch.zeros((batchsize, out_ch, *full_shape[-3:]), dtype=out_ft.dtype, device=x.device)
        out_full[..., :m_x, :m_y, :m_z] = out_ft

        # Inverse FFT to return to spatial domain
        out = torch.fft.irfftn(out_full, s=x.shape[-3:], dim=(-3, -2, -1))
        return out


class FNOSeg3DModel(BaseModel):
    """Simplified FNOSeg3D model for 3D segmentation."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        width: int = 32,
        modes: Tuple[int, int, int] = (8, 8, 8),
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(config)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.modes = modes

        self.input_proj = nn.Conv3d(in_channels, width, kernel_size=1)
        self.spectral1 = SpectralConv3d(width, width, modes)
        self.spectral2 = SpectralConv3d(width, width, modes)
        self.w1 = nn.Conv3d(width, width, kernel_size=1)
        self.w2 = nn.Conv3d(width, width, kernel_size=1)
        self.norm = nn.GroupNorm(8, width)
        self.relu = nn.ReLU(inplace=True)
        self.output_proj = nn.Conv3d(width, out_channels, kernel_size=1)

        logger.info(f"Initialized FNOSeg3DModel (in={in_channels}, out={out_channels}, width={width})")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass expects input shape: (B, C, X, Y, Z)
        Returns logits with shape: (B, out_channels, X, Y, Z)
        """
        # Ensure input has channel dim
        if x.ndim != 5:
            raise ValueError("Expected input tensor of shape (B, C, X, Y, Z)")

        h = self.input_proj(x)

        # spectral conv expects (B, C, X, Y, Z)
        s = self.spectral1(h)
        s = s + self.w1(h)
        s = self.relu(self.norm(s))

        h = s
        s = self.spectral2(h)
        s = s + self.w2(h)
        s = self.relu(self.norm(s))

        out = self.output_proj(s)
        return out

    def load_weights(self, checkpoint_path: Path, strict: bool = True) -> None:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state = checkpoint["state_dict"]
        else:
            state = checkpoint

        self.load_state_dict(state, strict=strict)
        logger.info(f"Loaded FNOSeg3D weights from {checkpoint_path}")

    def save_weights(self, checkpoint_path: Path, additional_info: Optional[Dict[str, Any]] = None) -> None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": self.config,
        }
        if additional_info:
            checkpoint.update(additional_info)
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved FNOSeg3D weights to {checkpoint_path}")
