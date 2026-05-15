"""
FNOSeg3D model implementation.

Uses Fourier Neural Operators for 3D image segmentation.
Based on IBM's multimodal-3d-image-segmentation repository.
"""

from pathlib import Path
from typing import Optional, Dict, Any, Union, List
import torch
import torch.nn as nn

from mitoem2.models.base import BaseModel
from mitoem2.utils.logging import get_logger

logger = get_logger(__name__)


class FNOSeg3DModel(BaseModel):
    """
    FNOSeg3D model for 3D instance segmentation.

    This model uses Fourier Neural Operators (FNO) for efficient 3D segmentation.
    It supports 3D volumetric data and is designed to work with medical imaging datasets.

    Args:
        in_channels: Number of input channels (default: 1).
        out_channels: Number of output channels (default: 2, for binary segmentation).
        filters: Number of filters in the network (default: 24).
        num_transform_blocks: Number of FNO transform blocks (default: 8).
        num_modes: Number of frequency modes (k_max). Can be int or list [d, h, w]
                   (default: 16, meaning 16 modes in each dimension).
        use_resize: Whether to use learnable input/output resizing (default: True).
        use_deep_supervision: Whether to use deep supervision (default: False).
        activation: Activation function to use (default: 'selu').
        output_activation: Activation for output (default: 'softmax').
        device: Device to use ('cuda', 'cpu', etc.) (default: None for auto-detect).
        config: Optional configuration dictionary.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        filters: int = 24,
        num_transform_blocks: int = 8,
        num_modes: Union[int, List[int]] = 16,
        use_resize: bool = True,
        use_deep_supervision: bool = False,
        activation: str = "selu",
        output_activation: str = "softmax",
        device: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize FNOSeg3D model."""
        super().__init__(config)

        # Import NeuralOperatorSeg from IBM's multimodal-3d-image-segmentation repo
        try:
            import sys
            from pathlib import Path

            # Add the IBM repo to path
            ibm_repo_path = Path(__file__).parent.parent.parent.parent / "third_party" / "multimodal-3d-image-segmentation"
            if ibm_repo_path.exists():
                sys.path.insert(0, str(ibm_repo_path))

            from nets import NeuralOperatorSeg

        except ImportError as e:
            raise ImportError(
                "NeuralOperatorSeg from IBM's multimodal-3d-image-segmentation is required. "
                "Ensure the repo is cloned in third_party/ directory. "
                f"Error: {e}"
            ) from e

        # Validate num_modes
        if isinstance(num_modes, list):
            if len(num_modes) != 3:
                raise ValueError("num_modes as list must have 3 elements [depth, height, width]")
            self.num_modes = num_modes
        else:
            self.num_modes = num_modes

        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Create the underlying FNO model using NeuralOperatorSeg
        self.fno_model = NeuralOperatorSeg(
            in_channels=in_channels,
            out_channels=out_channels,
            filters=filters,
            num_transform_blocks=num_transform_blocks,
            num_modes=self.num_modes,
            transform_type="Fourier",
            weights_type="shared",
            use_resize=use_resize,
            use_deep_supervision=use_deep_supervision,
            use_bias_conv_branch=False,
            use_block_skip=True,
            use_block_concat=True,
            activation=activation,
            output_activation=output_activation,
            ndim=5,  # Always 5 for 3D (batch, channels, depth, height, width)
            device=self.device,
        )

        # Move model to device
        self.fno_model.to(self.device)

        # Store parameters for reconstruction
        self.config.update({
            "in_channels": in_channels,
            "out_channels": out_channels,
            "filters": filters,
            "num_transform_blocks": num_transform_blocks,
            "num_modes": self.num_modes,
            "use_resize": use_resize,
            "use_deep_supervision": use_deep_supervision,
            "activation": activation,
            "output_activation": output_activation,
            "device": device,
        })

        logger.info(
            f"Initialized FNOSeg3D model: "
            f"in_channels={in_channels}, out_channels={out_channels}, "
            f"filters={filters}, blocks={num_transform_blocks}, "
            f"num_modes={self.num_modes}, device={self.device}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, D, H, W) for 3D data.

        Returns:
            Output tensor of shape (B, out_channels, D, H, W).
        """
        if x.dim() != 5:
            raise ValueError(f"Expected 5D input (B, C, D, H, W), got {x.dim()}D input")

        # Ensure input is on the correct device
        x = x.to(self.device)

        # Forward pass through FNO model
        output = self.fno_model(x)

        return output

    def load_weights(self, checkpoint_path: Path, strict: bool = True) -> None:
        """
        Load model weights from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file.
            strict: Whether to strictly enforce that the keys match.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "fno_model" in checkpoint:
                state_dict = checkpoint["fno_model"]
            else:
                # Assume the dict itself is the state dict
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Load state dict into the underlying FNO model
        self.fno_model.load_state_dict(state_dict, strict=strict)
        logger.info(f"Loaded model weights from {checkpoint_path}")

    def save_weights(self, checkpoint_path: Path, additional_info: Optional[Dict[str, Any]] = None) -> None:
        """
        Save model weights to checkpoint.

        Args:
            checkpoint_path: Path to save checkpoint.
            additional_info: Additional information to save with checkpoint.
        """
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.fno_model.state_dict(),
            "config": self.config,
        }

        if additional_info:
            checkpoint.update(additional_info)

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved model weights to {checkpoint_path}")

    def get_num_parameters(self) -> int:
        """
        Get the number of trainable parameters.

        Returns:
            Number of parameters.
        """
        return sum(p.numel() for p in self.fno_model.parameters() if p.requires_grad)

    def to(self, device):
        """Move model to device."""
        self.fno_model.to(device)
        self.device = device
        return self

    def train(self, mode: bool = True):
        """Set model to training mode."""
        self.fno_model.train(mode)
        return self

    def eval(self):
        """Set model to evaluation mode."""
        self.fno_model.eval()
        return self

    def parameters(self):
        """Get model parameters."""
        return self.fno_model.parameters()

    def state_dict(self):
        """Get model state dict."""
        return self.fno_model.state_dict()

    def load_state_dict(self, state_dict, strict: bool = True):
        """Load model state dict."""
        self.fno_model.load_state_dict(state_dict, strict=strict)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Make model callable."""
        return self.forward(x)
