"""
Wrapper for the FNOSeg3D model using NeuralOperatorSeg from the
multimodal-3d-image-segmentation repository.

This wrapper will attempt to locate the cloned external repo under
`<repo_root>/external/multimodal-3d-image-segmentation` and dynamically
import a class named `NeuralOperatorSeg`. If not found, it raises a
helpful error telling the user to run the setup script to clone the repo.
"""
from pathlib import Path
from typing import Optional, Dict, Any
import sys
import importlib.util
import inspect

import torch

from mitoem2.models.base import BaseModel
from mitoem2.utils.logging import get_logger

logger = get_logger(__name__)


class FNOSeg3DModel(BaseModel):
    """Wrapper around external `NeuralOperatorSeg` implementation.

    The wrapper will forward the input tensor to the external model.
    The external repository should be cloned to `external/multimodal-3d-image-segmentation`.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        repo_root = Path(__file__).resolve().parents[3]
        external_path = repo_root / "external" / "multimodal-3d-image-segmentation"

        if not external_path.exists():
            raise ImportError(
                "FNOSeg3D external repo not found. Run scripts/setup_fnoseg3d.sh to clone it into 'external/'."
            )


        # Try importing as a package first (preferred) by adding external path to sys.path
        NeuralOperatorSeg = None
        try:
            if str(external_path) not in sys.path:
                sys.path.insert(0, str(external_path))

            # The external repo exposes a `nets` package with `architectures` module
            try:
                from nets.architectures import NeuralOperatorSeg as _NOS
            except Exception:
                # Try non-package path under `tensorflow/nets` as well
                try:
                    from tensorflow.nets.architectures import NeuralOperatorSeg as _NOS
                except Exception:
                    _NOS = None

            if _NOS is not None:
                NeuralOperatorSeg = _NOS
                logger.info("Imported NeuralOperatorSeg via package import")
        except Exception:
            # Ignore and fallback to scanning files
            NeuralOperatorSeg = None

        # Fallback: Search for a Python file that defines NeuralOperatorSeg
        if NeuralOperatorSeg is None:
            for py_file in external_path.rglob("*.py"):
                try:
                    spec = importlib.util.spec_from_file_location(f"fnovendor.{py_file.stem}", str(py_file))
                    if spec is None or spec.loader is None:
                        continue
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                except Exception:
                    # Ignore files that fail to import
                    continue

                if hasattr(module, "NeuralOperatorSeg"):
                    NeuralOperatorSeg = getattr(module, "NeuralOperatorSeg")
                    logger.info(f"Found NeuralOperatorSeg in {py_file}")
                    break

        if NeuralOperatorSeg is None:
            raise ImportError(
                "Could not find class `NeuralOperatorSeg` in the cloned repository. "
                "Ensure the repo at 'external/multimodal-3d-image-segmentation' contains the implementation."
            )

        # Build kwargs for constructor from config if possible
        model_cfg = self.config.get("fnoseg3d", {}) if self.config else {}

        # Provide sensible defaults for required constructor args
        # Smaller defaults for smoke runs to reduce memory usage
        default_ctor = {
            "in_channels": model_cfg.get("in_channels", 1),
            "out_channels": model_cfg.get("out_channels", 2),
            "filters": model_cfg.get("filters", 8),
            "num_transform_blocks": model_cfg.get("num_transform_blocks", 2),
            "num_modes": model_cfg.get("num_modes", (4, 4, 4)),
            "transform_type": model_cfg.get("transform_type", "Fourier"),
            "ndim": model_cfg.get("ndim", 5),
        }

        # Inspect signature and pass only matching keys, merged with defaults
        try:
            sig = inspect.signature(NeuralOperatorSeg)
            ctor_kwargs = {}
            for name in sig.parameters:
                if name in model_cfg:
                    ctor_kwargs[name] = model_cfg[name]
                elif name in default_ctor:
                    ctor_kwargs[name] = default_ctor[name]

            self.model = NeuralOperatorSeg(**ctor_kwargs)
        except Exception as exc:
            # If instantiation fails, surface a helpful error
            raise RuntimeError(f"Failed to instantiate NeuralOperatorSeg: {exc}")

        # Move to torch module attribute for compatibility
        if not isinstance(self.model, torch.nn.Module):
            raise TypeError("NeuralOperatorSeg instance is not a torch.nn.Module")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def load_weights(self, checkpoint_path: Path, strict: bool = True) -> None:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        state = torch.load(str(checkpoint_path), map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state_dict = state["model_state_dict"]
        elif isinstance(state, dict) and "state_dict" in state:
            state_dict = state["state_dict"]
        else:
            state_dict = state

        # Handle nested state dicts
        try:
            self.model.load_state_dict(state_dict, strict=strict)
        except Exception as exc:
            # Try to load partial keys
            logger.warning("Failed strict load, trying non-strict load: %s", exc)
            self.model.load_state_dict(state_dict, strict=False)
