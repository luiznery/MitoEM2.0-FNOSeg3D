"""
FNOSeg3D trainer implementation.

Trains FNOSeg3D models using Fourier Neural Operators for 3D segmentation.
"""
from pathlib import Path
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from mitoem2.training.trainer import BaseTrainer
from mitoem2.models.fnoseg3d import FNOSeg3DModel
from mitoem2.data.dataset import nnUNetDataset
from mitoem2.utils.logging import get_logger
from mitoem2.training.callbacks import Callback

logger = get_logger(__name__)


class EMAPseudoDiceCheckpointCallback(Callback):
    """Checkpoint callback that saves when EMA pseudo dice improves."""

    def __init__(self, checkpoint_dir: Path):
        """
        Initialize EMA pseudo dice checkpoint callback.

        Args:
            checkpoint_dir: Directory to save checkpoints.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_ema_pseudo_dice = float('-inf')

        from mitoem2.utils.checkpoint import save_checkpoint
        self.save_checkpoint = save_checkpoint

    def on_epoch_start(self, epoch: int, trainer: Any) -> None:
        """Called at the start of each epoch."""
        pass

    def on_epoch_end(self, epoch: int, trainer: Any, metrics: Dict[str, float]) -> None:
        """Called at the end of each epoch."""
        if "ema_pseudo_dice" in metrics:
            current_ema = metrics["ema_pseudo_dice"]
            if current_ema > self.best_ema_pseudo_dice:
                self.best_ema_pseudo_dice = current_ema
                checkpoint_path = self.checkpoint_dir / "checkpoint_best_ema_pseudo_dice.pth"
                self.save_checkpoint(
                    model=trainer.model,
                    optimizer=trainer.optimizer,
                    epoch=epoch,
                    metrics=metrics,
                    checkpoint_dir=self.checkpoint_dir,
                    checkpoint_name="checkpoint_best_ema_pseudo_dice",
                    is_best=True,
                )
                logger.info(f"Saved best EMA pseudo dice checkpoint: {checkpoint_path}")


class FNOSeg3DTrainer(BaseTrainer):
    """
    Trainer for FNOSeg3D models.

    This trainer handles training of FNOSeg3D models using the unified
    training interface.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # EMA pseudo dice tracking (similar to nnUNet)
        self.ema_pseudo_dice = None
        self.best_ema_pseudo_dice = float('-inf')
        self.ema_alpha = 0.9  # EMA smoothing factor

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        filters: Optional[int] = None,
        num_modes: Optional[int] = None,
        num_transform_blocks: Optional[int] = None,
    ) -> "FNOSeg3DTrainer":
        """
        Create trainer from configuration.

        Args:
            config: Configuration object or dictionary.

        Returns:
            FNOSeg3DTrainer instance.
        """
        # Extract dataset configuration
        dataset_cfg = config.dataset if hasattr(config, "dataset") else config.get("dataset", {})

        data_root = None
        dataset_id = None

        # Parse dataset config
        if isinstance(dataset_cfg, dict):
            data_root = dataset_cfg.get("data_root") or dataset_cfg.get("root") or dataset_cfg.get("path")
            dataset_id = dataset_cfg.get("id") or dataset_cfg.get("name")
        else:
            # Could be an object with attributes
            if hasattr(dataset_cfg, "id"):
                dataset_id = getattr(dataset_cfg, "id")
            elif hasattr(dataset_cfg, "name"):
                dataset_id = getattr(dataset_cfg, "name")
            elif isinstance(dataset_cfg, str):
                dataset_id = dataset_cfg

            if hasattr(dataset_cfg, "data_root"):
                data_root = getattr(dataset_cfg, "data_root")
            elif hasattr(dataset_cfg, "root"):
                data_root = getattr(dataset_cfg, "root")
            elif hasattr(dataset_cfg, "path"):
                data_root = getattr(dataset_cfg, "path")

        # Derive data_root if not explicitly provided
        if data_root is None:
            if dataset_id:
                repo_root = Path.cwd()
                data_root = str(repo_root / "datasets" / str(dataset_id))
            else:
                raise ValueError("Dataset path not provided in config.dataset; no dataset id to derive path from")

        # Create datasets
        train_ds = nnUNetDataset(Path(data_root), split="train", use_instances=False)
        val_ds = nnUNetDataset(Path(data_root), split="test", use_instances=False)

        # Extract training configuration
        batch_size = 1
        learning_rate = 1e-3
        device = None

        if hasattr(config, "training"):
            if isinstance(config.training, dict):
                batch_size = config.training.get("batch_size", batch_size)
                learning_rate = config.training.get("learning_rate", learning_rate)
            else:
                batch_size = getattr(config.training, "batch_size", batch_size)
                learning_rate = getattr(config.training, "learning_rate", learning_rate)

        # Create data loaders
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        # Extract model configuration
        model_cfg = config.model if hasattr(config, "model") else config.get("model", {})

        model_params = {
            "in_channels": 1,
            "out_channels": 2,
            "filters": 24,
            "num_transform_blocks": 8,
            "num_modes": 16,
            "use_resize": True,
            "use_deep_supervision": False,
            "activation": "selu",
            "output_activation": "softmax",
            "device": device,
        }

        # Update model parameters from config
        if isinstance(model_cfg, dict):
            model_params.update(model_cfg)
        else:
            for key, value in model_params.items():
                if hasattr(model_cfg, key):
                    model_params[key] = getattr(model_cfg, key)

        # Override with command-line parameters if provided
        if filters is not None:
            model_params["filters"] = filters
        if num_modes is not None:
            model_params["num_modes"] = num_modes
        if num_transform_blocks is not None:
            model_params["num_transform_blocks"] = num_transform_blocks

        # Create model
        model = FNOSeg3DModel(
            **model_params,
            config=config.to_dict() if hasattr(config, "to_dict") else (config if isinstance(config, dict) else {}),
        )

        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Create loss function
        criterion = nn.CrossEntropyLoss()

        # Extract device for trainer
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Create callbacks
        callbacks = []

        # Add EMA pseudo dice checkpoint callback
        if hasattr(config, "output") and hasattr(config.output, "model_dir"):
            model_dir = Path(config.output.model_dir)
        elif isinstance(config, dict) and "output" in config and "model_dir" in config["output"]:
            model_dir = Path(config["output"]["model_dir"])
        else:
            model_dir = Path("./fnoseg3d_results")

        ema_callback = EMAPseudoDiceCheckpointCallback(model_dir)
        callbacks.append(ema_callback)

        return cls(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            use_multi_gpu=False,
            callbacks=callbacks,
        )

    def validate(self) -> Dict[str, float]:
        """
        Validate the model and compute aggregated metrics including pseudo dice.

        Returns:
            Dictionary of validation metrics.
        """
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        total_dice = 0.0
        total_iou = 0.0
        all_pseudo_dice = []
        num_batches = 0

        with torch.no_grad():
            from tqdm import tqdm
            pbar = tqdm(self.val_loader, desc="Validation")
            for batch in pbar:
                # Move batch to device
                batch = self._move_batch_to_device(batch)

                # Validation step
                step_metrics = self.validate_step(batch)

                loss = step_metrics.get("loss", 0.0)
                dice = step_metrics.get("dice", 0.0)
                iou = step_metrics.get("iou", 0.0)
                pseudo_dice = step_metrics.get("pseudo_dice", [])

                total_loss += loss
                total_dice += dice
                total_iou += iou
                if pseudo_dice:
                    all_pseudo_dice.append(pseudo_dice)
                num_batches += 1

                pbar.set_postfix({"loss": f"{loss:.4f}", "dice": f"{dice:.4f}"})

        # Compute averages
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_dice = total_dice / num_batches if num_batches > 0 else 0.0
        avg_iou = total_iou / num_batches if num_batches > 0 else 0.0

        # Compute average pseudo dice across all batches and classes
        avg_pseudo_dice = []
        if all_pseudo_dice:
            # Transpose to get per-class averages
            pseudo_dice_array = np.array(all_pseudo_dice)  # Shape: (num_batches, num_classes)
            avg_pseudo_dice = pseudo_dice_array.mean(axis=0).tolist()  # Average across batches

        result = {
            "val_loss": avg_loss,
            "val_dice": avg_dice,
            "val_iou": avg_iou,
        }

        if avg_pseudo_dice:
            result["pseudo_dice"] = avg_pseudo_dice
            result["ema_pseudo_dice"] = self.ema_pseudo_dice

            # Log pseudo dice similar to nnUNet
            logger.info(f"Pseudo dice {avg_pseudo_dice}")
            logger.info(f"EMA pseudo Dice: {self.ema_pseudo_dice:.6f}")

            # Check for new best EMA pseudo dice
            if self.ema_pseudo_dice > self.best_ema_pseudo_dice:
                self.best_ema_pseudo_dice = self.ema_pseudo_dice
                logger.info(f"Yayy! New best EMA pseudo Dice: {self.best_ema_pseudo_dice:.6f}")

        return result

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform a training step.

        Args:
            batch: Batch of data with 'image' and 'mask' keys.

        Returns:
            Dictionary with training metrics.
        """
        self.model.train()

        # Extract images and masks from batch
        images = batch["image"]
        masks = batch["mask"]

        # Ensure images have correct shape: (B, C, D, H, W)
        if images.dim() == 4:
            images = images.unsqueeze(1).float()
        elif images.dim() == 5:
            images = images.float()
        else:
            images = images.unsqueeze(0).unsqueeze(1).float()

        # Convert masks to long type and binary labels (0 background, 1 foreground)
        masks = masks.long()
        masks = (masks > 0).long()

        # Ensure correct shape: (B, D, H, W)
        if masks.dim() == 5:
            masks = masks.squeeze(1)

        # Forward pass
        preds = self.model(images)

        # Compute loss
        loss = self.criterion(preds, masks)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Compute metrics
        with torch.no_grad():
            if preds.dim() > 1 and preds.size(1) > 1:
                pred_labels = torch.argmax(preds, dim=1)
            else:
                pred_labels = (torch.sigmoid(preds) > 0.5).long().squeeze(1)

            tgt = masks
            pred_fg = (pred_labels == 1).long()
            tgt_fg = (tgt == 1).long()

            # Compute Dice and IoU per sample, then average
            b = pred_fg.size(0)
            pred_flat = pred_fg.view(b, -1)
            tgt_flat = tgt_fg.view(b, -1)

            inter = (pred_flat & tgt_flat).sum(dim=1).float()
            denom = pred_flat.sum(dim=1).float() + tgt_flat.sum(dim=1).float()

            dice_per = torch.where(denom > 0, 2.0 * inter / (denom + 1e-8), torch.ones_like(denom))
            iou_per = torch.where(denom > 0, inter / (denom - inter + 1e-8), torch.ones_like(denom))

            dice = dice_per.mean().item()
            iou = iou_per.mean().item()

        return {"loss": loss.item(), "dice": dice, "iou": iou}

    def validate_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform a validation step.

        Args:
            batch: Batch of data with 'image' and 'mask' keys.

        Returns:
            Dictionary with validation metrics.
        """
        self.model.eval()

        with torch.no_grad():
            # Extract images and masks from batch
            images = batch["image"]
            masks = batch["mask"]

            # Ensure images have correct shape: (B, C, D, H, W)
            if images.dim() == 4:
                images = images.unsqueeze(1).float()
            elif images.dim() == 5:
                images = images.float()
            else:
                images = images.unsqueeze(0).unsqueeze(1).float()

            # Convert masks to long type and binary labels
            masks = masks.long()
            masks = (masks > 0).long()

            # Ensure correct shape: (B, D, H, W)
            if masks.dim() == 5:
                masks = masks.squeeze(1)

            # Forward pass
            preds = self.model(images)
            loss = self.criterion(preds, masks)

            # Compute metrics
            if preds.dim() > 1 and preds.size(1) > 1:
                pred_labels = torch.argmax(preds, dim=1)
                # Get softmax probabilities for pseudo dice
                pred_probs = torch.softmax(preds, dim=1)
            else:
                pred_labels = (torch.sigmoid(preds) > 0.5).long().squeeze(1)
                pred_probs = torch.sigmoid(preds)

            tgt = masks
            pred_fg = (pred_labels == 1).long()
            tgt_fg = (tgt == 1).long()

            # Compute Dice and IoU per sample, then average
            b = pred_fg.size(0)
            pred_flat = pred_fg.view(b, -1)
            tgt_flat = tgt_fg.view(b, -1)

            inter = (pred_flat & tgt_flat).sum(dim=1).float()
            denom = pred_flat.sum(dim=1).float() + tgt_flat.sum(dim=1).float()

            dice_per = torch.where(denom > 0, 2.0 * inter / (denom + 1e-8), torch.ones_like(denom))
            iou_per = torch.where(denom > 0, inter / (denom - inter + 1e-8), torch.ones_like(denom))

            dice = dice_per.mean().item()
            iou = iou_per.mean().item()

            # Compute pseudo dice (dice on softmax probabilities, similar to nnUNet)
            pseudo_dice_scores = []
            for class_idx in range(preds.size(1)):
                pred_class_prob = pred_probs[:, class_idx]  # Shape: (B, D, H, W)
                tgt_class = (tgt == class_idx).float()  # Shape: (B, D, H, W)

                pred_flat = pred_class_prob.view(b, -1)
                tgt_flat = tgt_class.view(b, -1)

                inter = (pred_flat * tgt_flat).sum(dim=1)
                pred_sum = pred_flat.sum(dim=1)
                tgt_sum = tgt_flat.sum(dim=1)

                # Pseudo dice: 2 * intersection / (pred_sum + tgt_sum)
                pseudo_dice_per = torch.where(
                    (pred_sum + tgt_sum) > 0,
                    2.0 * inter / (pred_sum + tgt_sum + 1e-8),
                    torch.ones_like(inter)
                )
                pseudo_dice_scores.append(pseudo_dice_per.mean().item())

            # Update EMA pseudo dice
            current_pseudo_dice = np.mean(pseudo_dice_scores)
            if self.ema_pseudo_dice is None:
                self.ema_pseudo_dice = current_pseudo_dice
            else:
                self.ema_pseudo_dice = self.ema_alpha * self.ema_pseudo_dice + (1 - self.ema_alpha) * current_pseudo_dice

        return {
            "loss": loss.item(),
            "dice": dice,
            "iou": iou,
            "pseudo_dice": pseudo_dice_scores,
            "ema_pseudo_dice": self.ema_pseudo_dice
        }

