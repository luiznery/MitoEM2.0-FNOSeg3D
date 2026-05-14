from pathlib import Path
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mitoem2.training.trainer import BaseTrainer
from mitoem2.models.fnoseg3d import FNOSeg3DModel
from mitoem2.data.dataset import nnUNetDataset
from mitoem2.utils.logging import get_logger

logger = get_logger(__name__)


class FNOSeg3DTrainer(BaseTrainer):
    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        # Dataset
        dataset_cfg = config.dataset if hasattr(config, "dataset") else config.get("dataset", {})

        data_root = None
        # If dataset_cfg is a dict-like, look for explicit path keys
        if isinstance(dataset_cfg, dict):
            data_root = dataset_cfg.get("data_root") or dataset_cfg.get("root") or dataset_cfg.get("path")
            dataset_id = dataset_cfg.get("id") or dataset_cfg.get("name")
        else:
            # Could be an object with attributes, or simply a string id
            dataset_id = None
            if hasattr(dataset_cfg, "id"):
                dataset_id = getattr(dataset_cfg, "id")
            elif isinstance(dataset_cfg, str):
                dataset_id = dataset_cfg

        # If no explicit path provided, derive from repo `datasets/` using dataset id
        if data_root is None:
            if dataset_id:
                    # Use current working directory (script is run from repo root) to derive datasets path
                    repo_root = Path.cwd()
                    data_root = str(repo_root / "datasets" / str(dataset_id))
            else:
                raise ValueError("Dataset path not provided in config.dataset; no dataset id to derive path from")

        # Use labelsTr by default (most datasets here provide `labelsTr`)
        train_ds = nnUNetDataset(Path(data_root), split="train", use_instances=False)
        val_ds = nnUNetDataset(Path(data_root), split="test", use_instances=False)

        batch_size = 1
        if hasattr(config, "training"):
            if isinstance(config.training, dict):
                batch_size = config.training.get("batch_size", batch_size)
            else:
                batch_size = getattr(config.training, "batch_size", batch_size)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

        model = FNOSeg3DModel(config=config.to_dict() if hasattr(config, "to_dict") else (config if isinstance(config, dict) else {}))

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        return cls(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=None,
            use_multi_gpu=False,
            callbacks=[],
        )

    def train_step(self, batch):
        self.model.train()
        images = batch["image"]
        # Ensure images have channel dimension: (B, C, D, H, W)
        if images.dim() == 4:
            images = images.unsqueeze(1).float()
        elif images.dim() == 5:
            images = images.float()
        else:
            images = images.unsqueeze(0).unsqueeze(1).float()

        masks = batch["mask"].long()
        # Convert instance masks to binary labels (0 background, 1 foreground)
        masks = (masks > 0).long()

        preds = self.model(images.to(self.device))
        # Ensure preds shape matches criterion
        if preds.dim() == masks.dim():
            # assume already per-voxel logits
            loss = self.criterion(preds, masks.to(self.device))
        else:
            # try to squeeze extra dims
            loss = self.criterion(preds, masks.to(self.device))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def validate_step(self, batch):
        self.model.eval()
        with torch.no_grad():
            images = batch["image"]
            if images.dim() == 4:
                images = images.unsqueeze(1).float()
            elif images.dim() == 5:
                images = images.float()
            else:
                images = images.unsqueeze(0).unsqueeze(1).float()

            masks = batch["mask"].long()
            masks = (masks > 0).long()
            preds = self.model(images.to(self.device))
            loss = self.criterion(preds, masks.to(self.device))
        return {"loss": loss.item()}
