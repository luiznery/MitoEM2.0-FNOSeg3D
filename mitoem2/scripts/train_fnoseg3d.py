#!/usr/bin/env python
"""Simple training loop for FNOSeg3D model.

This script is intentionally minimal: it loads a 3D dataset using the
existing `nnUNetDataset` class, constructs the `FNOSeg3DModel`, and runs
a basic training loop saving checkpoints to the output directory.
"""
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from mitoem2.models.fnoseg3d import FNOSeg3DModel
from mitoem2.data.dataset import nnUNetDataset
from mitoem2.utils.logging import setup_logger


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True, help="Path to nnUNet dataset folder")
    p.add_argument("--split", default="train", help="Dataset split (train or test)")
    p.add_argument("--output-dir", default="outputs/fnoseg3d", help="Directory to save checkpoints")
    p.add_argument("--in-channels", type=int, default=1)
    p.add_argument("--out-channels", type=int, default=1)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--modes", type=int, nargs=3, default=[8, 8, 8])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--resume", type=str, default=None)
    return p.parse_args()


def train():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("fnoseg3d_train", log_dir=output_dir)

        # Choose device; if GPU has low free memory, fall back to CPU to avoid OOM
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    # Try using instance labels (instancesTr); fall back to semantic labels (labelsTr)
    try:
        dataset = nnUNetDataset(Path(args.data_root), split=args.split, use_instances=True)
    except Exception:
        dataset = nnUNetDataset(Path(args.data_root), split=args.split, use_instances=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    model = FNOSeg3DModel(in_channels=args.in_channels, out_channels=args.out_channels, width=args.width, modes=tuple(args.modes))
    model.to(device)

    if args.resume:
        model.load_weights(Path(args.resume))

    # Loss and optimizer
    if args.out_channels == 1:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_dice_total = 0.0
        running_samples = 0
        for batch_idx, sample in enumerate(loader, 1):
            img = sample["image"].to(device)
            mask = sample["mask"].to(device)

            # Ensure shapes: (B, C, X, Y, Z)
            if img.ndim == 4:
                img = img.unsqueeze(1)
            if mask.ndim == 4 and args.out_channels == 1:
                mask = mask.unsqueeze(1).float()

            optimizer.zero_grad()
                logits = model(img)

            if args.out_channels == 1:
                loss = criterion(logits, mask)
            else:
                loss = criterion(logits, mask.squeeze(1).long())

                loss.backward()
            optimizer.step()

            running_loss += float(loss.item())

            # Compute voxel-wise Dice for this batch
            with torch.no_grad():
                if args.out_channels == 1:
                    probs = torch.sigmoid(logits)
                    # mask: (B,1,X,Y,Z) float
                    target = mask.float()
                    # flatten spatial dims
                    dims = tuple(range(2, probs.ndim))
                    intersection = (probs * target).sum(dim=dims)
                    sums = probs.sum(dim=dims) + target.sum(dim=dims)
                    batch_dice = ((2.0 * intersection + 1e-6) / (sums + 1e-6)).mean().item()
                else:
                    # multiclass: compute mean Dice over foreground classes
                    probs = F.softmax(logits, dim=1)
                    # target expected shape: (B, X, Y, Z)
                    if mask.ndim == 5:
                        t = mask.squeeze(1).long()
                    else:
                        t = mask.long()
                    num_classes = args.out_channels
                    # one-hot target
                    t_onehot = F.one_hot(t, num_classes=num_classes).permute(0,4,1,2,3).float()
                    dims = tuple(range(2, probs.ndim))
                    intersection = (probs * t_onehot).sum(dim=dims)
                    sums = probs.sum(dim=dims) + t_onehot.sum(dim=dims)
                    # exclude background class (index 0) from averaging if present
                    if num_classes > 1:
                        per_class_dice = ((2.0 * intersection + 1e-6) / (sums + 1e-6))[:, 1:]
                        batch_dice = per_class_dice.mean().item()
                    else:
                        batch_dice = ((2.0 * intersection + 1e-6) / (sums + 1e-6)).mean().item()

            batch_size = img.shape[0]
            running_dice_total += batch_dice * batch_size
            running_samples += batch_size
            if batch_idx % 10 == 0:
                cur_avg_loss = running_loss / batch_idx
                cur_avg_dice = running_dice_total / running_samples if running_samples > 0 else 0.0
                logger.info(f"Epoch {epoch} [{batch_idx}/{len(loader)}] loss={cur_avg_loss:.4f} dice={cur_avg_dice:.4f}")

        avg_loss = running_loss / len(loader)
        avg_dice = running_dice_total / running_samples if running_samples > 0 else 0.0
        logger.info(f"Epoch {epoch} finished. avg_loss={avg_loss:.4f} avg_dice={avg_dice:.4f}")

        # Save checkpoint
        ckpt_path = output_dir / f"fnoseg3d_epoch_{epoch}.pt"
        model.save_weights(ckpt_path)

    logger.info("Training complete")


if __name__ == "__main__":
    train()
