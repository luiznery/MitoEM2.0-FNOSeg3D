#!/usr/bin/env python
"""Train one 3D model at one level on random windows of a fixed physical size.

    python -m res3d.train --data-dir /data/res3d/Dataset001_ME2-Beta --out-dir /runs/beta/e3k2 --model e3k2 \
        --level 1 --window 96 --steps 3000 --optimizer sam

Several datasets at once: give several --data-dir and --train-nm (in-plane voxel size) instead of --level; each
dataset is used at its own level with that voxel size, and every window picks a dataset with equal odds.

Writes <out-dir>/config.json, metrics.jsonl (one line per epoch: train loss, lr, Dice on fixed test windows at the
training level and at L0), last.pt (resume) and final.pt. No checkpoint is selected on the test windows: they are
only used for the curves, as in the 2D v08 / v09 runs.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import MultiSampler, VolumeSampler, level_with_xy, load_meta
from .models import build_model, n_params


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", nargs="+", required=True, help="one or more prepared datasets")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--level", type=int, default=None, help="training level (0 = native)")
    p.add_argument("--train-nm", type=float, default=None,
                   help="instead of --level: train each dataset at its level with this in-plane voxel size (nm)")
    p.add_argument("--window", type=int, nargs="+", default=[96], help="window at the training level, voxels (z y x or one value)")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--steps-per-epoch", type=int, default=100)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--optimizer", default="sam", choices=["sam", "adamw"])
    p.add_argument("--sam-rho", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-min", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-frac", type=float, default=0.03)
    p.add_argument("--grad-clip", type=float, default=1.0, help="0 = off")
    p.add_argument("--filters", type=int, default=32)
    p.add_argument("--blocks", type=int, default=8)
    p.add_argument("--modes", type=int, default=8)
    p.add_argument("--radii", type=float, nargs="+", default=[4 / 64, 8 / 64])
    p.add_argument("--unet-channels", type=int, default=16)
    p.add_argument("--input-layer", default="conv", choices=["conv", "avgpool", "phys"],
                   help="neural operators: IBM's learned 2x2x2 stride-2 conv, 2x2x2 average + 1x1, or a physically sized kernel")
    p.add_argument("--input-radius", type=float, default=2 / 64, help="--input-layer phys: kernel radius (window units)")
    p.add_argument("--amp", action="store_true", help="fp16 autocast (U-Net only: the spectral layers need fp32)")
    p.add_argument("--val-windows", type=int, default=12, help="fixed test windows for the curves (per dataset when "
                   "there are several: --val-per-dataset)")
    p.add_argument("--val-per-dataset", type=int, default=2)
    p.add_argument("--val-l0-every", type=int, default=5, help="epochs between Dice at L0 on the same windows (0 = never)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def param_groups(net, weight_decay):
    params = [p for p in net.parameters() if p.requires_grad]
    return [{"params": [p for p in params if p.ndim > 1], "weight_decay": weight_decay},
            {"params": [p for p in params if p.ndim <= 1], "weight_decay": 0.0}]


def loss_fn(logits, y):
    p = torch.softmax(logits.float(), 1)[:, 1]
    t = (y == 1).float()
    inter = (p * t).sum()
    dice = 1 - (2 * inter + 1) / (p.sum() + t.sum() + 1)
    return F.cross_entropy(logits.float(), y) + dice


class ValWindows:
    """Fixed windows of the test volumes, the same tissue at the training level and at L0, for the curves."""

    def __init__(self, data_dir, meta, level, window, n, seed=123):
        rng = np.random.default_rng(seed)
        self.items = []
        self.level = level
        tests = meta["test"]
        for i in range(n):
            v = tests[i % len(tests)]
            d = Path(data_dir) / v
            st = meta["volumes"][v]
            imgs = {lv: np.load(d / f"img_L{lv}.npy", mmap_mode="r") for lv in {level, 0}}
            fgs = {lv: np.load(d / f"fg_L{lv}.npy", mmap_mode="r") for lv in {level, 0}}
            shp = imgs[level].shape
            w = [min(a, b) for a, b in zip(window, shp)]
            start = [int(rng.integers(0, s - wi + 1)) for s, wi in zip(shp, w)]
            ratio = [a // b for a, b in zip(meta["factors"][level], meta["factors"][0])]
            for lv in sorted({level, 0}):
                r = [1, 1, 1] if lv == level else ratio
                sl = tuple(slice(s0 * ri, (s0 + wi) * ri) for s0, wi, ri in zip(start, w, r))
                x = (np.asarray(imgs[lv][sl], np.float32) - st["mean"]) / st["std"]
                self.items.append((lv, x, np.asarray(fgs[lv][sl])))

    @torch.no_grad()
    def counts(self, net, dev, lv):
        inter = tot = 0
        for l, x, y in self.items:
            if l != lv:
                continue
            pred = net(torch.from_numpy(x)[None, None].to(dev)).argmax(1)[0].cpu().numpy()
            inter += int(((pred == 1) & (y == 1)).sum())
            tot += int((pred == 1).sum() + (y == 1).sum())
        return inter, tot

    def dice(self, net, dev, lv):
        inter, tot = self.counts(net, dev, lv)
        return 2 * inter / tot if tot else 1.0


def val_dice(vals, net, dev, which):
    """Dice of each dataset's windows and of all of them together; which = "train" (each dataset at its training
    level) or "L0"."""
    per = [v.counts(net, dev, v.level if which == "train" else 0) for v in vals]
    inter, tot = sum(i for i, _ in per), sum(t for _, t in per)
    return 2 * inter / tot if tot else 1.0, [round(2 * i / t, 4) if t else 1.0 for i, t in per]


def main(argv=None):
    a = parse_args(argv)
    if a.amp and a.optimizer == "sam":
        raise SystemExit("--amp is for the U-Net with AdamW; SAM runs in fp32")
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (a.level is None) == (a.train_nm is None):
        raise SystemExit("give exactly one of --level and --train-nm")
    metas = [load_meta(d) for d in a.data_dir]
    levels = [a.level if a.level is not None else level_with_xy(m, a.train_nm) for m in metas]
    window = tuple(a.window * 3 if len(a.window) == 1 else a.window)
    net = build_model(a.model, a.filters, a.blocks, a.modes, tuple(a.radii), train_window=min(window[1:]),
                      unet_channels=a.unet_channels, input_layer=a.input_layer, input_radius=a.input_radius).to(dev)
    cfg = {**vars(a), "window": window, "n_params": n_params(net)}
    cfg["train_levels"] = {m["dataset"]: lv for m, lv in zip(metas, levels)}
    cfg["train_voxel_nm"] = {m["dataset"]: m["volumes"][m["train"][0]]["levels"][lv]["voxel_nm"]
                             for m, lv in zip(metas, levels)}
    if len(metas) == 1:  # single-dataset runs keep the earlier config fields
        cfg["data_dir"], cfg["level"] = a.data_dir[0], levels[0]
        cfg["train_voxel_nm"] = cfg["train_voxel_nm"][metas[0]["dataset"]]
        cfg["window_nm"] = [w * v for w, v in zip(window, cfg["train_voxel_nm"])]
    json.dump(cfg, open(out / "config.json", "w"), indent=1)
    for m, lv in zip(metas, levels):
        vox = m["volumes"][m["train"][0]]["levels"][lv]["voxel_nm"]
        print(f"{m['dataset']}: level {lv}, voxel {vox} nm, window {[round(w * v) for w, v in zip(window, vox)]} nm, "
              f"{len(m['train'])} training volumes", flush=True)
    print(f"{a.model}: {cfg['n_params']:,} parameters", flush=True)

    samplers = [VolumeSampler(d, lv, window, seed=a.seed) for d, lv in zip(a.data_dir, levels)]
    sampler = samplers[0] if len(samplers) == 1 else MultiSampler(samplers, a.seed)
    nval = a.val_windows if len(metas) == 1 else a.val_per_dataset
    vals = [ValWindows(d, m, lv, window, nval) for d, m, lv in zip(a.data_dir, metas, levels)]
    base = torch.optim.AdamW(param_groups(net, a.weight_decay), lr=a.lr)
    warm = max(1, int(a.warmup_frac * a.steps))

    def lr_at(step):
        if step < warm:
            return a.lr * (step + 1) / warm
        t = (step - warm) / max(1, a.steps - warm)
        return a.lr_min + 0.5 * (a.lr - a.lr_min) * (1 + math.cos(math.pi * t))

    scaler = torch.amp.GradScaler("cuda", enabled=a.amp and dev.type == "cuda")
    step, epoch = 0, 0
    ck = out / "last.pt"
    if ck.exists():
        st = torch.load(ck, map_location=dev, weights_only=False)
        net.load_state_dict(st["net"])
        base.load_state_dict(st["opt"])
        step, epoch = st["step"], st["epoch"]
        if isinstance(sampler, MultiSampler):
            sampler.reseed(a.seed + step)
        else:
            sampler.rng = np.random.default_rng(a.seed + step)
        print(f"resumed at step {step}", flush=True)

    def forward_backward(x, y):
        with torch.autocast(dev.type, dtype=torch.float16, enabled=a.amp and dev.type == "cuda"):
            logits = net(x)
        loss = loss_fn(logits, y)
        scaler.scale(loss).backward()
        return loss

    t0 = time.time()
    while step < a.steps:
        net.train()
        losses = []
        for _ in range(min(a.steps_per_epoch, a.steps - step)):
            for g in base.param_groups:
                g["lr"] = lr_at(step)
            xb, yb = sampler.sample(a.batch)
            x, y = torch.from_numpy(xb).to(dev), torch.from_numpy(yb).to(dev)
            base.zero_grad(set_to_none=True)
            loss = forward_backward(x, y)
            if a.optimizer == "sam":  # gradient at w + rho g / |g|, then the AdamW step at w with it
                scaler.unscale_(base)
                grads = [p.grad for g in base.param_groups for p in g["params"] if p.grad is not None]
                norm = torch.norm(torch.stack([gr.norm(2) for gr in grads]), 2)
                eps = []
                with torch.no_grad():
                    for g in base.param_groups:
                        for p in g["params"]:
                            if p.grad is None:
                                continue
                            e = p.grad * (a.sam_rho / (norm + 1e-12))
                            p.add_(e)
                            eps.append((p, e))
                base.zero_grad(set_to_none=True)
                forward_backward(x, y)
                with torch.no_grad():
                    for p, e in eps:
                        p.sub_(e)
            if a.optimizer != "sam":
                scaler.unscale_(base)
            if a.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), a.grad_clip)
            scaler.step(base)
            scaler.update()
            losses.append(float(loss))
            step += 1
        epoch += 1
        net.eval()
        dice_t, per_t = val_dice(vals, net, dev, "train")
        rec = {"epoch": epoch, "step": step, "train_loss": float(np.mean(losses)), "lr": lr_at(step - 1),
               "val_dice": dice_t, "minutes": round((time.time() - t0) / 60, 2)}
        if len(vals) > 1:
            rec["val_by_dataset"] = {m["dataset"]: {"train": d} for m, d in zip(metas, per_t)}
        if any(levels) and a.val_l0_every and (epoch % a.val_l0_every == 0 or step >= a.steps):
            rec["val_dice_L0"], per_0 = val_dice(vals, net, dev, "L0")
            if len(vals) > 1:
                for m, d in zip(metas, per_0):
                    rec["val_by_dataset"][m["dataset"]]["L0"] = d
        with open(out / "metrics.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
        torch.save({"net": net.state_dict(), "opt": base.state_dict(), "step": step, "epoch": epoch}, ck)
    torch.save({"net": net.state_dict(), "config": cfg}, out / "final.pt")
    json.dump({"steps": step, "minutes": round((time.time() - t0) / 60, 1)}, open(out / "train_done.json", "w"))


if __name__ == "__main__":
    main()
