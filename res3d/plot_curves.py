#!/usr/bin/env python
"""Training curves of res3d runs (metrics.jsonl): loss, and Dice on fixed test windows at the training level and at L0.

    python -m res3d.plot_curves --out curves.png --title "Beta" --run "E3-3D + k2=/runs/beta/e3k2" "U-Net=/runs/beta/unet3d"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", nargs="+", required=True, metavar="NAME=DIR")
    p.add_argument("--out", required=True)
    p.add_argument("--title", default="")
    a = p.parse_args(argv)
    runs = [r.split("=", 1) for r in a.run]
    fig, axes = plt.subplots(2, len(runs), figsize=(4.2 * len(runs), 6.4), squeeze=False)
    for col, (name, d) in enumerate(runs):
        f = Path(d) / "metrics.jsonl"
        recs = [json.loads(line) for line in open(f)] if f.exists() else []
        cfg = json.load(open(Path(d) / "config.json")) if (Path(d) / "config.json").exists() else {}
        ep = [r["epoch"] for r in recs]
        ax = axes[0, col]
        ax.plot(ep, [r["train_loss"] for r in recs], color="#2a6fdb", label="train loss")
        ax.set_title(f"{name} ({cfg.get('n_params', 0):,} params, {len(recs)} epochs)", fontsize=9)
        ax.set_ylabel("loss" if col == 0 else "")
        ax.grid(alpha=0.3)
        ax = axes[1, col]
        ax.plot(ep, [r["val_dice"] for r in recs], color="#e8590c", label=f"test windows, training level L{cfg.get('level', '?')}")
        l0 = [(r["epoch"], r["val_dice_L0"]) for r in recs if "val_dice_L0" in r]
        if l0:
            ax.plot(*zip(*l0), "o-", color="#7b2fbf", ms=3, label="same windows at L0 (finest)")
        ax.set_ylim(0, 1)
        ax.set_xlabel("epoch (100 steps)")
        ax.set_ylabel("Dice" if col == 0 else "")
        ax.grid(alpha=0.3)
        if recs:
            last = recs[-1]
            txt = f"last: {last['val_dice']:.3f} at L{cfg.get('level', '?')}"
            if l0:
                txt += f", {l0[-1][1]:.3f} at L0"
            ax.text(0.98, 0.04, txt, transform=ax.transAxes, ha="right", fontsize=8)
        if col == 0:
            ax.legend(fontsize=7, loc="upper left")
    fig.suptitle(a.title, fontsize=11)
    fig.tight_layout()
    fig.savefig(a.out, dpi=150, bbox_inches="tight")
    print(f"written {a.out}")


if __name__ == "__main__":
    main()
