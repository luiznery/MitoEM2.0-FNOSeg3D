#!/usr/bin/env python
"""Test a trained 3D model at every level and score it against the L0 labels.

    python -m res3d.evaluate --data-dir /data/res3d/Dataset001_ME2-Beta --run-dir /runs/beta/e3k2 --levels 0 1 2

For each test volume and level: sliding windows that cover the same tissue as the training window (so more voxels at
finer levels), 25 % overlap, smooth blending; the foreground (argmax) is repeated up to the L0 grid and scored against
the L0 labels:
- Dice and IoU of the foreground;
- AP50 / AP75 = TP / (TP + FP + FN) with one-to-one (Hungarian) matching of instances by IoU, where the predicted
  instances are the 6-connected components of the foreground (components under 0.001 um^3 dropped).
The model's score at its training level is also the resize baseline at every other level (shrink the input to the
training voxel size, predict, upsample). Writes <run-dir>/eval/results.json, results.md and middle-slice overlays.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

from .data import load_meta, train_level_of, window_size
from .models import build_model


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--levels", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--overlap", type=float, default=0.25)
    p.add_argument("--min-um3", type=float, default=0.001, help="drop predicted components smaller than this")
    p.add_argument("--out-name", default="eval")
    p.add_argument("--input-lowpass", action="store_true",
                   help="diagnostic: Gaussian-smooth the input of finer levels (sigma = half the block factor to the "
                        "training level per axis), i.e. remove the detail and noise the model never saw in training")
    p.add_argument("--input-from-train-level", action="store_true",
                   help="diagnostic: at finer levels, feed the training-level image interpolated (linear) to the finer "
                        "grid: the same content on a finer grid, so only the grid changes")
    return p.parse_args(argv)


def starts(n: int, w: int, stride: int) -> list[int]:
    if n <= w:
        return [0]
    s = list(range(0, n - w + 1, stride))
    if s[-1] != n - w:
        s.append(n - w)
    return s


@torch.no_grad()
def predict(net, img: np.ndarray, window, overlap: float, mean: float, std: float, dev) -> np.ndarray:
    """Foreground probability by sliding windows with a smooth blending weight."""
    shape = img.shape
    w = [min(a, b) for a, b in zip(window, shape)]
    w = [x - x % 2 for x in w]  # even sizes for the stride-2 input layer / pooling
    stride = [max(1, int(round(x * (1 - overlap)))) for x in w]
    axes = [0.1 + 0.9 * np.sin(np.pi * (np.arange(x) + 0.5) / x) for x in w]
    wmap = (axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]).astype(np.float32)
    acc = np.zeros(shape, np.float32)
    wsum = np.zeros(shape, np.float32)
    for z in starts(shape[0], w[0], stride[0]):
        for y in starts(shape[1], w[1], stride[1]):
            for x in starts(shape[2], w[2], stride[2]):
                sl = (slice(z, z + w[0]), slice(y, y + w[1]), slice(x, x + w[2]))
                t = torch.from_numpy((img[sl].astype(np.float32) - mean) / std)[None, None].to(dev)
                prob = torch.softmax(net(t).float(), 1)[0, 1].cpu().numpy()
                acc[sl] += prob * wmap
                wsum[sl] += wmap
    np.maximum(wsum, 1e-6, out=wsum)
    acc /= wsum  # in place: the full-resolution arrays are large
    return acc


def upsample(a: np.ndarray, r) -> np.ndarray:
    for ax, k in enumerate(r):
        if k > 1:
            a = np.repeat(a, k, axis=ax)
    return a


def instance_scores(gt: np.ndarray, fg: np.ndarray, min_vox: int) -> dict:
    pr, n = ndimage.label(fg)
    sizes = np.bincount(pr.ravel())
    keep = sizes >= min_vox
    keep[0] = False
    if not keep[1:].all():  # drop small components, then number the rest again
        pr, n = ndimage.label(keep[pr])
        sizes = np.bincount(pr.ravel())
    gsz = np.bincount(gt.ravel())
    n_gt = int((gsz[1:] > 0).sum())
    both = (gt > 0) & (pr > 0)
    pairs, inter = np.unique(gt[both].astype(np.int64) * (n + 1) + pr[both], return_counts=True)
    gi, pi = pairs // (n + 1), pairs % (n + 1)
    iou = inter / (gsz[gi] + sizes[pi] - inter)
    out = {"n_gt": n_gt, "n_pred": int(n)}
    if n_gt == 0 or n == 0 or len(iou) == 0:
        for t in (50, 75):
            out[f"ap{t}"] = 1.0 if n_gt == n == 0 else 0.0
        return out
    ug, gidx = np.unique(gi, return_inverse=True)
    up, pidx = np.unique(pi, return_inverse=True)
    m = np.zeros((len(ug), len(up)), np.float32)
    m[gidx, pidx] = iou
    r, c = linear_sum_assignment(-m)
    matched = m[r, c]
    for t in (50, 75):
        tp = int((matched >= t / 100).sum())
        out[f"ap{t}"] = tp / (tp + (n - tp) + (n_gt - tp))
    return out


def overlay(img, fg, gt_fg, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z = img.shape[0] // 2
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img[z], cmap="gray")
    ax.contour(gt_fg[z], levels=[0.5], colors=["#00e05a"], linewidths=0.8)
    ax.contour(fg[z], levels=[0.5], colors=["#ff3030"], linewidths=0.8)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    a = parse_args(argv)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = Path(a.run_dir)
    cfg = json.load(open(run / "config.json"))
    meta = load_meta(a.data_dir)
    net = build_model(cfg["model"], cfg["filters"], cfg["blocks"], cfg["modes"], tuple(cfg["radii"]),
                      train_window=min(cfg["window"][1:]), unet_channels=cfg["unet_channels"],
                      input_layer=cfg.get("input_layer", "conv"), input_radius=cfg.get("input_radius", 2 / 64)).to(dev)
    net.load_state_dict(torch.load(run / "final.pt", map_location=dev, weights_only=False)["net"])
    net.eval()
    out = run / a.out_name
    out.mkdir(exist_ok=True)
    tl = train_level_of(cfg, meta)
    res = {"model": cfg["model"], "train_level": tl, "dataset": meta["dataset"], "levels": {}, "per_volume": {}}
    for v in meta["test"]:
        d = Path(a.data_dir) / v
        st = meta["volumes"][v]
        gt = np.load(d / "lab_L0.npy")
        gt_fg = gt > 0
        vox0 = np.prod(st["levels"][0]["voxel_nm"]) / 1e9  # um^3
        min_vox = max(1, int(a.min_um3 / vox0))
        res["per_volume"][v] = {}
        for lv in a.levels:
            t0 = time.time()
            img = np.load(d / f"img_L{lv}.npy")
            if a.input_from_train_level and lv < tl:
                src = np.load(d / f"img_L{tl}.npy").astype(np.float32)
                img = ndimage.zoom(src, [t / s for t, s in zip(img.shape, src.shape)], order=1, grid_mode=True,
                                   mode="nearest")
            if a.input_lowpass and lv < tl:
                r = [b // c for b, c in zip(meta["factors"][tl], meta["factors"][lv])]
                img = ndimage.gaussian_filter(img.astype(np.float32), [0.5 * x if x > 1 else 0 for x in r])
            win = window_size(meta, lv, tl, cfg["window"])
            prob = predict(net, img, win, a.overlap, st["mean"], st["std"], dev)
            ratio = [b // c for b, c in zip(meta["factors"][lv], meta["factors"][0])]
            fg = upsample(prob >= 0.5, ratio)
            inter = int((fg & gt_fg).sum())
            s = int(fg.sum() + gt_fg.sum())
            r = {"dice": 2 * inter / s if s else 1.0, "iou": inter / (s - inter) if s - inter else 1.0,
                 "window": list(win), "voxel_nm": st["levels"][lv]["voxel_nm"]}
            r.update(instance_scores(gt, fg, min_vox))
            r["seconds"] = round(time.time() - t0, 1)
            res["per_volume"][v][f"L{lv}"] = r
            print(f"{cfg['model']} {v} L{lv} ({r['voxel_nm']} nm, window {win}): Dice {r['dice']:.3f} IoU {r['iou']:.3f} "
                  f"AP50 {r['ap50']:.3f} AP75 {r['ap75']:.3f} ({r['n_pred']} pred / {r['n_gt']} gt), {r['seconds']} s",
                  flush=True)
            if v == meta["test"][0]:
                overlay(np.load(d / "img_L0.npy"), fg, gt_fg, out / f"overlay_{v}_L{lv}.png",
                        f"{cfg['model']} trained at L{tl}, tested at L{lv} {r['voxel_nm']} nm: "
                        f"Dice {r['dice']:.3f}")
            del prob, fg
    for lv in a.levels:
        k = f"L{lv}"
        res["levels"][k] = {m: float(np.mean([res["per_volume"][v][k][m] for v in meta["test"]]))
                            for m in ("dice", "iou", "ap50", "ap75")}
    json.dump(res, open(out / "results.json", "w"), indent=1)
    lines = ["| level | voxel (nm) | Dice | IoU | AP50 | AP75 |", "|---|---|---|---|---|---|"]
    for lv in a.levels:
        k = f"L{lv}"
        m = res["levels"][k]
        vn = res["per_volume"][meta["test"][0]][k]["voxel_nm"]
        lines.append(f"| {k}{' (train)' if lv == tl else ''} | {' x '.join(f'{x:g}' for x in vn)} | "
                     f"{m['dice']:.3f} | {m['iou']:.3f} | {m['ap50']:.3f} | {m['ap75']:.3f} |")
    (out / "results.md").write_text(f"# {cfg['model']} on {meta['dataset']} (trained at L{tl})\n\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
