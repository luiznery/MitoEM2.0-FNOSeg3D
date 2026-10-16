#!/usr/bin/env python
"""Export a test crop for the slice viewer: the image at every level (JPEG atlases of the z-slices), the L0 ground truth
and every run's thresholded prediction at its own level (bit-packed, gzip), plus a manifest.

    python -m res3d.export_viewer --data-dir /data/res3d/Dataset008_ME2-Stem --out /runs/viewer/stem \
        --crop 100 512 512 --runs /runs/stem/e3k2 /runs/stem/fnoseg3d /runs/stem/unet3d

The crop (L0 voxels, z y x, multiples of the coarsest block) is placed where the most ground-truth mitochondria are,
with 8-35 % foreground. Predictions are made by the same sliding windows as res3d.evaluate on the crop plus half a
window of context on each side, then cut to the crop.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .data import load_meta, train_level_of, window_size
from .evaluate import predict, upsample
from .models import QUADRATURES, build_model, set_quadrature


def choose_crop(lab: np.ndarray, size, block) -> list[int]:
    """Start (z, y, x) of the crop with the most ground-truth instances among crops with 8-35 % foreground."""
    sub = (2, 8, 8) if lab.shape[0] > size[0] else (1, 8, 8)
    small = np.ascontiguousarray(lab[::sub[0], ::sub[1], ::sub[2]])
    steps = [max(b, (s // 4) // b * b) for s, b in zip(size, block)]
    best, best_start = -1, [0, 0, 0]
    for z in range(0, lab.shape[0] - size[0] + 1, steps[0]):
        for y in range(0, lab.shape[1] - size[1] + 1, steps[1]):
            for x in range(0, lab.shape[2] - size[2] + 1, steps[2]):
                c = small[z // sub[0]:(z + size[0]) // sub[0], y // sub[1]:(y + size[1]) // sub[1],
                          x // sub[2]:(x + size[2]) // sub[2]]
                fg = float((c > 0).mean())
                if not 0.08 <= fg <= 0.35:
                    continue
                n = len(np.unique(c)) - 1
                if n > best:
                    best, best_start = n, [z, y, x]
    print(f"crop start {best_start}, about {best} instances", flush=True)
    return best_start


def atlases(stack: np.ndarray, out: Path, stem: str, max_side: int = 2560, quality: int = 72, fmt: str = "webp") -> dict:
    """z-slices tiled row-major into grayscale image atlases (WebP or JPEG) of at most max_side pixels per side."""
    n, h, w = stack.shape
    cols = max(1, min(max_side // w, int(np.ceil(np.sqrt(n)))))
    rows = max(1, min(max_side // h, int(np.ceil(n / cols))))
    per = cols * rows
    files = []
    for i, s in enumerate(range(0, n, per)):
        part = stack[s:s + per]
        r = int(np.ceil(len(part) / cols))
        canvas = np.zeros((r * h, cols * w), np.uint8)
        for j, sl in enumerate(part):
            canvas[(j // cols) * h:(j // cols + 1) * h, (j % cols) * w:(j % cols + 1) * w] = sl
        name = f"{stem}_{i}.{fmt}"
        Image.fromarray(canvas).save(out / name, quality=quality, **({"method": 6} if fmt == "webp" else {}))
        files.append(name)
    return {"files": files, "cols": cols, "per": per, "w": w, "h": h, "n": n}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--crop", type=int, nargs=3, required=True, help="crop size in L0 voxels (z y x)")
    p.add_argument("--runs", nargs="+", required=True,
                   help="run directories, optionally NAME=DIR and DIR:QUADRATURE (default --kernel-quadrature)")
    p.add_argument("--volume", default=None, help="test volume (default: the first)")
    p.add_argument("--image-format", default="webp", choices=["webp", "jpg"])
    p.add_argument("--quality", type=int, default=72)
    p.add_argument("--kernel-quadrature", default="point", choices=QUADRATURES)
    p.add_argument("--images-only", action="store_true",
                   help="rewrite only the image atlases of an existing export (same crop, predictions kept)")
    a = p.parse_args(argv)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = load_meta(a.data_dir)
    v = a.volume or meta["test"][0]
    d = Path(a.data_dir) / v
    st = meta["volumes"][v]
    factors = meta["factors"]
    block = np.max(np.array(factors), axis=0)
    lab = np.load(d / "lab_L0.npy", mmap_mode="r")
    out = Path(a.out)
    if a.images_only:
        man = json.load(open(out / "manifest.json"))
        start, size = man["crop_start"], man["crop_size"]
        for old in out.glob("em_L*"):
            old.unlink()
        for k, f in enumerate(factors):
            img = np.load(d / f"img_L{k}.npy", mmap_mode="r")
            slk = tuple(slice(s // b, (s + n) // b) for s, n, b in zip(start, size, f))
            man["levels"][f"L{k}"] = atlases(np.asarray(img[slk]), out, f"em_L{k}", quality=a.quality, fmt=a.image_format)
        json.dump(man, open(out / "manifest.json", "w"), indent=1)
        print(f"rewrote images in {out}: {sorted((p.name, p.stat().st_size) for p in out.iterdir())}", flush=True)
        return
    size = [int(min(c, s) // b * b) for c, s, b in zip(a.crop, lab.shape, block)]
    start = [int(s) for s in choose_crop(lab, size, block)]
    out.mkdir(parents=True, exist_ok=True)

    man = {"dataset": meta["dataset"], "volume": v, "crop_start": start, "crop_size": size, "factors": factors,
           "voxel_nm": [lv["voxel_nm"] for lv in st["levels"]], "levels": {}, "masks": {}, "dice": {}, "runs": {}}
    blobs, offset = [], 0

    def add_mask(key, m):
        nonlocal offset
        b = np.packbits(m.astype(bool).ravel()).tobytes()
        man["masks"][key] = {"offset": offset, "shape": list(m.shape)}
        blobs.append(b)
        offset += len(b)

    sl0 = tuple(slice(s, s + n) for s, n in zip(start, size))
    gt = np.asarray(lab[sl0]) > 0
    add_mask("gt", gt)
    man["gt_instances"] = int(len(np.unique(np.asarray(lab[sl0]))) - 1)
    for k, f in enumerate(factors):
        img = np.load(d / f"img_L{k}.npy", mmap_mode="r")
        slk = tuple(slice(s // b, (s + n) // b) for s, n, b in zip(start, size, f))
        man["levels"][f"L{k}"] = atlases(np.asarray(img[slk]), out, f"em_L{k}", quality=a.quality, fmt=a.image_format)
    short = meta["dataset"].split("ME2-")[-1].lower()
    for spec in a.runs:
        name, _, path = spec.rpartition("=")
        path, _, quad = path.partition(":")
        run = Path(path)
        name = name or run.name
        cfg = json.load(open(run / "config.json"))
        net = build_model(cfg["model"], cfg["filters"], cfg["blocks"], cfg["modes"], tuple(cfg["radii"]),
                          train_window=min(cfg["window"][1:]), unet_channels=cfg["unet_channels"],
                          input_layer=cfg.get("input_layer", "conv"),
                          input_radius=cfg.get("input_radius", 2 / 64)).to(dev).eval()
        net.load_state_dict(torch.load(run / "final.pt", map_location=dev, weights_only=False)["net"])
        set_quadrature(net, quad or a.kernel_quadrature)
        man["runs"][name] = {"model": cfg["model"], "train_level": train_level_of(cfg, meta),
                             "kernel_quadrature": quad or a.kernel_quadrature}
        res = run / f"eval_{short}" / "results.json"
        if not res.exists():
            res = run / "eval" / "results.json"
        if res.exists():
            r = json.load(open(res))
            man["runs"][name]["test_dice"] = {k: r["per_volume"][v][k]["dice"] for k in r["per_volume"][v]}
        man["dice"][name] = {}
        for k, f in enumerate(factors):
            img = np.load(d / f"img_L{k}.npy", mmap_mode="r")
            win = window_size(meta, k, train_level_of(cfg, meta), cfg["window"])
            s_k = [s // b for s, b in zip(start, f)]
            n_k = [n // b for n, b in zip(size, f)]
            lo = [max(0, s - w // 2) for s, w in zip(s_k, win)]
            hi = [min(dim, s + n + w // 2) for s, n, w, dim in zip(s_k, n_k, win, img.shape)]
            region = np.asarray(img[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
            prob = predict(net, region, win, 0.25, st["mean"], st["std"], dev)
            cut = prob[s_k[0] - lo[0]:s_k[0] - lo[0] + n_k[0], s_k[1] - lo[1]:s_k[1] - lo[1] + n_k[1],
                       s_k[2] - lo[2]:s_k[2] - lo[2] + n_k[2]] >= 0.5
            add_mask(f"{name}/L{k}", cut)
            up = upsample(cut, f)
            inter = int((up & gt).sum())
            tot = int(up.sum() + gt.sum())
            man["dice"][name][f"L{k}"] = round(2 * inter / tot, 4) if tot else 1.0
            print(f"{name} L{k}: crop Dice {man['dice'][name][f'L{k}']:.3f}", flush=True)
        net = None
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    with gzip.open(out / "masks.bin.gz", "wb", compresslevel=9) as fh:
        for b in blobs:
            fh.write(b)
    json.dump(man, open(out / "manifest.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"wrote {out}: {sorted(p.name for p in out.iterdir())}", flush=True)


if __name__ == "__main__":
    main()
