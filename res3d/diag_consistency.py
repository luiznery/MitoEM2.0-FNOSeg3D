#!/usr/bin/env python
"""Where does a trained model stop being resolution-consistent? Feed the same content on the training grid and on a
finer grid (the training-level image interpolated by --zoom per axis), record every layer's output, average the
finer one back to the training grid and print the relative difference per layer.

    python -m res3d.diag_consistency --data-dir /data/res3d/Dataset008_ME2-Stem --run-dir /runs/stem/fnoseg3d --zoom 1 4 4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from .data import load_meta, train_level_of
from .models import build_model


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--zoom", type=int, nargs=3, required=True)
    p.add_argument("--ckpt", default="final.pt")
    p.add_argument("--windows", type=int, default=3)
    a = p.parse_args(argv)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = Path(a.run_dir)
    cfg = json.load(open(run / "config.json"))
    meta = load_meta(a.data_dir)
    net = build_model(cfg["model"], cfg["filters"], cfg["blocks"], cfg["modes"], tuple(cfg["radii"]),
                      train_window=min(cfg["window"][1:]), unet_channels=cfg["unet_channels"],
                      input_layer=cfg.get("input_layer", "conv"), input_radius=cfg.get("input_radius", 2 / 64)).to(dev).eval()
    net.load_state_dict(torch.load(run / a.ckpt, map_location=dev, weights_only=False)["net"])
    names = ["conv_in", "conv1"] + [f"layers.{i}" for i in range(cfg["blocks"])] + ["conv_out"]
    mods = dict(net.named_modules())
    names = [n for n in names if n in mods]
    acts = {}

    def hook(name):
        def f(_, __, out):
            acts[name] = out.detach().float()
        return f
    for n in names:
        mods[n].register_forward_hook(hook(n))
    v = meta["test"][0]
    st = meta["volumes"][v]
    img = np.load(Path(a.data_dir) / v / f"img_L{train_level_of(cfg, meta)}.npy")
    rng = np.random.default_rng(0)
    w = cfg["window"]
    diffs = {n: [] for n in names}
    for _ in range(a.windows):
        s = [int(rng.integers(0, max(1, d - wi + 1))) for d, wi in zip(img.shape, w)]
        x = (img[s[0]:s[0] + w[0], s[1]:s[1] + w[1], s[2]:s[2] + w[2]].astype(np.float32) - st["mean"]) / st["std"]
        xf = ndimage.zoom(x, a.zoom, order=1, grid_mode=True, mode="nearest")
        with torch.no_grad():
            net(torch.from_numpy(x)[None, None].to(dev))
            ref = {n: acts[n] for n in names}
            net(torch.from_numpy(xf)[None, None].to(dev))
        for n in names:
            fine = acts[n]
            back = F.adaptive_avg_pool3d(fine, ref[n].shape[-3:]) if fine.shape != ref[n].shape else fine
            diffs[n].append(float((back - ref[n]).norm() / ref[n].norm().clamp_min(1e-12)))
    print(f"{cfg['model']} ({a.run_dir}), zoom {a.zoom}: relative difference of each layer's output, finer grid "
          f"averaged back vs training grid (same content)")
    for n in names:
        print(f"  {n:10s} {np.mean(diffs[n]):.3f}")


if __name__ == "__main__":
    main()
