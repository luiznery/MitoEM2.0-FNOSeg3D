#!/usr/bin/env python
"""Smoke test on random data: the 3D FFT local kernel against F.conv3d, parameters, time and peak memory of a training
step (with SAM: two forward-backward passes) and of inference windows, for each model.

    python -m res3d.smoke --window 96 --batch 2 --infer 48 96 192 96x384x384
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from .models import MODEL_NAMES, PhysLocalConv3d, build_model, depthwise_conv3d_fft, n_params
from .train import loss_fn


def timed(fn, reps: int, dev) -> float:
    for _ in range(2):
        fn()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--window", type=int, default=96)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--infer", nargs="+", default=["48", "96", "192", "96x384x384"])
    p.add_argument("--models", nargs="+", default=list(MODEL_NAMES))
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--filters", type=int, default=32)
    p.add_argument("--modes", type=int, default=8)
    p.add_argument("--skip-check", action="store_true")
    p.add_argument("--input-layer", default="conv")
    a = p.parse_args(argv)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu'}")
    torch.manual_seed(0)
    for size in () if a.skip_check else ((24, 24, 24), (48, 48, 48), (30, 40, 50)):
        conv = PhysLocalConv3d(8, 8 / 64, ref_size=48).to(dev)
        with torch.no_grad():
            conv.weight.normal_()
        x = torch.randn(1, 8, *size, device=dev)
        k = conv.kernel(size)
        d = F.conv3d(x, k, padding=tuple(s // 2 for s in k.shape[-3:]), groups=8)
        f = depthwise_conv3d_fft(x, k)
        print(f"FFT check grid {size}, kernel {tuple(k.shape[-3:])}: rel. diff {float((f - d).abs().max() / d.abs().max()):.1e}")
    w = a.window
    for name in a.models:
        net = build_model(name, filters=a.filters, modes=a.modes, train_window=w, input_layer=a.input_layer).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
        x = torch.randn(a.batch, 1, w, w, w, device=dev)
        y = (torch.rand(a.batch, w, w, w, device=dev) > 0.8).long()

        def step():
            opt.zero_grad(set_to_none=True)
            loss_fn(net(x), y).backward()
            opt.step()
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t = timed(step, a.reps, dev)
        mem = torch.cuda.max_memory_allocated() / 2 ** 30 if dev.type == "cuda" else float("nan")
        print(f"{name}: {n_params(net):,} params; train step (batch {a.batch}, {w}^3) {t:.3f} s fwd+bwd, "
              f"x2 with SAM; peak {mem:.2f} GiB", flush=True)
        net.eval()
        for spec in a.infer:
            shp = tuple(int(v) for v in spec.split("x")) if "x" in spec else (int(spec),) * 3
            xi = torch.randn(1, 1, *shp, device=dev)
            if dev.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            try:
                with torch.no_grad():
                    ti = timed(lambda: net(xi), 2, dev)
                mem = torch.cuda.max_memory_allocated() / 2 ** 30 if dev.type == "cuda" else float("nan")
                print(f"  inference {shp}: {ti:.3f} s, peak {mem:.2f} GiB", flush=True)
            except torch.cuda.OutOfMemoryError:
                print(f"  inference {shp}: out of memory", flush=True)
            xi = None
        net = opt = None
        if dev.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
