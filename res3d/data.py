"""MitoEM 2.0 volumes as level pyramids, and random training windows of a fixed physical size.

Layout written by ``prepare`` (one folder per dataset):
    <out>/<dataset>/meta.json                 voxel size per level (nm, z y x), volume names, train / test split
    <out>/<dataset>/<volume>/img_L<k>.npy     uint8 image at level k (block average of L0)
    <out>/<dataset>/<volume>/fg_L<k>.npy      uint8 foreground at level k (majority of the block)
    <out>/<dataset>/<volume>/lab_L0.npy       uint32 instance labels at L0 (scoring reference)
All arrays are (z, y, x) and cropped to a multiple of the coarsest block, so every level covers the same tissue.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# block factors (z, y, x) per level: isotropic data is halved on every axis, 8 x 8 x 30 nm data only in y and x
LEVEL_FACTORS = {
    "iso": [(1, 1, 1), (2, 2, 2), (4, 4, 4)],
    "xy": [(1, 1, 1), (1, 2, 2), (1, 4, 4)],
}


def block_mean(a: np.ndarray, f) -> np.ndarray:
    """Average over (fz, fy, fx) blocks; the shape must be a multiple of f."""
    if tuple(f) == (1, 1, 1):
        return a.astype(np.float32)
    z, y, x = a.shape
    out = np.empty((z // f[0], y // f[1], x // f[2]), np.float32)
    step = max(1, 64 // f[0]) * f[0]  # in slabs to limit memory
    for z0 in range(0, z, step):
        s = a[z0:z0 + step].astype(np.float32)
        out[z0 // f[0]:(z0 + s.shape[0]) // f[0]] = s.reshape(s.shape[0] // f[0], f[0], y // f[1], f[1],
                                                               x // f[2], f[2]).mean((1, 3, 5))
    return out


def read_nifti(path: Path):
    """Array as (z, y, x) and voxel size (z, y, x) in the file's units (nm here)."""
    import nibabel as nib
    img = nib.load(str(path))
    a = np.asanyarray(img.dataobj)
    sx, sy, sz = (float(v) for v in img.header.get_zooms()[:3])
    return np.ascontiguousarray(np.transpose(a, (2, 1, 0))), (sz, sy, sx)


def prepare(raw_dir: str, out_dir: str, mode: str) -> dict:
    """Convert an nnU-Net style MitoEM 2.0 folder (imagesTr, labelsTr, imagesTs, labelsTs) into level pyramids."""
    raw, name = Path(raw_dir), Path(raw_dir).name
    out = Path(out_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    factors = LEVEL_FACTORS[mode]
    big = np.max(np.array(factors), axis=0)
    meta = {"dataset": name, "mode": mode, "factors": factors, "volumes": {}, "train": [], "test": []}
    for side, img_dir, lab_dir in (("train", "imagesTr", "labelsTr"), ("test", "imagesTs", "labelsTs")):
        for img_path in sorted((raw / img_dir).glob("*_0000.nii*")):
            vol = img_path.name.split("_0000.nii")[0]
            lab_path = next((raw / lab_dir).glob(f"{vol}.nii*"))
            img, spacing = read_nifti(img_path)
            lab, _ = read_nifti(lab_path)
            shape = tuple(int(s // b * b) for s, b in zip(img.shape, big))
            img = img[:shape[0], :shape[1], :shape[2]]
            lab = lab[:shape[0], :shape[1], :shape[2]].astype(np.uint32)
            d = out / vol
            d.mkdir(exist_ok=True)
            np.save(d / "lab_L0.npy", lab)
            fg = (lab > 0).astype(np.uint8)
            levels = []
            for k, f in enumerate(factors):
                np.save(d / f"img_L{k}.npy", np.clip(np.round(block_mean(img, f)), 0, 255).astype(np.uint8))
                np.save(d / f"fg_L{k}.npy", (block_mean(fg, f) >= 0.5).astype(np.uint8))
                levels.append({"shape": [s // fi for s, fi in zip(shape, f)],
                               "voxel_nm": [round(v * fi, 3) for v, fi in zip(spacing, f)]})
            meta["volumes"][vol] = {"levels": levels, "mean": float(img.mean()), "std": float(img.std()),
                                    "n_instances": int(len(np.unique(lab)) - 1), "fg_frac": float(fg.mean())}
            meta[side].append(vol)
            print(f"{name}/{vol} ({side}): L0 {shape} at {spacing} nm, {meta['volumes'][vol]['n_instances']} "
                  f"instances, foreground {100 * fg.mean():.1f} %", flush=True)
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    return meta


def load_meta(data_dir: str) -> dict:
    return json.load(open(Path(data_dir) / "meta.json"))


def window_size(meta: dict, level: int, train_level: int, train_window) -> tuple:
    """Voxels per axis of a window covering the same tissue as ``train_window`` voxels at the training level."""
    ft, fl = meta["factors"][train_level], meta["factors"][level]
    return tuple(int(round(w * a / b)) for w, a, b in zip(train_window, ft, fl))


class VolumeSampler:
    """Random training windows at one level: flips on every axis, 90-degree rotations in the (y, x) plane, brightness /
    contrast jitter; a share of the windows is centred on a foreground voxel."""

    def __init__(self, data_dir: str, level: int, window, fg_share: float = 0.67, seed: int = 0):
        self.meta = load_meta(data_dir)
        self.window = tuple(window)
        self.fg_share = fg_share
        self.rng = np.random.default_rng(seed)
        self.vols = []
        for v in self.meta["train"]:
            d = Path(data_dir) / v
            img, fg = np.load(d / f"img_L{level}.npy"), np.load(d / f"fg_L{level}.npy")
            st = self.meta["volumes"][v]
            pad = [(0, max(0, w - s)) for w, s in zip(self.window, img.shape)]
            if any(p[1] for p in pad):  # a volume thinner than the window (e.g. z): pad with the volume mean
                img = np.pad(img, pad, constant_values=int(round(st["mean"])))
                fg = np.pad(fg, pad)
            self.vols.append((img, fg, st["mean"], st["std"], np.argwhere(fg[::4, ::4, ::4]) * 4))

    def sample(self, batch: int):
        xs, ys = [], []
        for _ in range(batch):
            img, fg, mu, sd, fg_idx = self.vols[self.rng.integers(len(self.vols))]
            w = self.window
            if len(fg_idx) and self.rng.random() < self.fg_share:
                c = fg_idx[self.rng.integers(len(fg_idx))]
                start = [int(np.clip(ci - wi // 2 + self.rng.integers(-wi // 4, wi // 4 + 1), 0, s - wi))
                         for ci, wi, s in zip(c, w, img.shape)]
            else:
                start = [int(self.rng.integers(0, s - wi + 1)) for wi, s in zip(w, img.shape)]
            sl = tuple(slice(s0, s0 + wi) for s0, wi in zip(start, w))
            x = (img[sl].astype(np.float32) - mu) / sd
            y = fg[sl].astype(np.int64)
            for ax in range(3):
                if self.rng.random() < 0.5:
                    x, y = np.flip(x, ax), np.flip(y, ax)
            if w[1] == w[2]:
                r = int(self.rng.integers(4))
                x, y = np.rot90(x, r, (1, 2)), np.rot90(y, r, (1, 2))
            x = x * self.rng.uniform(0.9, 1.1) + self.rng.uniform(-0.1, 0.1)
            xs.append(np.ascontiguousarray(x))
            ys.append(np.ascontiguousarray(y))
        return np.stack(xs)[:, None], np.stack(ys)
