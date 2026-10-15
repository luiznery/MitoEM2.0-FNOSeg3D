#!/usr/bin/env python
"""Download MitoEM 2.0 datasets from Zenodo and write the level pyramids used by res3d.

    python -m res3d.prepare --out /data/res3d --zip-dir /data/zips Dataset001_ME2-Beta:iso Dataset008_ME2-Stem:xy

``iso``: 16 nm isotropic FIB-SEM, levels halve every axis; ``xy``: 8 x 8 x 30 nm, levels halve y and x only.
Datasets that already have a meta.json are skipped. ``--max-xy N`` keeps the central N x N region of wider volumes
(only the Pyra test volume, 4096 x 4096, is wider than 2048).
"""
from __future__ import annotations

import argparse
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from .data import prepare

URL = "https://zenodo.org/records/17635006/files/{}.zip?download=1"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("datasets", nargs="+", metavar="NAME:MODE")
    p.add_argument("--out", required=True)
    p.add_argument("--zip-dir", required=True)
    p.add_argument("--raw-dir", default="/tmp/raw")
    p.add_argument("--max-xy", type=int, default=None)
    p.add_argument("--force", action="store_true", help="prepare again even if meta.json exists")
    a = p.parse_args(argv)
    Path(a.zip_dir).mkdir(parents=True, exist_ok=True)
    for item in a.datasets:
        name, mode = item.split(":")
        if (Path(a.out) / name / "meta.json").exists():
            if not a.force:
                print(f"{name}: already prepared", flush=True)
                continue
            shutil.rmtree(Path(a.out) / name)
        z = Path(a.zip_dir) / f"{name}.zip"
        for attempt in range(6):
            if z.exists():
                break
            print(f"downloading {name}", flush=True)
            tmp = z.with_suffix(".part")
            try:
                with urllib.request.urlopen(URL.format(name), timeout=600) as r, open(tmp, "wb") as f:
                    shutil.copyfileobj(r, f, length=1 << 24)
                tmp.rename(z)
            except urllib.error.HTTPError as e:  # Zenodo rate limit
                if e.code != 429 or attempt == 5:
                    raise
                time.sleep(30 * (attempt + 1))
        zipfile.ZipFile(z).extractall(a.raw_dir)
        prepare(str(Path(a.raw_dir) / name), a.out, mode, a.max_xy)
        shutil.rmtree(Path(a.raw_dir) / name, ignore_errors=True)


if __name__ == "__main__":
    main()
