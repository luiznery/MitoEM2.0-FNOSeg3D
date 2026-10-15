#!/usr/bin/env python
"""Download MitoEM 2.0 datasets from Zenodo and write the level pyramids used by res3d.

    python -m res3d.prepare --out /data/res3d --zip-dir /data/zips Dataset001_ME2-Beta:iso Dataset008_ME2-Stem:xy

``iso``: 16 nm isotropic FIB-SEM, levels halve every axis; ``xy``: 8 x 8 x 30 nm, levels halve y and x only.
"""
from __future__ import annotations

import argparse
import shutil
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
    a = p.parse_args(argv)
    Path(a.zip_dir).mkdir(parents=True, exist_ok=True)
    for item in a.datasets:
        name, mode = item.split(":")
        z = Path(a.zip_dir) / f"{name}.zip"
        if not z.exists():
            print(f"downloading {name}", flush=True)
            tmp = z.with_suffix(".part")
            with urllib.request.urlopen(URL.format(name), timeout=600) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, length=1 << 24)
            tmp.rename(z)
        zipfile.ZipFile(z).extractall(a.raw_dir)
        prepare(str(Path(a.raw_dir) / name), a.out, mode)
        shutil.rmtree(Path(a.raw_dir) / name, ignore_errors=True)


if __name__ == "__main__":
    main()
