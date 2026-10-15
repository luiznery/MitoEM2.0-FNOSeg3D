#!/usr/bin/env python
"""Download MitoEM 2.0 datasets from Zenodo and write the level pyramids used by res3d.

    python -m res3d.prepare --out /data/res3d --zip-dir /data/zips Dataset001_ME2-Beta:iso Dataset008_ME2-Stem:xy

``iso``: 16 nm isotropic FIB-SEM, levels halve every axis; ``xy``: 8 x 8 x 30 nm, levels halve y and x only.
Datasets that already have a meta.json are skipped. ``--max-xy N`` keeps the central N x N region of wider volumes
(only the Pyra test volume, 4096 x 4096, is wider than 2048).
"""
from __future__ import annotations

import argparse
import http.client
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from .data import prepare

URL = "https://zenodo.org/records/17635006/files/{}.zip?download=1"


def download(url: str, dst: Path, attempts: int = 60) -> None:
    """Download with HTTP range requests, resuming after dropped connections (Zenodo resets long transfers)."""
    tmp = dst.with_suffix(".part")
    total = None
    for attempt in range(attempts):
        have = tmp.stat().st_size if tmp.exists() else 0
        if total is not None and have >= total:
            break
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                if r.status == 206:
                    total = int(r.headers["Content-Range"].split("/")[-1])
                else:  # the server sent the whole file
                    total = int(r.headers.get("Content-Length") or 0) or None
                    tmp.unlink(missing_ok=True)
                with open(tmp, "ab") as f:
                    shutil.copyfileobj(r, f, length=1 << 24)
            if total is None or tmp.stat().st_size >= total:
                break
        except (urllib.error.URLError, ConnectionError, OSError, http.client.HTTPException) as e:
            if isinstance(e, urllib.error.HTTPError) and e.code not in (429, 500, 502, 503, 504):
                raise
            done = tmp.stat().st_size if tmp.exists() else 0
            print(f"download interrupted at {done / 1e9:.2f} GB ({e}); resuming", flush=True)
            time.sleep(min(60, 5 * (attempt + 1)))
    else:
        raise RuntimeError(f"{url}: download did not finish after {attempts} attempts")
    tmp.rename(dst)
    print(f"downloaded {dst.name}: {dst.stat().st_size / 1e9:.2f} GB", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("datasets", nargs="+", metavar="NAME:MODE")
    p.add_argument("--out", required=True)
    p.add_argument("--zip-dir", required=True)
    p.add_argument("--raw-dir", default="/tmp/raw")
    p.add_argument("--max-xy", type=int, default=None)
    p.add_argument("--force", action="store_true", help="prepare again even if meta.json exists")
    p.add_argument("--crop", default="centre", choices=["centre", "labelled"],
                   help="--max-xy region: the central one or the one with the most labelled voxels")
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
        if not z.exists():
            print(f"downloading {name}", flush=True)
            download(URL.format(name), z)
        zipfile.ZipFile(z).extractall(a.raw_dir)
        prepare(str(Path(a.raw_dir) / name), a.out, mode, a.max_xy, a.crop)
        shutil.rmtree(Path(a.raw_dir) / name, ignore_errors=True)


if __name__ == "__main__":
    main()
