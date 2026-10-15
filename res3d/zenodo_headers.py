#!/usr/bin/env python
"""Print every MitoEM 2.0 volume's size and voxel size straight from the Zenodo zips, reading only the zip directory
and the first bytes of each NIfTI (HTTP range requests), without downloading the archives.

    python -m res3d.zenodo_headers [Pyra Sperm ...]
"""
from __future__ import annotations

import io
import json
import struct
import sys
import time
import urllib.error
import urllib.request
import zipfile
import zlib

RECORD = "https://zenodo.org/api/records/17635006"


def fetch(req, tries: int = 8):
    """urlopen(...).read() with back-off on Zenodo's rate limit (HTTP 429)."""
    for i in range(tries):
        try:
            with urllib.request.urlopen(req) as r:
                return r.read(), r
        except urllib.error.HTTPError as e:
            if e.code != 429 or i == tries - 1:
                raise
            time.sleep(float(e.headers.get("Retry-After") or 10 * (i + 1)))


class HttpFile(io.RawIOBase):
    """A read-only, seekable file over HTTP range requests."""

    def __init__(self, url: str):
        self.url, self.pos = url, 0
        _, r = fetch(urllib.request.Request(url, headers={"Range": "bytes=0-0"}))
        self.url = r.geturl()
        self.size = int(r.headers["Content-Range"].split("/")[-1])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n <= 0 or self.pos >= self.size:
            return b""
        end = min(self.size, self.pos + n) - 1
        data, _ = fetch(urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"}))
        self.pos += len(data)
        return data

    def readinto(self, b):
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)


def nifti_header(zf: zipfile.ZipFile, name: str):
    """(dims x y z, voxel size x y z) from the first bytes of a .nii.gz inside the zip."""
    with zf.open(name) as fh:
        head = fh.read(8192)
    raw = zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(head, 400)
    dim = struct.unpack("<8h", raw[40:56])
    pix = struct.unpack("<8f", raw[76:108])
    return list(dim[1:4]), [round(p, 3) for p in pix[1:4]]


def main():
    rec = json.loads(fetch(RECORD)[0])
    only = sys.argv[1:]  # optional dataset names, e.g. Pyra Sperm
    for f in sorted(rec["files"], key=lambda f: f["key"]):
        if only and not any(o in f["key"] for o in only):
            continue
        url = f["links"]["self"]
        zf = zipfile.ZipFile(HttpFile(url))
        vols = [n for n in zf.namelist() if n.endswith(".nii.gz") and "/images" in n]
        print(f"DATASET {f['key']} {f['size'] / 1e9:.2f} GB, {len(vols)} image volumes", flush=True)
        for n in sorted(vols):
            dims, vox = nifti_header(zf, n)
            time.sleep(0.5)
            print(f"VOL {f['key']} {n} dims(x,y,z)={dims} voxel_nm(x,y,z)={vox}", flush=True)


if __name__ == "__main__":
    main()
