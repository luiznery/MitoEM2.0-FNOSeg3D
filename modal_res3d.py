"""Run the res3d (3D resolution-transfer) jobs on Modal.

    modal run modal_res3d.py::cpu --command "python -m res3d.prepare --out /data/res3d --zip-dir /data/zips Dataset001_ME2-Beta:iso"
    modal run modal_res3d.py::gpu --command "python -m res3d.train ..."
    modal deploy modal_res3d.py      # then spawn long jobs with modal.Function.from_name("res3d", "gpu_job").spawn(cmd)

Volumes: ``res3d-data`` (/data: zips and level pyramids) and ``res3d-runs`` (/runs: checkpoints, metrics, results).
The res3d package is sent with every run (redeploy after code changes). GPU: T4 unless RES3D_GPU is set.
"""
from __future__ import annotations

import os
import subprocess
import threading
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent
GPU = os.environ.get("RES3D_GPU", "T4")

app = modal.App("res3d")
data_vol = modal.Volume.from_name("res3d-data", create_if_missing=True)
runs_vol = modal.Volume.from_name("res3d-runs", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "numpy==1.26.4", "scipy==1.13.1", "nibabel==5.2.1", "matplotlib==3.9.2")
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_dir(str(REPO / "res3d"), "/root/repo/res3d", ignore=["**/__pycache__"])
)
VOLUMES = {"/data": data_vol, "/runs": runs_vol}


def _sh(command: str, commit_every: int = 60) -> None:
    """Run a shell command from the repo root; commit both volumes every minute and at the end (even on failure)."""
    print(f"$ {command}", flush=True)
    stop = threading.Event()

    def _commit():
        for v in (runs_vol, data_vol):
            try:
                v.commit()
            except Exception as e:  # an intermediate commit failure is not fatal
                print(f"volume commit failed: {e}", flush=True)

    def _periodic():
        while not stop.wait(commit_every):
            _commit()

    threading.Thread(target=_periodic, daemon=True).start()
    try:
        subprocess.run(["bash", "-lc", command], cwd="/root/repo", check=True)
    finally:
        stop.set()
        _commit()


@app.function(image=image, volumes=VOLUMES, cpu=4, memory=32768, timeout=6 * 3600)
def cpu_job(command: str) -> None:
    _sh(command)


@app.function(image=image, volumes=VOLUMES, gpu=GPU, cpu=4, memory=32768, timeout=12 * 3600, retries=2)
def gpu_job(command: str) -> None:
    """A training or evaluation command; training resumes from last.pt if the container is replaced."""
    runs_vol.reload()
    _sh(command)


@app.local_entrypoint()
def cpu(command: str):
    cpu_job.remote(command)


@app.local_entrypoint()
def gpu(command: str):
    gpu_job.remote(command)
