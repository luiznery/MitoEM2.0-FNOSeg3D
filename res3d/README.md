# res3d: train coarse, segment fine, in 3D

Port of the 2D resolution-transfer study (glomeruli in PAS histology, repo `fno_2d_segmentation`) to volume EM
mitochondria (MitoEM 2.0). A model is trained at a coarse voxel size and tested at every voxel size; predictions are
repeated up to the finest grid (L0) and scored against the L0 labels.

## Data (`res3d.prepare`, `res3d.data`)

| Dataset | Microscope | L0 voxel (nm) | Levels | Train / test volumes |
|---|---|---|---|---|
| Dataset001_ME2-Beta | FIB-SEM | 16 x 16 x 16 | L1 = 32, L2 = 64 nm (every axis halved) | 4 / 3 |
| Dataset008_ME2-Stem | SBF-SEM | 30 (z) x 8 x 8 | L1 = 30 x 16 x 16, L2 = 30 x 32 x 32 (y, x halved) | 2 / 1 |

Images are block-averaged, the foreground at each level is the majority of the block, the L0 instance labels are
kept for scoring. Training windows have a fixed physical size (96 voxels per axis at the training level); at test
time the windows cover the same tissue, so they have more voxels at finer levels.

## Models (`res3d.models`)

- `fnoseg3d`: IBM's FNOSeg (NeuralOperatorSeg, Fourier, shared weights) with its per-voxel 1x1 branch.
- `e3`: FNOSeg without the 1x1 branch + a physically sized local kernel (radius 4/64 of the window) per block.
- `e3k2`: `e3` + a second local kernel (radius 8/64). The kernels are trilinear interpolations of a 5^3 control grid,
  rebuilt for any grid, with quadrature weights, applied as zero-padded FFT convolutions (same result as
  `F.conv3d`, cost independent of the kernel size).
- `unet3d`: plain 3D U-Net (4 levels, 16-128 channels, instance norm, LeakyReLU).

IBM's networks are vendored in `res3d/ibm_nets` (Apache-2.0, unchanged).

## Training and evaluation

```bash
python -m res3d.prepare --out /data/res3d --zip-dir /data/zips Dataset001_ME2-Beta:iso Dataset008_ME2-Stem:xy
python -m res3d.train --data-dir /data/res3d/Dataset001_ME2-Beta --out-dir /runs/beta/e3k2 --model e3k2 \
    --filters 64 --modes 10 --level 1 --window 96 --steps 6000 --optimizer sam
python -m res3d.evaluate --data-dir /data/res3d/Dataset001_ME2-Beta --run-dir /runs/beta/e3k2 --levels 0 1 2
python -m res3d.plot_curves --out curves.png --run "E3-3D + k2=/runs/beta/e3k2"
python -m res3d.smoke --window 96   # FFT check, time and memory per model on random data
```

- Training: random windows (2/3 centred on foreground), flips, 90-degree rotations in (y, x), brightness / contrast
  jitter; cross-entropy + soft Dice; neural operators with SAM (rho 0.1) around AdamW 1e-3, warm-up + cosine,
  clipping 1.0; the U-Net with AdamW 3e-4. The curves use fixed test windows (never for model selection); the final
  weights are evaluated.
- Evaluation: sliding windows (25 % overlap, smooth blending) at each level; Dice / IoU of the foreground and
  AP50 / AP75 (TP / (TP + FP + FN), Hungarian matching by IoU, predicted instances = 6-connected components,
  components under 0.001 um^3 dropped). A model's score at its training level is also its resize baseline at the
  other levels.

`modal_res3d.py` runs everything on Modal (volumes `res3d-data`, `res3d-runs`).
