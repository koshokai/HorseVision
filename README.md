# Horse Vision — Panoramic Depth from a Dual-Fisheye Rig

Feed-forward depth estimation for a **back-to-back dual-fisheye camera**. The two
fisheye views are predicted independently by a shared Depth-Anything-V2 (DAv2)
encoder, fused into an equirectangular panorama by geometric splatting, and then
refined by a lightweight ViT (`MiniDA3`).

Self-supervision comes from the **overlapping field of view** between the two
fisheye lenses: each view is warped into the other using the predicted depth, and
the photometric agreement yields a per-pixel **confidence map** that is also
exported for inspection.

| | |
|---|---|
| Input | 2 × fisheye RGB (518 × 518) |
| Output | Panoramic depth (518 × 1036) + per-view fisheye depth + confidence maps |
| Peak GPU memory | **1.14 GB** |
| End-to-end runtime | **3.9 s** (including model loading, RTX 4090) |
| Scenes | `indoor` (max depth 20 m) · `outdoor` (max depth 80 m) |

---

## 1. Requirements

Tested configuration:

| Package | Version |
|---|---|
| Python | 3.9.19 |
| PyTorch | 2.6.0 (CUDA 12.4) |
| timm | 0.9.16 |
| opencv-python | 4.11.0 |
| numpy | 1.23.5 |
| Pillow | 10.2.0 |

```bash
conda create -n horse_vision python=3.9
conda activate horse_vision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install timm==0.9.16 opencv-python numpy==1.23.5 Pillow
```

A CUDA GPU is required. Memory demand is modest (~1.2 GB), so any card from a
GTX 1060 upward should be sufficient.

---

## 2. Setting up Depth-Anything-V2 (required)

**This repository does not bundle Depth-Anything-V2.** The backbone is imported at
runtime, so DAv2 must be available before any script will start.

### 2.1 Obtain DAv2

```bash
git clone https://github.com/DepthAnything/Depth-Anything-V2.git
```

The directory must contain the importable package `depth_anything_v2/`.

### 2.2 Make it discoverable

`depthnet.py` looks for DAv2 in two locations, in order:

```python
possible_paths = [
    Path('set/your/path/Depth-Anything-V2'),   # ← edit this line
    current_dir / 'Depth-Anything-V2',         # ← or place DAv2 here
]
```

Pick whichever is more convenient:

**Option A — place it inside this repository** (no code change):

```bash
cd horse_vision_code
git clone https://github.com/DepthAnything/Depth-Anything-V2.git
# or, if you already have a copy elsewhere:
ln -s /path/to/Depth-Anything-V2 ./Depth-Anything-V2
```

**Option B — edit the path.** Replace `'set/your/path/Depth-Anything-V2'` on
line 14 of `depthnet.py` with your absolute path.

On success you will see:

```
[INFO] Found Depth-Anything-V2: /path/to/Depth-Anything-V2
[INFO] ✓ DepthAnythingV2 imported successfully!
```

Cloning the repository is enough to get started — the official DAv2 *checkpoints*
are only needed if you want to run the unmodified DAv2 baseline. See §5.2.

---

## 3. Repository layout

```
horse_vision_code/
├── test_single_conf.py       ← main test script (start here)
├── test_single_outdoor.py    ← variant without confidence maps, needs GT panorama
├── download_models.py        fetches the checkpoints from Releases (§4)
├── models_sha256.txt         checksums for the six checkpoints
├── depthnet.py               DAv2 wrapper, shared encoder, checkpoint discovery
├── networks.py               DepthAnythingV2FisheyeNet
├── cross_atte_ori.py         MiniDA3_ViTBase — panoramic refinement network
├── repro_tensor.py           FisheyeSelfSupervisedLoss — warping + confidence
├── dataloaders/
│   └── dataset_test_new.py
├── models/                   ← checkpoints, see §4
│   ├── indoor/
│   └── outdoor/
├── test_imgs/                sample inputs
│   ├── indoor_imgs/{fish,pano}/{left,right}/
│   └── occ_img/{occ_fish,occ_erp}/{left,right}/
└── Depth-Anything-V2/        ← you provide this (§2)
```

---

## 4. Model zoo

Six checkpoints, **2.2 GB total**, published under
[Releases](https://github.com/koshokai/HorseVision/releases).

### 4.1 Download

```bash
python download_models.py                  # all six (2.2 GB)
python download_models.py --scene outdoor  # only what test_single_conf.py needs
python download_models.py --verify         # re-check files already on disk
```

The script uses only the Python standard library, places each file at the right
path, and verifies its SHA-256. Files already present and intact are skipped, so
it is safe to re-run after an interrupted download.

To fetch a single file by hand instead:

```bash
mkdir -p models/outdoor/self_supervised_fish_carla
wget -O models/outdoor/self_supervised_fish_carla/self_fish.tar \
  https://github.com/koshokai/HorseVision/releases/download/v1.0/outdoor_self_supervised_fish_carla_self_fish.tar
```

Checksums for all six files are in `models_sha256.txt`:

```bash
cd models && sha256sum -c ../models_sha256.txt
```

### 4.2 Layout

```
models/
├── indoor/                              max depth 20 m
│   ├── supervised_st3D/sup_fish.tar             372 MB   fisheye, supervised (Structured3D)
│   ├── self_supervised_fish_hm3d/self_fish.tar  373 MB   fisheye, self-supervised (HM3D)
│   └── supervised_erp/sup_erp.pth               355 MB   MiniDA3 panoramic refiner
└── outdoor/                             max depth 80 m
    ├── supervised_fish_carla/sup_fish.tar       372 MB   fisheye, supervised (CARLA)
    ├── self_supervised_fish_carla/self_fish.tar 373 MB   fisheye, self-supervised (CARLA)
    └── supervised_erp/sup_erp.pth               355 MB   MiniDA3 panoramic refiner
```

Release assets are named with the directory prefix flattened — for example
`models/outdoor/supervised_erp/sup_erp.pth` is published as
`outdoor_supervised_erp_sup_erp.pth`.

Each scene provides **two fisheye variants**:

- `supervised_*` — trained with ground-truth depth.
- `self_supervised_*` — trained only from cross-view photometric consistency
  (no depth labels). This is the variant used by default in `test_single_conf.py`.

The `supervised_erp/sup_erp.pth` refiner is shared by both variants.

---

## 5. Quick start

```bash
conda activate horse_vision
cd horse_vision_code
python test_single_conf.py
```

This writes `test_result_integrated.png` to the current directory.

### 5.1 Configuration

All settings are at the top of `test_single_conf.py` (lines 13–24):

```python
scene = 'outdoor'                 # 'indoor' or 'outdoor'
MAX_DEPTH = 80.0                  # 20.0 for indoor, 80.0 for outdoor
PRETRAINED_FISH    = "models/outdoor/self_supervised_fish_carla/self_fish.tar"
CHECKPOINT_MINIDA3 = "models/outdoor/supervised_erp/sup_erp.pth"
paths = {
    'l_rgb':      "test_imgs/occ_img/occ_fish/left/0_rgb.png",
    'r_rgb':      "test_imgs/occ_img/occ_fish/right/0_rgb.png",
    'l_depth':    "test_imgs/occ_img/occ_fish/left/0_depth.npy",
    'r_depth':    "test_imgs/occ_img/occ_fish/right/0_depth.npy",
    'pano_depth': "test_imgs/occ_img/occ_erp/left/0_pano_depth.npy",
}
```

To switch to indoor, change all four together:

```python
scene = 'indoor'
MAX_DEPTH = 20.0
PRETRAINED_FISH    = "models/indoor/self_supervised_fish_hm3d/self_fish.tar"
CHECKPOINT_MINIDA3 = "models/indoor/supervised_erp/sup_erp.pth"
paths = { ... "test_imgs/indoor_imgs/fish/left/..." ... }
```

`MAX_DEPTH` must match the scene: the network regresses metric depth and the
visualisation clips to this range.

### 5.2 Which weights to download, and where to put them

Two sets of weights are involved. Download both, place them as shown, and the
scripts will run — no other changes are needed.

**Our trained models** — run `python download_models.py` (§4). It places every
file at the path the scripts already expect:

```
horse_vision_code/models/outdoor/self_supervised_fish_carla/self_fish.tar
horse_vision_code/models/outdoor/supervised_erp/sup_erp.pth
horse_vision_code/models/indoor/...
```

The default values in `test_single_conf.py` then work unchanged:

```python
PRETRAINED_FISH    = "models/outdoor/self_supervised_fish_carla/self_fish.tar"
CHECKPOINT_MINIDA3 = "models/outdoor/supervised_erp/sup_erp.pth"
```

**Official Depth-Anything-V2 weights** — download from
[the DAv2 release page](https://github.com/DepthAnything/Depth-Anything-V2) and
place them in `Depth-Anything-V2/checkpoints/`:

| Scene | File |
|---|---|
| `indoor` | `depth_anything_v2_metric_hypersim_vitb.pth` |
| `outdoor` | `depth_anything_v2_metric_vkitti_vitb.pth` |

These are used when `PRETRAINED_FISH` is left unset, which gives you the
unmodified DAv2 baseline for comparison.

> If neither is found, the encoder is randomly initialised and a `[WARNING]` is
> printed. Predictions will be meaningless — always check the startup log.

---

## 6. Output format

`test_result_integrated.png` stacks seven rows, each 518 px tall; the first five
are left/right pairs side by side (1036 px wide).

| Row | Content |
|---|---|
| 1 | Input fisheye RGB — left, right |
| 2 | Reprojected (synthesised) views — each lens warped into the other using the predicted depth |
| 3 | Confidence maps — bright = photometrically consistent. The crescent shape is the overlapping field of view; only that region is observable by both lenses |
| 4 | **Predicted** fisheye depth |
| 5 | Ground-truth fisheye depth |
| 6 | **Predicted** panoramic depth (splatting + MiniDA3 refinement) |
| 7 | Ground-truth panoramic depth |

Depth is rendered with a jet colormap clipped to `[0, MAX_DEPTH]`; red is far,
blue is near.

`test_single_outdoor.py` produces a five-row variant (rows 2 and 3 omitted) and
requires the ground-truth panorama to be present.

---

## 7. Camera model

The fisheye intrinsics and the rig extrinsics are hard-coded in
`SmoothDepthGenerator` (`test_single_conf.py`). They must match the rig the
checkpoints were trained on:

```python
LEFT_PARAMS  = {'f': 145.41, 'k1': -0.0239, 'cx_offset': -5.0, 'cy_offset': -5.0}
RIGHT_PARAMS = {'f': 149.93, 'k1': -0.0313, 'cx_offset':  5.0, 'cy_offset':  5.0}
# rig: 0.2 m baseline, -75° relative yaw
```

Field of view is clipped at 110.1°. Using a different rig requires re-deriving
these parameters and retraining.

---

## 8. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `FileNotFoundError: Depth-Anything-V2 directory not found!!` | §2 not completed. The directory must contain `depth_anything_v2/`. |
| `ModuleNotFoundError: No module named 'depth_anything_v2'` | DAv2 was found but is incomplete — re-clone it. |
| `[WARNING] ⚠️ Pretrained weights not found!` | Neither a checkpoint in `models/` nor an official DAv2 checkpoint was located. Output will be noise. |
| `FileNotFoundError` on a `models/...` path | The model archive (§4) has not been extracted, or the working directory is not the repository root. All paths in the scripts are **relative** — run from `horse_vision_code/`. |
| `Warning: ... does not exist` for a `.npy` | A ground-truth file is missing. The script substitutes zeros and continues; only the GT rows of the visualisation are affected. |
| Depth looks inverted or saturated | `MAX_DEPTH` does not match `scene` (20 m indoor / 80 m outdoor). |

---

## 9. Citation

```bibtex
@inproceedings{horsevision,
  title     = {Equine-Inspired Vision System: Partially Self-supervised 360{\textdegree} Stereo Panoramic Depth Estimation},
  author    = {Yu, Han and Li, Jianfeng and Li, Shigang},
  booktitle = {Proceedings of the Asian Conference on Computer Vision (ACCV)},
  series    = {Lecture Notes in Computer Science},
  publisher = {Springer},
  year      = {2026}
}
```

This work builds on
[Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2);
please cite it as well.
