# SAGE-OR: Semi-supervised Adaptive Scene Graph Generation for Operating Rooms

**Brandon Leblanc, Charalambos Poullis**

Concordia University, ICT Lab

**Paper** (BMVC 2026 Oral — to appear) · [**arXiv**](https://arxiv.org/abs/2608.15336) · [**Project Page**](https://thefourthkaramazov.github.io/SAGE-OR/)

---

## Abstract

Current surgical scene graph generation methods depend on dense multi-modal supervision and specialized hardware (synchronized RGB-D sensors, calibration rigs), making dataset construction expensive and restricting all existing benchmarks to simulated environments. We propose SAGE-OR, a feature-centric framework that replaces the traditional detect-then-reason paradigm with a decoupled representation-reasoning paradigm in which localization is derived from frozen foundation models, encoded implicitly in pre-computed features, and used without any localization supervision, while a lightweight graph transformer performs relational reasoning over cached features. We employ a semi-supervised formulation with general-purpose segmentation prompts to eliminate localization supervision while enabling unsupervised context augmentation through additional prompt-driven entities, such as hands, which are absent from annotations. General-purpose prompts are used to induce near-perfect recall, while precision is delegated to downstream attention-based reasoning, enabling simple adaptation to new entities via prompt-level modification. This design enables a lightweight 15M-parameter graph transformer that trains in 1.4 hours and runs relational inference at ${\sim}$1ms per frame with peak memory under 2GB, suitable for edge hardware used in the operating room; feature extraction runs offline as a separate caching stage (4.27s per frame). On the 4D-OR benchmark, the core model achieves 76\% F1, matching the fully supervised 4D-OR baseline while eliminating all localization annotations, and unsupervised hand augmentation raises this to 86\%, within 4 points of state-of-the-art (SOTA) methods requiring dense multi-modal supervision, providing a practical pathway for adaptation to new surgical settings without annotation other than relationship and class labels.

---

## Installation

### 1. Clone with submodules
```bash
git clone --recurse-submodules git@github.com:TheFourthKaramazov/SAGE-OR.git
cd SAGE-OR
```
Already cloned without submodules? Run:
```bash
git submodule update --init --recursive
```
> **Note:** the SAM3 submodule is intentionally pinned (commit `f6e51f5`). Do **not** run `git submodule update --remote` on it — newer SAM3 revisions change segmentation behavior and are incompatible with the released caches.

### 2. Create the conda environment
```bash
conda env create -f environment.yml
conda activate sage-or
```

### 3. Download DINOv3 weights
DINOv3 weights are gated by Meta. Request access and download `dinov3_vitl16_pretrain.pth` from the [official DINOv3 repository](https://github.com/facebookresearch/dinov3) (fill out the Meta form, then follow the download link).

```bash
mkdir -p weights/dinov3
# Place the downloaded file at:
#   weights/dinov3/dinov3_vitl16_pretrain.pth
```
SAGE-OR also supports ResNet, EfficientNet, and CLIP encoders (set `features.dino_model` in `configs/real-or.yaml`) — results vary widely by encoder, see the encoder ablation in the paper. To add a new one, create a file under `src/features/` following the existing examples and register it in `src/features/visual_encoder.py`.

### 4. Download the model and data
See [Downloads](#downloads) for the pretrained checkpoint and the data + cache package.

---

## Downloads

Hosted on Google Drive as three packages. **Inference and training need only the Checkpoint + Caches** (~440 MB); the 4D-OR data package is only for regenerating caches from scratch.

| Package | Size | Download | Extract to |
|---|---|---|---|
| **Checkpoint** (`sage_or.pt`) | 157 MB | [Google Drive](https://drive.google.com/file/d/1sEuudU6quLqTaS8nmYxwMElXgLabzSmW/view?usp=drive_link) | repo root → `checkpoints/sage_or.pt` |
| **Caches** (feature cache + GT labels) | 286 MB | [Google Drive](https://drive.google.com/file/d/1MGO2Xg0GApPJPeqTrVnrdOXASLQg3k0i/view?usp=drive_link) | repo root → `caches/`, `data/4D-OR/OR_4D_data/` |
| **4D-OR data** (condensed, RGB-only) | 25 GB | [Google Drive](https://drive.google.com/file/d/1Q9KiPxQPdOee_UP030KKl_g4l6ERdYRe/view?usp=drive_link) | into `data/` → `data/4D-OR/` |

The 4D-OR package is our condensed, RGB-only version of [4D-OR](https://github.com/egeozsoy/4D-OR) (25 GB vs 163 GB original). Sufficient for RGB-only methods like SAGE-OR. DINOv3 weights are separate and Meta-gated (Installation step 3), and only needed to regenerate caches.

---

## Quick Start: Inference Only

Download the **Checkpoint** and **Caches** packages ([Downloads](#downloads)) and extract both at the repo root, then run evaluation:
```bash
python -m src.eval.evaluate --config configs/real-or.yaml --checkpoint checkpoints/sage_or.pt
```

---

## Training

### Using pre-computed caches (recommended)

If you have pre-computed feature caches:

```bash
python -m src.training.train_scene_graph --config configs/real-or.yaml
```

Optional arguments:
- `--resume checkpoints/latest.pt` - Resume from checkpoint
- `--eval_every 1` - Evaluate every epoch (default: 10)

---

## Cache Generation (From Scratch)

To generate all caches from raw images, follow these steps in order.

> Optional — most users only need the **Caches** package. Rebuilding from scratch requires the **4D-OR data** package (extracted to `data/4D-OR/`) and the DINOv3 weights from Installation step 3.

### Step 1: Generate manifest

Creates an index of all frames, images, and train/val/test splits.

```bash
python -m src.dataset.manifest_generation --data-dir data/4D-OR --output-dir processed_data --cache-dir caches/4d_or_cache
```

### Step 2: Generate Pi3 geometry cache

Extracts 3D pointmaps from multi-view images using Pi3.

```bash
python -m src.dataset.pointmap_cache_generator --manifest processed_data/manifest.csv --output-dir caches/4d_or_cache --resume
```

### Step 3: Generate SAM3 segmentation masks

Segments objects using prompt-based segmentation.

```bash
python -m src.dataset.mask_generator --config configs/sam3_prompts.yaml --manifest processed_data/manifest.csv --output-dir caches/sam3_masks --resume
```

### Step 4: Generate feature cache

Combines Pi3 geometry, SAM3 masks, and DINOv3 visual features.

```bash
python -m src.dataset.feature_cache_generator --config configs/real-or.yaml --output-dir caches/features_dinov3_vitl16 --resume
```

### Step 5: Train

```bash
python -m src.training.train_scene_graph --config configs/real-or.yaml
```

---

## Project Structure

```
SAGE-OR/
├── configs/                    # Configuration files
│   ├── real-or.yaml           # Main training config
│   └── sam3_prompts.yaml      # SAM3 segmentation prompts
├── checkpoints/               # Model checkpoints
├── caches/                    # Pre-computed caches
│   ├── 4d_or_cache/          # Pi3 geometry cache
│   ├── sam3_masks/           # SAM3 segmentation masks
│   └── features_dinov3_vitl16/ # Final feature cache
├── data/                      # Dataset
│   └── 4D-OR/                # 4D-OR dataset
├── processed_data/            # Manifest and processed images
├── weights/                   # Model weights
│   └── dinov3/               # DINOv3 weights
├── external/                  # Git submodules
│   ├── dinov3/               # DINOv3 encoder
│   ├── pi3/                  # Pi3 geometry
│   └── sam3/                 # SAM3 segmentation
└── src/                       # Source code
    ├── dataset/              # Data loading and cache generation
    ├── features/             # Visual feature extraction
    ├── fusion/               # Multi-view fusion
    ├── models/               # Graph transformer model
    └── training/             # Training scripts
```

---

## Citation

If you use this code, please cite our paper (to appear; arXiv version coming soon):

```bibtex
@inproceedings{leblanc2026sageor,
  title={SAGE-OR: Semi-supervised Adaptive Scene Graph Generation for Operating Rooms},
  author={Leblanc, Brandon and Poullis, Charalambos},
  booktitle={Proceedings of the British Machine Vision Conference (BMVC)},
  year={2026},
  note={To appear}
}
```


The models and prior work we build on are credited under [Acknowledgments](#acknowledgments).

---

## Acknowledgments

Developed at the Immersive and Creative Technologies Lab (ICT Lab), Concordia University.

We thank the authors of [Pi3](https://github.com/yyfz/Pi3), [SAM 3](https://github.com/facebookresearch/sam3), [DINOv3](https://github.com/facebookresearch/dinov3), [LABRAD-OR](https://github.com/egeozsoy/LABRAD-OR), and the [4D-OR](https://github.com/egeozsoy/4D-OR) dataset for making their work publicly available.

---

## License

The SAGE-OR code is released under the MIT License (see [LICENSE](LICENSE)). The 4D-OR data redistributed here is also MIT, matching the original dataset.
