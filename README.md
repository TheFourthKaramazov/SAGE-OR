# SAGE-OR: Semi-supervised Adaptive Scene Graph Generation for Operating Rooms

**Brandon Leblanc, Charalambos Poullis**

Concordia University, ICT Lab

[Paper](#) | [Project Page](#)

---

## Abstract

Scene graph generation has emerged as an interpretable framework for surgical applications including workflow recognition, safety monitoring, and context-aware assistance, but current methods depend on dense multi-modal supervision and specialized hardware (synchronized RGB-D sensors, calibration rigs) that make dataset construction expensive, limit adaptation to new procedures, and restrict all existing benchmarks to simulated environments. We propose SAGE-OR, a feature-centric framework that replaces the traditional detect-then-reason paradigm with a decoupled representation-reasoning paradigm in which localization is treated as a latent variable encoded implicitly in pre-computed features, and a lightweight graph transformer performs relational reasoning over cached features. We employ a semi-supervised formulation with general-purpose segmentation prompts to eliminate localization supervision while enabling unsupervised context augmentation through additional prompt-driven entities, such as hands, which are absent from annotations. General-purpose prompts are used to induce near-perfect recall in frozen foundation models, leveraging this property to mitigate failure modes, while enabling simple adaptation to new entities and procedures via generic prompt-level modification. Precision is delegated to downstream attention-based reasoning. This design enables a lightweight 15M-parameter graph transformer (4× smaller than prior work) that trains in 1.4 hours and has a peak memory usage under 2GB, making it suitable for edge hardware used in the operating room. SAGE-OR achieves comparable performance to state-of-the-art (SOTA) methods on the 4D-OR benchmark while requiring substantially reduced supervision and eliminating all localization annotations, providing a practical pathway for rapid adaptation to new surgical settings without additional annotation other than relationship and class labels.

---

## Installation

### 1. Clone with submodules
```bash
git clone --recurse-submodules git@github.com:TheFourthKaramazov/SAGE-OR.git
cd SAGE-OR
```

### 2. Create conda environment
```bash
conda env create -f environment.yml
conda activate ssg
```

### 3. Install PyTorch (CUDA)
```bash
# Adjust cu126 to match your CUDA version (cu118, cu121, cu126)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### 4. Install requirements
```bash
pip install -r requirements.txt
```

### 5. Download DINOv3 weights

DINOv3-L weights must be downloaded manually:

```bash
mkdir -p weights/dinov3
# Download dinov3_vitl16_pretrain.pth (~1.2GB) from:
# [PLACEHOLDER: Add download link]
# Place in weights/dinov3/dinov3_vitl16_pretrain.pth
```

### 6. Download pre-trained SAGE-OR checkpoint (optional)

```bash
mkdir -p checkpoints
# Download sage_or.pt from:
# [PLACEHOLDER: Add download link]
# Place in checkpoints/sage_or.pt
```

---

## Quick Start: Inference Only

If you just want to run inference with the pre-trained model:

### 1. Download pre-computed caches
```bash
# [PLACEHOLDER: Add download instructions for caches]
# Place in caches/features_dinov3_vitl16/
```

### 2. Run evaluation
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

To generate all caches from raw images, follow these steps in order:

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

If you use this code, please cite our paper:

```bibtex
@inproceedings{leblanc2026sageor,
  title={SAGE-OR: Semi-supervised Adaptive Scene Graph Generation for Operating Rooms},
  author={Leblanc, Brandon and Poullis, Charalambos},
  booktitle={British Machine Vision Conference (BMVC)},
  year={2026}
}
```

Please also cite the 4D-OR dataset and Pi3:

```bibtex
@inproceedings{ozsoy20224dor,
  title={4D-OR: Semantic Scene Graphs for OR Domain Modeling},
  author={Özsoy, Ege and Örnek, Evin Pınar and Eck, Ulrich and Czempiel, Tobias and Tombari, Federico and Navab, Nassir},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
  year={2022}
}

@article{liu2025pi3,
  title={Pi3: Scalable Zero-shot Multi-View 3D Reconstruction from Single and Few Images},
  author={Liu, Yang and others},
  journal={arXiv preprint},
  year={2025}
}
```

---

## Acknowledgments

This work was conducted at the Immersive and Creative Technologies Lab (ICT Lab) at Concordia University.

We thank the authors of Pi3, SAM3, and the 4D-OR dataset for making their work publicly available.

---

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
