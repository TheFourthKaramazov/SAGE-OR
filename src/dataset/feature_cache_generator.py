#!/usr/bin/env python3
"""
Generate Feature Cache for Scene Graph Training

Pre-computes and caches:
- Instance fusion (3D IoU clustering across cameras)
- DINOv3 visual features per instance
- Geometric features (centroid, bbox, spread, etc.)

Usage:
    python -m src.dataset.feature_cache_generator --config configs/real-or.yaml
    python -m src.dataset.feature_cache_generator --config configs/real-or.yaml --split train
    python -m src.dataset.feature_cache_generator --config configs/real-or.yaml --resume
"""

import argparse
import json
import yaml
import numpy as np
import torch
import cv2
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import time
import sys

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.fusion.instance_fusion import InstanceFusion
from src.fusion.feature_pooling import FeaturePooler
from src.features.visual_encoder import create_encoder, get_encoder_info


def load_sam3_config(sam3_cache_dir: Path) -> dict:
    """Load SAM3 configuration from cache metadata."""
    metadata_path = sam3_cache_dir / 'metadata.json'
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"SAM3 metadata not found at {metadata_path}. "
            "Please generate SAM3 cache first."
        )

    with open(metadata_path) as f:
        metadata = json.load(f)

    return metadata['config']


def build_category_mappings(sam3_config: dict) -> tuple:
    """
    Build category mappings from SAM3 config.

    Returns:
        (category_to_idx, idx_to_category, hybrid_categories, all_categories)
    """
    prompts = sam3_config['prompts']

    # Alphabetically sorted for deterministic ordering
    all_categories = sorted(prompts.keys())

    # Build index mapping
    category_to_idx = {cat: idx for idx, cat in enumerate(all_categories)}
    idx_to_category = {idx: cat for cat, idx in category_to_idx.items()}

    # Determine hybrid categories (use high resolution)
    hybrid_categories = {
        cat for cat, cfg in prompts.items()
        if cfg.get('resolution') == 'high'
    }

    return category_to_idx, idx_to_category, hybrid_categories, all_categories


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_gt_index(gt_dir: Path, split: str) -> set:
    """Load GT to get list of valid (take_idx, scan_idx) for this split."""
    gt_path = gt_dir / f'relationships_{split}.json'

    with open(gt_path) as f:
        data = json.load(f)

    frames = set()
    for scan in data['scans']:
        take_idx = scan['take_idx']
        scan_idx = int(scan['scan'])
        frames.add((take_idx, scan_idx))

    return frames


def load_frame_data(
    take_id: int,
    scan_idx: int,
    manifest_df: pd.DataFrame,
    pi3_cache_dir: Path,
    sam3_cache_dir: Path,
    all_categories: list,
) -> dict:
    """Load Pi3, SAM3, and RGB data for a frame."""
    take_str = f'take{take_id:02d}'

    # Load Pi3 cache
    pi3_path = pi3_cache_dir / take_str / f'frame_{scan_idx:04d}.npz'
    pi3_data = np.load(pi3_path)

    pointmaps = pi3_data['xyz_local'].astype(np.float32)
    confidences = pi3_data['conf'].astype(np.float32)[:, :, :, 0]
    camera_poses = pi3_data['camera_poses']
    camera_ids = pi3_data['camera_ids']

    # Load RGB images (both low-res and high-res)
    row = manifest_df[
        (manifest_df['take_id'] == take_id) &
        (manifest_df['scan_idx'] == scan_idx)
    ].iloc[0]

    images = {}  # Low-res for baseline (person, table, anesthesia)
    images_highres = {}  # High-res for hybrid (tools, hands)

    for cam_id in camera_ids:
        cam_id = int(cam_id)

        # Load low-res (processed) image
        img_path = row.get(f'cam{cam_id}_processed')
        if pd.notna(img_path) and Path(img_path).exists():
            img = cv2.imread(str(img_path))
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float()
                images[cam_id] = img_tensor

        # Load high-res (original) image for hybrid extraction
        img_path_highres = row.get(f'cam{cam_id}_rgb')
        if pd.notna(img_path_highres) and Path(img_path_highres).exists():
            img_hr = cv2.imread(str(img_path_highres))
            if img_hr is not None:
                images_highres[cam_id] = cv2.cvtColor(img_hr, cv2.COLOR_BGR2RGB)

    # Get target size
    if images:
        sample_img = list(images.values())[0]
        target_h, target_w = sample_img.shape[1], sample_img.shape[2]
    else:
        target_h, target_w = 392, 518

    # Load SAM3 masks
    masks_per_camera = {}
    scores_per_camera = {}

    for cam_id in camera_ids:
        cam_id = int(cam_id)
        mask_path = sam3_cache_dir / take_str / f'frame_{scan_idx:04d}_cam{cam_id}.npz'

        if not mask_path.exists():
            continue

        mask_data = np.load(mask_path)
        masks_per_camera[cam_id] = {}
        scores_per_camera[cam_id] = {}

        for category in all_categories:
            masks_key = f'{category}_masks'
            scores_key = f'{category}_scores'

            if masks_key in mask_data and mask_data[masks_key].size > 0:
                masks = mask_data[masks_key]

                # Resize if needed
                if masks.shape[1] != target_h or masks.shape[2] != target_w:
                    resized = []
                    for m in masks:
                        r = cv2.resize(m.astype(np.uint8), (target_w, target_h),
                                      interpolation=cv2.INTER_NEAREST)
                        resized.append(r.astype(bool))
                    masks = np.stack(resized) if resized else masks

                masks_per_camera[cam_id][category] = masks
                scores_per_camera[cam_id][category] = mask_data[scores_key]

    return {
        'pointmaps': pointmaps,
        'confidences': confidences,
        'camera_poses': camera_poses,
        'camera_ids': camera_ids,
        'images': images,
        'images_highres': images_highres,  # For hybrid extraction (tools, hands)
        'masks_per_camera': masks_per_camera,
        'scores_per_camera': scores_per_camera,
    }


def extract_geometric_features(instance, geometric_feat_dim: int = 13) -> np.ndarray:
    """Extract geometric features for an instance."""
    features = []

    # 3D centroid (3)
    features.extend(instance.centroid_world.tolist())

    # Point cloud stats
    points = instance.points_world
    if len(points) > 0:
        # Bounding box size (3)
        bbox_min = points.min(axis=0)
        bbox_max = points.max(axis=0)
        bbox_size = bbox_max - bbox_min
        features.extend(bbox_size.tolist())

        # Point cloud spread/std (3)
        std = points.std(axis=0)
        features.extend(std.tolist())

        # Number of points normalized (1)
        features.append(len(points) / 1000.0)

        # Number of cameras (1)
        features.append(len(instance.camera_ids) / 6.0)

        # Confidence score (1)
        features.append(instance.score)

        # Vertical extent (1)
        features.append(bbox_size[2] if len(bbox_size) > 2 else 0)
    else:
        features.extend([0] * 10)

    return np.array(features[:geometric_feat_dim], dtype=np.float32)


def get_category_hint(
    category: str,
    category_to_idx: dict,
    category_hint_dim: int
) -> np.ndarray:
    """Get one-hot category hint from SAM3 category."""
    hint = np.zeros(category_hint_dim, dtype=np.float32)
    idx = category_to_idx.get(category, 0)
    hint[idx] = 1.0
    return hint


def main():
    parser = argparse.ArgumentParser(
        description="Generate feature cache for scene graph training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/real-or.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='caches/features',
        help='Output cache directory'
    )
    parser.add_argument(
        '--split',
        type=str,
        default=None,
        help='Process single split (train/validation/test). Default: all three'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Skip already cached frames'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device for DINOv3 inference'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of frames per split (for debugging)'
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Get encoder name early (for output dir naming)
    encoder_name = config.get('features', {}).get('encoder') or \
                   config.get('features', {}).get('dino_model', 'dinov3_vitl16')

    # Paths
    manifest_path = project_root / config['data']['manifest_path']
    pi3_cache_dir = project_root / config['data']['pi3_cache_dir']
    sam3_cache_dir = project_root / config['data']['sam3_cache_dir']
    gt_dir = project_root / config['data']['gt_dir']

    # Load SAM3 config and build category mappings
    sam3_config = load_sam3_config(sam3_cache_dir)
    category_to_idx, idx_to_category, hybrid_categories, all_categories = \
        build_category_mappings(sam3_config)

    print(f"\n{'=' * 60}")
    print("SAM3 CATEGORIES")
    print(f"{'=' * 60}")
    print(f"All categories ({len(all_categories)}): {all_categories}")
    print(f"Hybrid categories (high-res): {sorted(hybrid_categories)}")
    print(f"Baseline categories (low-res): {sorted(set(all_categories) - hybrid_categories)}")

    # Auto-derive output directory name from encoder if using default
    if args.output_dir == 'caches/features':
        output_dir = project_root / f'caches/features_{encoder_name}'
    else:
        output_dir = project_root / args.output_dir

    # Load manifest
    manifest_df = pd.read_csv(manifest_path)

    # Feature dimensions from config
    visual_feat_dim = config['model']['visual_feat_dim']
    geometric_feat_dim = config['model']['geometric_feat_dim']

    # Auto-set category_hint_dim from SAM3 categories
    category_hint_dim = len(all_categories)
    config_hint_dim = config['model'].get('category_hint_dim')
    if config_hint_dim and config_hint_dim != category_hint_dim:
        print(f"\nWARNING: Config category_hint_dim={config_hint_dim} doesn't match "
              f"SAM3 categories ({category_hint_dim}). Using {category_hint_dim}.")

    # Initialize fusion
    fusion_cfg = config.get('fusion', {})
    fusion = InstanceFusion(
        conf_threshold=fusion_cfg.get('conf_threshold', 0.1),
        min_points=fusion_cfg.get('min_points', 10),
        max_points_per_mask=fusion_cfg.get('max_points_per_mask', 1000),
        centroid_threshold=fusion_cfg.get('centroid_threshold', 0.5),
        categories=all_categories,
    )

    # IoU threshold is used in fuse_frame() method, not __init__
    iou_threshold = fusion_cfg.get('iou_threshold', 0.15)

    print(f"\n{'=' * 60}")
    print("Feature Cache Generation for Scene Graph Training")
    print(f"{'=' * 60}")
    print(f"Device: {device}")
    print(f"Config: {args.config}")
    print(f"Output: {output_dir}")
    print(f"Resume: {args.resume}")
    print(f"Extraction: HYBRID (crop+zoom for tools/hands, baseline for others)")
    print()
    print("Fusion parameters:")
    print(f"  conf_threshold: {fusion_cfg.get('conf_threshold', 0.1)}")
    print(f"  min_points: {fusion_cfg.get('min_points', 10)}")
    print(f"  max_points_per_mask: {fusion_cfg.get('max_points_per_mask', 1000)}")
    print(f"  iou_threshold: {iou_threshold} (hardcoded in fuse_frame, not configurable)")
    print()

    # Initialize visual encoder
    print(f"Loading visual encoder: {encoder_name}...")
    encoder_info = get_encoder_info(encoder_name)
    print(f"  {encoder_info['description']}")
    print(f"  Output dimension: {encoder_info['feat_dim']}")

    extractor = create_encoder(encoder_name, device=str(device))

    # Get aggregation method from config
    aggregation_method = config.get('features', {}).get('aggregation', 'mean')
    pooler = FeaturePooler(
        extractor,
        aggregation=aggregation_method,
        hybrid_categories=hybrid_categories
    )
    print(f"Aggregation method: {aggregation_method}")

    # Process splits
    splits = [args.split] if args.split else ['train', 'validation', 'test']

    total_processed = 0
    total_skipped = 0
    total_failed = 0

    for split in splits:
        print(f"\n{'=' * 40}")
        print(f"Processing {split} split")
        print(f"{'=' * 40}")

        # Get frames for this split
        gt_frames = load_gt_index(gt_dir, split)

        # Filter to frames we have cached data for
        frames = []
        for (take_idx, scan_idx) in gt_frames:
            take_str = f'take{take_idx:02d}'
            pi3_path = pi3_cache_dir / take_str / f'frame_{scan_idx:04d}.npz'
            sam3_path = sam3_cache_dir / take_str / f'frame_{scan_idx:04d}_cam1.npz'

            if pi3_path.exists() and sam3_path.exists():
                frames.append((take_idx, scan_idx))

        frames = sorted(frames)

        if args.limit:
            frames = frames[:args.limit]

        print(f"Found {len(frames)} frames with Pi3 + SAM3 caches")

        processed = 0
        skipped = 0
        failed = 0

        # Process each frame
        for take_id, scan_idx in tqdm(frames, desc=f'Caching {split}'):
            # Set deterministic seed for reproducibility
            np.random.seed(take_id * 10000 + scan_idx)

            # Output path
            take_str = f'take{take_id:02d}'
            cache_dir = output_dir / take_str
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / f'frame_{scan_idx:04d}.npz'

            # Skip if already cached
            if args.resume and cache_path.exists():
                skipped += 1
                continue

            try:
                # Load frame data
                data = load_frame_data(
                    take_id, scan_idx, manifest_df,
                    pi3_cache_dir, sam3_cache_dir,
                    all_categories
                )

                # Run fusion
                instances = fusion.fuse_frame(
                    data['masks_per_camera'],
                    data['scores_per_camera'],
                    data['pointmaps'],
                    data['confidences'],
                    data['camera_poses'],
                    data['camera_ids'],
                )

                n_instances = len(instances)

                # Extract features
                if n_instances > 0:
                    # Visual features (DINOv3)
                    # Uses hybrid extraction for tools/hands (crop from high-res)
                    visual_features = pooler.pool_all_instances(
                        instances,
                        data['images'],
                        data['masks_per_camera'],
                        data['images_highres'],  # For hybrid extraction
                    ).cpu().numpy()

                    # Geometric features
                    geometric_features = np.stack([
                        extract_geometric_features(inst, geometric_feat_dim)
                        for inst in instances
                    ])

                    # Category hints
                    category_hints = np.stack([
                        get_category_hint(inst.category, category_to_idx, category_hint_dim)
                        for inst in instances
                    ])

                    # Centroids
                    centroids = np.stack([
                        inst.centroid_world for inst in instances
                    ])

                    # SAM3 categories (as indices for storage)
                    sam3_categories = np.array([
                        category_to_idx[inst.category] for inst in instances
                    ], dtype=np.int32)
                else:
                    visual_features = np.zeros((0, visual_feat_dim), dtype=np.float32)
                    geometric_features = np.zeros((0, geometric_feat_dim), dtype=np.float32)
                    category_hints = np.zeros((0, category_hint_dim), dtype=np.float32)
                    centroids = np.zeros((0, 3), dtype=np.float32)
                    sam3_categories = np.array([], dtype=np.int32)

                # Save cache
                np.savez(
                    cache_path,
                    visual_features=visual_features.astype(np.float16),  # Compress
                    geometric_features=geometric_features.astype(np.float16),
                    category_hints=category_hints.astype(np.float16),
                    centroids=centroids.astype(np.float32),
                    sam3_categories=sam3_categories,
                    n_instances=n_instances,
                )

                processed += 1

            except Exception as e:
                print(f"\nError processing take{take_id:02d} frame {scan_idx}: {e}")
                failed += 1
                continue

            # Clear GPU cache periodically
            if processed % 50 == 0 and device.type == 'cuda':
                torch.cuda.empty_cache()

        print(f"\n{split}: processed={processed}, skipped={skipped}, failed={failed}")
        total_processed += processed
        total_skipped += skipped
        total_failed += failed

    # Save metadata
    metadata = {
        "feature_extractor": encoder_name,
        "feature_extractor_info": encoder_info['description'],
        "extraction_mode": "hybrid",
        "sam3_categories": all_categories,
        "category_to_idx": category_to_idx,
        "idx_to_category": idx_to_category,
        "hybrid_categories": sorted(list(hybrid_categories)),
        "visual_feat_dim": visual_feat_dim,
        "geometric_feat_dim": geometric_feat_dim,
        "category_hint_dim": category_hint_dim,
        "fusion_config": fusion_cfg,
        "total_frames": total_processed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    metadata_path = output_dir / "metadata.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Print summary
    print(f"\n{'=' * 60}")
    print("FEATURE CACHE GENERATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total processed: {total_processed}")
    print(f"Total skipped: {total_skipped}")
    print(f"Total failed: {total_failed}")
    print(f"Cache saved to: {output_dir}")
    print(f"\nPer-frame contents:")
    print(f"  visual_features:    [N, {visual_feat_dim}] float16")
    print(f"  geometric_features: [N, {geometric_feat_dim}] float16")
    print(f"  category_hints:     [N, {category_hint_dim}] float16")
    print(f"  centroids:          [N, 3] float32")
    print(f"  sam3_categories:    [N] int32")
    print(f"  n_instances:        scalar")

    return 0


if __name__ == "__main__":
    exit(main())
