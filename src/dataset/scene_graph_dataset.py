"""
Scene Graph Dataset for 4D-OR.

Loads frames, runs fusion, extracts features, and provides GT labels.
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import cv2

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.fusion.instance_fusion import InstanceFusion, FusedInstance
from src.fusion.feature_pooling import FeaturePooler
from src.features.dinov3 import DINOv3Extractor


# Category compatibility: SAM3 category → valid GT classes
CATEGORY_TO_GT_CLASSES = {
    'person': ['Patient', 'human_0', 'human_1', 'human_2', 'human_3', 'human_4', 'human_5'],
    'table': ['instrument_table', 'operating_table', 'secondary_table'],
    'anesthesia': ['anesthesia_equipment'],
    'tool': ['instrument'],
    'hand': [],  # No GT mapping
}

# GT class to index
GT_CLASS_TO_IDX = {
    'Patient': 0,
    'anesthesia_equipment': 1,
    'human_0': 2,
    'human_1': 3,
    'human_2': 4,
    'human_3': 5,
    'human_4': 6,
    'human_5': 7,
    'instrument': 8,
    'instrument_table': 9,
    'operating_table': 10,
    'secondary_table': 11,
}

IDX_TO_GT_CLASS = {v: k for k, v in GT_CLASS_TO_IDX.items()}


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


def load_gt_relationships(gt_dir: Path, split: str) -> Dict:
    """Load GT relationships and index by (take_idx, scan_idx)."""
    gt_path = gt_dir / f'relationships_{split}.json'

    with open(gt_path) as f:
        data = json.load(f)

    gt_index = {}
    for scan in data['scans']:
        take_idx = scan['take_idx']
        scan_idx = int(scan['scan'])
        gt_index[(take_idx, scan_idx)] = scan

    return gt_index


def load_relationship_classes(gt_dir: Path) -> Dict[str, int]:
    """Load relationship class names."""
    rel_path = gt_dir / 'relationships.txt'

    rel_to_idx = {}
    with open(rel_path) as f:
        for idx, line in enumerate(f):
            rel_name = line.strip()
            if rel_name:
                rel_to_idx[rel_name] = idx

    return rel_to_idx


class SceneGraphDataset(Dataset):
    """
    Dataset for scene graph prediction.

    Each item is a frame with:
    - Fused instances (from SAM3 + Pi3)
    - Visual features (DINOv3)
    - Geometric features (3D centroid, bbox)
    - GT object classes and relationships
    """

    def __init__(
        self,
        config: Dict,
        split: str = 'train',
        extractor: Optional[DINOv3Extractor] = None,
        max_instances: int = 32,
    ):
        self.config = config
        self.split = split
        self.max_instances = max_instances

        # Paths
        project_root = Path(__file__).parent.parent.parent
        self.manifest_path = project_root / config['data']['manifest_path']
        self.pi3_cache_dir = project_root / config['data']['pi3_cache_dir']
        self.sam3_cache_dir = project_root / config['data']['sam3_cache_dir']
        self.gt_dir = project_root / config['data']['gt_dir']

        # Load SAM3 config and build category mappings
        sam3_config = load_sam3_config(self.sam3_cache_dir)
        self.category_to_idx, self.idx_to_category, self.hybrid_categories, self.all_categories = \
            build_category_mappings(sam3_config)

        # Load GT
        self.gt_index = load_gt_relationships(self.gt_dir, split)
        self.rel_to_idx = load_relationship_classes(self.gt_dir)
        self.num_relations = len(self.rel_to_idx)

        # Load manifest and filter to split
        self.manifest_df = pd.read_csv(self.manifest_path)
        self.frames = self._get_frames_for_split()

        # Fusion module
        fusion_cfg = config.get('fusion', {})
        self.fusion = InstanceFusion(
            conf_threshold=fusion_cfg.get('conf_threshold', 0.1),
            min_points=fusion_cfg.get('min_points', 10),
            centroid_threshold=fusion_cfg.get('centroid_threshold', 0.5),
            categories=self.all_categories,
        )

        # Feature extractor (shared across dataset)
        self.extractor = extractor
        if self.extractor is not None:
            self.pooler = FeaturePooler(
                self.extractor,
                aggregation='mean',
                hybrid_categories=self.hybrid_categories
            )
        else:
            self.pooler = None

        # Dimensions from config
        self.visual_feat_dim = config['model']['visual_feat_dim']
        self.geometric_feat_dim = config['model']['geometric_feat_dim']
        self.category_hint_dim = len(self.all_categories)  # Auto-set from SAM3

    def _get_frames_for_split(self) -> List[Tuple[int, int]]:
        """Get list of (take_id, scan_idx) for this split."""
        frames = []

        for (take_idx, scan_idx), gt_data in self.gt_index.items():
            # Check if we have cached data for this frame
            take_str = f'take{take_idx:02d}'
            pi3_path = self.pi3_cache_dir / take_str / f'frame_{scan_idx:04d}.npz'
            sam3_path = self.sam3_cache_dir / take_str / f'frame_{scan_idx:04d}_cam1.npz'

            if pi3_path.exists() and sam3_path.exists():
                frames.append((take_idx, scan_idx))

        return sorted(frames)

    def __len__(self) -> int:
        return len(self.frames)

    def _load_frame_data(self, take_id: int, scan_idx: int) -> Dict:
        """Load Pi3, SAM3, and RGB data for a frame."""
        take_str = f'take{take_id:02d}'

        # Load Pi3 cache
        pi3_path = self.pi3_cache_dir / take_str / f'frame_{scan_idx:04d}.npz'
        pi3_data = np.load(pi3_path)

        pointmaps = pi3_data['xyz_local'].astype(np.float32)
        confidences = pi3_data['conf'].astype(np.float32)[:, :, :, 0]
        camera_poses = pi3_data['camera_poses']
        camera_ids = pi3_data['camera_ids']

        # Load RGB images
        row = self.manifest_df[
            (self.manifest_df['take_id'] == take_id) &
            (self.manifest_df['scan_idx'] == scan_idx)
        ].iloc[0]

        images = {}
        for cam_id in camera_ids:
            cam_id = int(cam_id)
            img_path = row.get(f'cam{cam_id}_processed')
            if pd.notna(img_path) and Path(img_path).exists():
                img = cv2.imread(str(img_path))
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float()
                    images[cam_id] = img_tensor

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
            mask_path = self.sam3_cache_dir / take_str / f'frame_{scan_idx:04d}_cam{cam_id}.npz'

            if not mask_path.exists():
                continue

            mask_data = np.load(mask_path)
            masks_per_camera[cam_id] = {}
            scores_per_camera[cam_id] = {}

            for category in self.all_categories:
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
            'masks_per_camera': masks_per_camera,
            'scores_per_camera': scores_per_camera,
        }

    def _extract_geometric_features(self, instance: FusedInstance) -> np.ndarray:
        """Extract geometric features for an instance."""
        features = []

        # 3D centroid (3)
        features.extend(instance.centroid_world.tolist())

        # Point cloud stats (13)
        points = instance.points_world
        if len(points) > 0:
            # Bounding box size
            bbox_min = points.min(axis=0)
            bbox_max = points.max(axis=0)
            bbox_size = bbox_max - bbox_min
            features.extend(bbox_size.tolist())  # 3

            # Point cloud spread (std)
            std = points.std(axis=0)
            features.extend(std.tolist())  # 3

            # Number of points (normalized)
            features.append(len(points) / 1000.0)  # 1

            # Number of cameras
            features.append(len(instance.camera_ids) / 6.0)  # 1

            # Confidence score
            features.append(instance.score)  # 1

            # Vertical extent (height indicator)
            features.append(bbox_size[2] if len(bbox_size) > 2 else 0)  # 1
        else:
            features.extend([0] * 13)

        return np.array(features[:self.geometric_feat_dim], dtype=np.float32)

    def _get_category_hint(self, category: str) -> np.ndarray:
        """Get one-hot category hint from SAM3 category."""
        hint = np.zeros(self.category_hint_dim, dtype=np.float32)
        idx = self.category_to_idx.get(category, 0)
        hint[idx] = 1.0
        return hint

    def _get_gt_for_frame(self, take_id: int, scan_idx: int) -> Dict:
        """Get GT labels for a frame."""
        gt_data = self.gt_index.get((take_id, scan_idx), {'objects': {}, 'relationships': []})

        # Object classes: {obj_id: class_name}
        gt_objects = gt_data.get('objects', {})

        # Relationships: [(subj_id, obj_id, rel_name), ...]
        gt_relationships = gt_data.get('relationships', [])

        return {
            'objects': gt_objects,
            'relationships': gt_relationships,
        }

    def __getitem__(self, idx: int) -> Dict:
        take_id, scan_idx = self.frames[idx]

        # Load frame data
        data = self._load_frame_data(take_id, scan_idx)

        # Run fusion
        instances = self.fusion.fuse_frame(
            data['masks_per_camera'],
            data['scores_per_camera'],
            data['pointmaps'],
            data['confidences'],
            data['camera_poses'],
            data['camera_ids'],
        )

        n_instances = len(instances)

        # Extract features
        if self.pooler is not None and n_instances > 0:
            visual_features = self.pooler.pool_all_instances(
                instances, data['images'], data['masks_per_camera']
            ).cpu().numpy()
        else:
            visual_features = np.zeros((n_instances, self.visual_feat_dim), dtype=np.float32)

        geometric_features = np.stack([
            self._extract_geometric_features(inst) for inst in instances
        ]) if n_instances > 0 else np.zeros((0, self.geometric_feat_dim), dtype=np.float32)

        category_hints = np.stack([
            self._get_category_hint(inst.category) for inst in instances
        ]) if n_instances > 0 else np.zeros((0, self.category_hint_dim), dtype=np.float32)

        centroids = np.stack([
            inst.centroid_world for inst in instances
        ]) if n_instances > 0 else np.zeros((0, 3), dtype=np.float32)

        # SAM3 categories for compatibility checking during matching
        sam3_categories = [inst.category for inst in instances]

        # Get GT
        gt = self._get_gt_for_frame(take_id, scan_idx)

        # Convert GT to tensors
        gt_classes = []
        gt_obj_ids = []
        for obj_id, class_name in gt['objects'].items():
            if class_name in GT_CLASS_TO_IDX:
                gt_classes.append(GT_CLASS_TO_IDX[class_name])
                gt_obj_ids.append(int(obj_id))

        # Relationship matrix [M, M] where M = num GT objects
        # Matches 4D-OR encoding: 0-13 = actual relations, 14 = 'none'
        n_gt = len(gt_classes)
        none_idx = self.num_relations  # 'none' is at index 14 (after 0-13)
        gt_rel_matrix = np.full((n_gt, n_gt), none_idx, dtype=np.int64)

        obj_id_to_idx = {obj_id: i for i, obj_id in enumerate(gt_obj_ids)}

        for rel in gt['relationships']:
            subj_id, obj_id, rel_idx = rel[0], rel[1], rel[2]
            if subj_id in obj_id_to_idx and obj_id in obj_id_to_idx:
                i, j = obj_id_to_idx[subj_id], obj_id_to_idx[obj_id]
                gt_rel_matrix[i, j] = rel_idx  # Direct index: 0-13 = actual relations

        # Pad to max_instances
        def pad_to_max(arr, max_len, pad_value=0):
            if len(arr) >= max_len:
                return arr[:max_len]
            pad_shape = (max_len - len(arr),) + arr.shape[1:]
            padding = np.full(pad_shape, pad_value, dtype=arr.dtype)
            return np.concatenate([arr, padding], axis=0)

        n_actual = min(n_instances, self.max_instances)
        mask = np.zeros(self.max_instances, dtype=bool)
        mask[:n_actual] = True

        return {
            # Input features
            'visual_features': torch.from_numpy(pad_to_max(visual_features, self.max_instances)),
            'geometric_features': torch.from_numpy(pad_to_max(geometric_features, self.max_instances)),
            'category_hints': torch.from_numpy(pad_to_max(category_hints, self.max_instances)),
            'centroids': torch.from_numpy(pad_to_max(centroids, self.max_instances)),
            'mask': torch.from_numpy(mask),

            # For Hungarian matching
            'sam3_categories': sam3_categories[:n_actual],
            'n_instances': n_actual,

            # GT
            'gt_classes': torch.tensor(gt_classes, dtype=torch.long),
            'gt_rel_matrix': torch.from_numpy(gt_rel_matrix),
            'n_gt': n_gt,

            # Metadata
            'take_id': take_id,
            'scan_idx': scan_idx,
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate function to handle variable-size GT."""
    # Stack fixed-size tensors
    result = {
        'visual_features': torch.stack([b['visual_features'] for b in batch]),
        'geometric_features': torch.stack([b['geometric_features'] for b in batch]),
        'category_hints': torch.stack([b['category_hints'] for b in batch]),
        'centroids': torch.stack([b['centroids'] for b in batch]),
        'mask': torch.stack([b['mask'] for b in batch]),
        'n_instances': [b['n_instances'] for b in batch],
        'sam3_categories': [b['sam3_categories'] for b in batch],

        # GT - keep as lists (variable size per sample)
        'gt_classes': [b['gt_classes'] for b in batch],
        'gt_rel_matrix': [b['gt_rel_matrix'] for b in batch],
        'n_gt': [b['n_gt'] for b in batch],

        # Metadata
        'take_id': [b['take_id'] for b in batch],
        'scan_idx': [b['scan_idx'] for b in batch],
    }

    # Add hot_rels if present (for proxy task)
    if 'hot_rels' in batch[0]:
        result['hot_rels'] = torch.stack([b['hot_rels'] for b in batch])

    # Add frame_embedding if present (for global scene context)
    if 'frame_embedding' in batch[0]:
        result['frame_embedding'] = torch.stack([b['frame_embedding'] for b in batch])

    return result
