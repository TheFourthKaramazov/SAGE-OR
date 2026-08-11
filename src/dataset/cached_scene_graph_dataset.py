"""
Cached Scene Graph Dataset for 4D-OR.

Loads pre-computed features from cache instead of running fusion + DINOv3.
Much faster than SceneGraphDataset for training.
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# GT class to index (same as scene_graph_dataset.py)
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

# SAM3 category mappings will be loaded from cache metadata
# (no longer hardcoded)


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


class CachedSceneGraphDataset(Dataset):
    """
    Fast dataset that loads pre-computed features from cache.

    Requires running src/dataset/feature_cache_generator.py first.
    """

    def __init__(
        self,
        config: Dict,
        split: str = 'train',
        max_instances: int = 32,
    ):
        self.config = config
        self.split = split
        self.max_instances = max_instances

        # Paths
        project_root = Path(__file__).parent.parent.parent
        self.feature_cache_dir = project_root / config['data']['feature_cache_dir']
        self.gt_dir = project_root / config['data']['gt_dir']

        # Load feature cache metadata (contains category mappings)
        metadata_path = self.feature_cache_dir / 'metadata.json'
        with open(metadata_path) as f:
            cache_metadata = json.load(f)

        # Load category mappings from cache (with backward compatibility for old caches)
        if 'idx_to_category' in cache_metadata:
            # New cache format (post-refactoring)
            self.idx_to_category = {int(k): v for k, v in cache_metadata['idx_to_category'].items()}
            self.category_to_idx = cache_metadata['category_to_idx']
            self.all_categories = cache_metadata['sam3_categories']
        else:
            # Old cache format (pre-refactoring) - use hardcoded mappings
            self.idx_to_category = {0: 'person', 1: 'table', 2: 'anesthesia', 3: 'tool', 4: 'hand'}
            self.category_to_idx = {'person': 0, 'table': 1, 'anesthesia': 2, 'tool': 3, 'hand': 4}
            self.all_categories = ['person', 'table', 'anesthesia', 'tool', 'hand']

        # Load GT
        self.gt_index = load_gt_relationships(self.gt_dir, split)
        self.rel_to_idx = load_relationship_classes(self.gt_dir)
        self.num_relations = len(self.rel_to_idx)

        # Get frames that have both GT and cached features
        self.frames = self._get_cached_frames()

        # Dimensions from config
        self.visual_feat_dim = config['model']['visual_feat_dim']
        self.geometric_feat_dim = config['model']['geometric_feat_dim']
        # Use cache metadata if available, otherwise use config
        self.category_hint_dim = cache_metadata.get('category_hint_dim', config['model']['category_hint_dim'])

        # Ablation: exclude categories (e.g., hand, knee, door)
        self.exclude_categories = set(config['data'].get('exclude_categories', []))
        if self.exclude_categories:
            print(f"[{split}] ABLATION: Excluding categories: {sorted(self.exclude_categories)}")

        # Backward compatibility: support exclude_hands flag
        if config['data'].get('exclude_hands', False):
            self.exclude_categories.add('hand')
            print(f"[{split}] ABLATION: Excluding hand instances (via exclude_hands flag)")



    def _get_cached_frames(self) -> List[Tuple[int, int]]:
        """Get list of (take_id, scan_idx) that have cached features."""
        frames = []

        for (take_idx, scan_idx) in self.gt_index.keys():
            take_str = f'take{take_idx:02d}'
            cache_path = self.feature_cache_dir / take_str / f'frame_{scan_idx:04d}.npz'

            if cache_path.exists():
                frames.append((take_idx, scan_idx))

        return sorted(frames)

    def __len__(self) -> int:
        return len(self.frames)

    def _get_gt_for_frame(self, take_id: int, scan_idx: int) -> Dict:
        """Get GT labels for a frame."""
        gt_data = self.gt_index.get((take_id, scan_idx), {'objects': {}, 'relationships': []})

        gt_objects = gt_data.get('objects', {})
        gt_relationships = gt_data.get('relationships', [])

        return {
            'objects': gt_objects,
            'relationships': gt_relationships,
        }

    def __getitem__(self, idx: int) -> Dict:
        take_id, scan_idx = self.frames[idx]

        # Load cached features (use context manager to avoid file descriptor leak)
        take_str = f'take{take_id:02d}'
        cache_path = self.feature_cache_dir / take_str / f'frame_{scan_idx:04d}.npz'

        with np.load(cache_path) as cache:
            visual_features = cache['visual_features'].astype(np.float32)
            geometric_features = cache['geometric_features'].astype(np.float32)
            category_hints = cache['category_hints'].astype(np.float32)
            centroids = cache['centroids'].astype(np.float32)
            sam3_category_indices = cache['sam3_categories'].copy()
            n_instances = int(cache['n_instances'])

        # Load frame-level embedding (global scene context) if available
        frame_embedding_path = self.feature_cache_dir / take_str / f'frame_{scan_idx:04d}_embedding.npy'
        if frame_embedding_path.exists():
            frame_embedding = np.load(frame_embedding_path).astype(np.float32)
        else:
            # Fallback: zero embedding if not cached (shouldn't happen after generation complete)
            frame_embedding = np.zeros(1024, dtype=np.float32)

        # Convert SAM3 category indices back to strings
        sam3_categories = [self.idx_to_category[i] for i in sam3_category_indices]

        # ABLATION: Filter out excluded categories (e.g., hand, knee, door)
        if self.exclude_categories:
            # Create mask: True for instances not in excluded categories
            keep_mask = np.array([cat not in self.exclude_categories for cat in sam3_categories])

            # Apply filter to all arrays
            visual_features = visual_features[keep_mask]
            geometric_features = geometric_features[keep_mask]
            category_hints = category_hints[keep_mask]
            centroids = centroids[keep_mask]
            sam3_categories = [cat for cat in sam3_categories if cat not in self.exclude_categories]

            # Update instance count
            n_instances = len(sam3_categories)

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
        # 0-13 = actual relations, 14 = 'none'
        n_gt = len(gt_classes)
        none_idx = self.num_relations  # 'none' is at index 14
        gt_rel_matrix = np.full((n_gt, n_gt), none_idx, dtype=np.int64)

        obj_id_to_idx = {obj_id: i for i, obj_id in enumerate(gt_obj_ids)}

        for rel in gt['relationships']:
            subj_id, obj_id, rel_idx = rel[0], rel[1], rel[2]
            if subj_id in obj_id_to_idx and obj_id in obj_id_to_idx:
                i, j = obj_id_to_idx[subj_id], obj_id_to_idx[obj_id]
                gt_rel_matrix[i, j] = rel_idx

        # Compute hot_rels: binary vector of which relations exist in this frame
        # Used for proxy task (frame-level relationship existence prediction)
        hot_rels = np.zeros(self.num_relations + 1, dtype=np.float32)  # +1 for 'none'
        for rel in gt['relationships']:
            rel_idx = rel[2]
            if rel_idx < self.num_relations:  # Exclude 'none' from hot_rels
                hot_rels[rel_idx] = 1.0

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

        # Build result dict
        result = {
            # Input features
            'visual_features': torch.from_numpy(pad_to_max(visual_features, self.max_instances)),
            'geometric_features': torch.from_numpy(pad_to_max(geometric_features, self.max_instances)),
            'category_hints': torch.from_numpy(pad_to_max(category_hints, self.max_instances)),
            'centroids': torch.from_numpy(pad_to_max(centroids, self.max_instances)),
            'mask': torch.from_numpy(mask),
            'frame_embedding': torch.from_numpy(frame_embedding),  # Global scene context

            # For Hungarian matching
            'sam3_categories': sam3_categories[:n_actual],
            'n_instances': n_actual,

            # GT
            'gt_classes': torch.tensor(gt_classes, dtype=torch.long),
            'gt_rel_matrix': torch.from_numpy(gt_rel_matrix),
            'hot_rels': torch.from_numpy(hot_rels),  # For proxy task
            'n_gt': n_gt,

            # Metadata
            'take_id': take_id,
            'scan_idx': scan_idx,
        }

        return result


# Re-export collate_fn from scene_graph_dataset (same logic)
from src.dataset.scene_graph_dataset import collate_fn
