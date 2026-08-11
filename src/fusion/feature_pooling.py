"""
Feature pooling: extract DINOv3 features per fused instance.

For each fused instance:
1. Get masks from all cameras that see it
2. For small objects (tools, hands): crop from high-res, resize to 518x518 (hybrid)
3. For large objects: use full low-res image (baseline)
4. Pool DINOv3 features within mask
5. Aggregate across cameras (multi-view fusion)

Hybrid approach gives 84x more patches for small objects with minimal slowdown.
"""

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.features.dinov3 import DINOv3Extractor
from src.fusion.instance_fusion import FusedInstance


def get_mask_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Get bounding box (x, y, w, h) of mask."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any() or not cols.any():
        return (0, 0, 0, 0)

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    return (int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1))


def crop_from_highres(
    image_highres: np.ndarray,
    mask_lowres: np.ndarray,
    target_size: int = 518,
    padding_ratio: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crop from high-res image using low-res mask coordinates.

    Args:
        image_highres: High-res RGB image [H, W, 3] (e.g., 1536x2048)
        mask_lowres: Low-res mask [H, W] (e.g., 392x518)
        target_size: Output size for crop (518 for DINOv3)
        padding_ratio: Padding around bbox (0.2 = 20%)

    Returns:
        img_crop: Cropped and resized image [target_size, target_size, 3]
        mask_crop: Cropped and resized mask [target_size, target_size]
    """
    img_H, img_W = image_highres.shape[:2]
    mask_H, mask_W = mask_lowres.shape[:2]

    scale_y = img_H / mask_H
    scale_x = img_W / mask_W

    x, y, w, h = get_mask_bbox(mask_lowres)

    if w == 0 or h == 0:
        return (
            np.zeros((target_size, target_size, 3), dtype=np.uint8),
            np.zeros((target_size, target_size), dtype=bool),
        )

    # Add padding
    pad_x = int(w * padding_ratio)
    pad_y = int(h * padding_ratio)

    # Compute crop coords in mask space
    x1_mask = max(0, x - pad_x)
    y1_mask = max(0, y - pad_y)
    x2_mask = min(mask_W, x + w + pad_x)
    y2_mask = min(mask_H, y + h + pad_y)

    # Scale to high-res image space
    x1_img = int(x1_mask * scale_x)
    y1_img = int(y1_mask * scale_y)
    x2_img = int(x2_mask * scale_x)
    y2_img = int(y2_mask * scale_y)

    # Crop from high-res image
    img_crop = image_highres[y1_img:y2_img, x1_img:x2_img]
    mask_crop = mask_lowres[y1_mask:y2_mask, x1_mask:x2_mask]

    if img_crop.size == 0:
        return (
            np.zeros((target_size, target_size, 3), dtype=np.uint8),
            np.zeros((target_size, target_size), dtype=bool),
        )

    # Resize to target size
    img_resized = cv2.resize(
        img_crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR
    )
    mask_resized = cv2.resize(
        mask_crop.astype(np.uint8),
        (target_size, target_size),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)

    return img_resized, mask_resized


class FeaturePooler:
    """
    Pool DINOv3 features per fused instance with multi-view aggregation.
    """

    def __init__(
        self,
        extractor: DINOv3Extractor,
        aggregation: str = 'mean',  # 'mean', 'max', or 'attention'
        hybrid_categories: set = None,  # Categories that use high-res extraction
    ):
        self.extractor = extractor
        self.aggregation = aggregation
        self.feat_dim = extractor.feat_dim  # 1024 for DINOv3 ViT-L
        # Default to original hybrid categories for backward compatibility
        self.hybrid_categories = hybrid_categories or {'tool', 'hand'}

    def pool_instance_features(
        self,
        instance: FusedInstance,
        images: Dict[int, torch.Tensor],  # {cam_id: [3, H, W]} low-res
        masks: Dict[int, Dict[str, np.ndarray]],  # {cam_id: {category: [N, H, W]}}
        images_highres: Optional[Dict[int, np.ndarray]] = None,  # {cam_id: [H, W, 3]} high-res
    ) -> torch.Tensor:
        """
        Pool features for a single fused instance.

        Args:
            instance: FusedInstance with camera_ids and mask_indices
            images: Low-res RGB images per camera {cam_id: [3, H, W]} (tensor)
            masks: SAM3 masks per camera {cam_id: {category: [N, H, W]}}
            images_highres: Optional high-res RGB images {cam_id: [H, W, 3]} (numpy)
                           If provided, used for hybrid extraction on small objects

        Returns:
            features: [feat_dim] pooled features for this instance
        """
        view_features = []
        use_hybrid = instance.category in self.hybrid_categories and images_highres is not None

        for cam_id in instance.camera_ids:
            if cam_id not in images or cam_id not in masks:
                continue

            # Get the specific mask for this instance in this camera
            mask = instance.get_mask(cam_id, masks)
            if mask is None:
                continue

            if use_hybrid and cam_id in images_highres:
                # Hybrid extraction for small objects (tools, hands)
                # Crop from high-res, resize to 518x518
                img_highres = images_highres[cam_id]
                img_crop, mask_crop = crop_from_highres(img_highres, mask, target_size=518)

                if mask_crop.sum() < 10:
                    continue

                # Convert to tensor
                img_tensor = torch.from_numpy(img_crop).permute(2, 0, 1).float().unsqueeze(0)
                mask_tensor = torch.from_numpy(mask_crop).unsqueeze(0)
            else:
                # Baseline extraction for large objects
                image = images[cam_id]  # [3, H, W]
                if image.dim() == 3:
                    image = image.unsqueeze(0)  # [1, 3, H, W]
                img_tensor = image
                mask_tensor = torch.from_numpy(mask).unsqueeze(0)  # [1, H, W]

            # Pool features within mask (use configured aggregation method)
            features = self.extractor.extract_masked(img_tensor, mask_tensor, pool_type=self.aggregation)  # [1, feat_dim]
            view_features.append(features.squeeze(0))  # [feat_dim]

        if not view_features:
            # No valid views - return zeros
            return torch.zeros(self.feat_dim, device=self.extractor.device)

        # Stack view features
        view_features = torch.stack(view_features)  # [n_views, feat_dim]

        # Aggregate across views
        if self.aggregation == 'mean':
            return view_features.mean(dim=0)
        elif self.aggregation == 'max':
            return view_features.max(dim=0)[0]
        elif self.aggregation == 'attention':
            # Simple self-attention aggregation
            # Compute attention weights based on feature norms
            norms = view_features.norm(dim=1, keepdim=True)  # [n_views, 1]
            weights = F.softmax(norms, dim=0)  # [n_views, 1]
            return (view_features * weights).sum(dim=0)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

    def pool_all_instances(
        self,
        instances: List[FusedInstance],
        images: Dict[int, torch.Tensor],
        masks: Dict[int, Dict[str, np.ndarray]],
        images_highres: Optional[Dict[int, np.ndarray]] = None,
    ) -> torch.Tensor:
        """
        Pool features for all fused instances.

        Args:
            instances: List of FusedInstance
            images: Low-res RGB images per camera {cam_id: [3, H, W]}
            masks: SAM3 masks per camera
            images_highres: Optional high-res RGB images {cam_id: [H, W, 3]}
                           If provided, used for hybrid extraction on tools/hands

        Returns:
            features: [n_instances, feat_dim] pooled features
        """
        if not instances:
            return torch.zeros(0, self.feat_dim, device=self.extractor.device)

        all_features = []
        for instance in instances:
            feat = self.pool_instance_features(instance, images, masks, images_highres)
            all_features.append(feat)

        return torch.stack(all_features)  # [n_instances, feat_dim]

    def get_instance_metadata(
        self,
        instances: List[FusedInstance],
    ) -> Dict:
        """
        Extract metadata for all instances (for downstream use).

        Returns dict with:
            - categories: [n_instances] category indices (0=person, 1=table, 2=anesthesia)
            - centroids: [n_instances, 3] world centroids
            - scores: [n_instances] confidence scores
            - n_views: [n_instances] number of cameras seeing each instance
        """
        category_map = {'person': 0, 'table': 1, 'anesthesia': 2, 'tool': 3, 'hand': 4}

        categories = []
        centroids = []
        scores = []
        n_views = []

        for inst in instances:
            categories.append(category_map[inst.category])
            centroids.append(inst.centroid_world)
            scores.append(inst.score)
            n_views.append(len(set(inst.camera_ids)))  # Unique cameras

        return {
            'categories': np.array(categories),
            'centroids': np.stack(centroids) if centroids else np.zeros((0, 3)),
            'scores': np.array(scores),
            'n_views': np.array(n_views),
        }
