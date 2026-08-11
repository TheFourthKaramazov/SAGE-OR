"""
Instance fusion: match masks across cameras using 3D overlap.

Given:
- SAM3 masks per camera (person, table, anesthesia)
- Pi3 pointmaps per camera
- Camera extrinsics

Output:
- Fused instances (masks from multiple cameras grouped by 3D overlap)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist


@dataclass
class FusedInstance:
    """A single fused instance from multiple camera views."""
    category: str  # Final voted category: 'person', 'table', 'anesthesia', 'tool', 'hand'
    instance_id: int

    # Which cameras see this instance
    camera_ids: List[int]

    # Mask index per camera (camera_id -> mask_idx in that camera's mask array)
    # For cross-category fusion, this maps to the mask in the ORIGINAL category
    mask_indices: Dict[int, int]

    # Original category per camera (camera_id -> original SAM3 category)
    # Only populated in cross-category mode; may differ from final voted category
    mask_categories: Optional[Dict[int, str]] = None

    # 3D centroid in world coordinates
    centroid_world: np.ndarray = None  # [3]

    # Fused 3D points (sampled, for geometry features)
    points_world: np.ndarray = None  # [N, 3]

    # Confidence score (average of mask scores)
    score: float = 0.0

    def get_mask(self, cam_id: int, masks_per_camera: Dict[int, Dict[str, np.ndarray]]) -> Optional[np.ndarray]:
        """
        Get the mask for this instance from a specific camera.

        Args:
            cam_id: Camera ID
            masks_per_camera: {cam_id: {category: [N, H, W]}}

        Returns:
            mask: [H, W] boolean mask or None if not visible in this camera
        """
        if cam_id not in self.mask_indices:
            return None

        mask_idx = self.mask_indices[cam_id]

        # Determine which category to look up
        if self.mask_categories and cam_id in self.mask_categories:
            # Cross-category mode: use original category
            cat = self.mask_categories[cam_id]
        else:
            # Per-category mode: use final category
            cat = self.category

        if cam_id not in masks_per_camera:
            return None
        if cat not in masks_per_camera[cam_id]:
            return None
        if mask_idx >= len(masks_per_camera[cam_id][cat]):
            return None

        return masks_per_camera[cam_id][cat][mask_idx]


class InstanceFusion:
    """
    Fuse masks across cameras using 3D overlap.

    Algorithm:
    1. For each mask, extract 3D points using Pi3 pointmap
    2. Transform to world coordinates
    3. Compute 3D centroids per mask
    4. Cluster masks by centroid distance (within each category)
    5. Merge overlapping masks into fused instances
    """

    def __init__(
        self,
        centroid_threshold: float = 0.5,  # Max distance (meters) to consider same instance
        min_points: int = 100,  # Min valid points to consider a mask
        conf_threshold: float = 0.3,  # Pi3 confidence threshold
        max_points_per_mask: int = 1000,  # Subsample for efficiency
        categories: list = None,  # SAM3 categories to process
    ):
        self.centroid_threshold = centroid_threshold
        self.min_points = min_points
        self.conf_threshold = conf_threshold
        self.max_points_per_mask = max_points_per_mask
        # Default to original 5 categories for backward compatibility
        self.categories = categories or ['person', 'table', 'anesthesia', 'tool', 'hand']

    def extract_mask_points(
        self,
        mask: np.ndarray,
        pointmap: np.ndarray,
        confidence: np.ndarray,
        camera_pose: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extract 3D points within a mask and transform to world coordinates.

        Args:
            mask: [H, W] boolean mask
            pointmap: [H, W, 3] XYZ in camera coordinates
            confidence: [H, W] Pi3 confidence
            camera_pose: [4, 4] camera-to-world transformation

        Returns:
            points_world: [N, 3] points in world coordinates (or None if too few)
            centroid_world: [3] centroid in world coordinates (or None)
        """
        # Get valid points within mask
        valid = mask & (confidence > self.conf_threshold)

        if valid.sum() < self.min_points:
            return None, None

        # Extract points
        points_local = pointmap[valid]  # [N, 3]

        # Subsample if too many
        if len(points_local) > self.max_points_per_mask:
            indices = np.random.choice(len(points_local), self.max_points_per_mask, replace=False)
            points_local = points_local[indices]

        # Transform to world coordinates
        # points_world = R @ points_local + t
        R = camera_pose[:3, :3]
        t = camera_pose[:3, 3]
        points_world = (R @ points_local.T).T + t

        # Compute centroid
        centroid_world = points_world.mean(axis=0)

        return points_world, centroid_world

    def compute_3d_iou(
        self,
        points1: np.ndarray,
        points2: np.ndarray,
        voxel_size: float = 0.05,
    ) -> float:
        """
        Compute 3D IoU between two point clouds using voxelization.

        Args:
            points1: [N, 3] first point cloud
            points2: [M, 3] second point cloud
            voxel_size: Size of voxels for discretization

        Returns:
            iou: Intersection over union (0 to 1)
        """
        # Voxelize both point clouds
        def voxelize(points):
            voxels = np.floor(points / voxel_size).astype(int)
            return set(map(tuple, voxels))

        voxels1 = voxelize(points1)
        voxels2 = voxelize(points2)

        intersection = len(voxels1 & voxels2)
        union = len(voxels1 | voxels2)

        if union == 0:
            return 0.0

        return intersection / union

    def cluster_by_iou(
        self,
        mask_infos: List[dict],
        iou_threshold: float = 0.1,
    ) -> np.ndarray:
        """
        Cluster masks by 3D IoU using greedy merging.

        Args:
            mask_infos: List of dicts with 'points' key
            iou_threshold: Min IoU to merge (0.1 = 10% overlap)

        Returns:
            labels: Cluster label per mask
        """
        n = len(mask_infos)
        if n <= 1:
            return np.zeros(n, dtype=int)

        # Compute pairwise IoU
        iou_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                iou = self.compute_3d_iou(
                    mask_infos[i]['points'],
                    mask_infos[j]['points'],
                )
                iou_matrix[i, j] = iou
                iou_matrix[j, i] = iou

        # Greedy clustering: merge if IoU > threshold
        labels = np.arange(n)  # Start with each mask as its own cluster

        for i in range(n):
            for j in range(i + 1, n):
                if iou_matrix[i, j] >= iou_threshold:
                    # Merge clusters
                    old_label = labels[j]
                    new_label = labels[i]
                    labels[labels == old_label] = new_label

        # Relabel to consecutive integers
        unique_labels = np.unique(labels)
        label_map = {old: new for new, old in enumerate(unique_labels)}
        labels = np.array([label_map[l] for l in labels])

        return labels

    def cluster_by_iou_with_category_constraint(
        self,
        mask_infos: List[dict],
        iou_threshold: float = 0.1,
        voxel_size: float = 0.05,
    ) -> np.ndarray:
        """
        Cluster masks by 3D IoU, but only merge if categories are compatible.

        Two masks can only merge if:
        1. They belong to the same category group
        2. Their 3D IoU exceeds iou_threshold

        Note: We don't use centroid distance because Pi3 depth is relative/
        scale-ambiguous. IoU is scale-invariant when using adaptive voxel_size.

        Category groups:
        - Group 0: person
        - Group 1: table, anesthesia
        - Group 2: tool, hand

        Args:
            mask_infos: List of dicts with 'points', 'category_group' keys
            iou_threshold: Min IoU to merge
            voxel_size: Voxel size for IoU computation (should be relative to scene scale)

        Returns:
            labels: Cluster label per mask
        """
        n = len(mask_infos)
        if n <= 1:
            return np.zeros(n, dtype=int)

        # Compute pairwise IoU (only for compatible category pairs)
        iou_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                # Check: Categories must be compatible
                if mask_infos[i]['category_group'] != mask_infos[j]['category_group']:
                    continue

                # Compute IoU with adaptive voxel size
                iou = self.compute_3d_iou(
                    mask_infos[i]['points'],
                    mask_infos[j]['points'],
                    voxel_size=voxel_size,
                )
                iou_matrix[i, j] = iou
                iou_matrix[j, i] = iou

        # Greedy clustering: merge if IoU > threshold
        labels = np.arange(n)

        for i in range(n):
            for j in range(i + 1, n):
                if iou_matrix[i, j] >= iou_threshold:
                    old_label = labels[j]
                    new_label = labels[i]
                    labels[labels == old_label] = new_label

        # Relabel to consecutive integers
        unique_labels = np.unique(labels)
        label_map = {old: new for new, old in enumerate(unique_labels)}
        labels = np.array([label_map[l] for l in labels])

        return labels

    def cluster_by_centroid(
        self,
        centroids: List[np.ndarray],
        threshold: float,
    ) -> np.ndarray:
        """
        Cluster centroids using hierarchical clustering.
        DEPRECATED: Use cluster_by_iou for better results.

        Args:
            centroids: List of [3] centroid positions
            threshold: Max distance to merge

        Returns:
            labels: Cluster label per centroid
        """
        if len(centroids) <= 1:
            return np.zeros(len(centroids), dtype=int)

        centroids_array = np.stack(centroids)

        # Compute pairwise distances
        distances = pdist(centroids_array)

        # Hierarchical clustering
        Z = linkage(distances, method='average')

        # Cut at threshold
        labels = fcluster(Z, t=threshold, criterion='distance') - 1  # 0-indexed

        return labels

    def fuse_frame(
        self,
        masks_per_camera: Dict[int, Dict[str, np.ndarray]],
        scores_per_camera: Dict[int, Dict[str, np.ndarray]],
        pointmaps: np.ndarray,
        confidences: np.ndarray,
        camera_poses: np.ndarray,
        camera_ids: np.ndarray,
        cross_category: bool = True,
    ) -> List[FusedInstance]:
        """
        Fuse all masks for a single frame across cameras.

        Args:
            masks_per_camera: {cam_id: {'person': [N,H,W], 'table': [M,H,W], ...}}
            scores_per_camera: {cam_id: {'person': [N], 'table': [M], ...}}
            pointmaps: [num_cams, H, W, 3] Pi3 pointmaps
            confidences: [num_cams, H, W] Pi3 confidence
            camera_poses: [num_cams, 4, 4] extrinsics
            camera_ids: [num_cams] camera IDs (1-6)
            cross_category: If True, cluster across categories then vote (fixes cross-view inconsistency)

        Returns:
            List of FusedInstance
        """
        if cross_category:
            return self._fuse_frame_cross_category(
                masks_per_camera, scores_per_camera, pointmaps,
                confidences, camera_poses, camera_ids
            )
        else:
            return self._fuse_frame_per_category(
                masks_per_camera, scores_per_camera, pointmaps,
                confidences, camera_poses, camera_ids
            )

    # Category compatibility groups for cross-category fusion
    # Only categories in the same group can be merged (SAM3 confusion)
    # Categories in different groups are different object types
    CATEGORY_GROUPS = {
        'person': 0,      # People are distinct
        'table': 1,       # Large equipment group
        'anesthesia': 1,  # Large equipment group (often confused with table)
        'tool': 2,        # Small items group
        'hand': 2,        # Small items group (tools in hands)
    }

    def _compute_scene_scale(self, mask_infos: List[dict]) -> float:
        """
        Compute scene scale from all points for adaptive voxel sizing.

        Returns the diagonal extent of the bounding box containing all points.
        This makes IoU computation robust to Pi3's arbitrary depth scale.
        """
        if not mask_infos:
            return 1.0

        all_points = np.vstack([info['points'] for info in mask_infos])

        # Compute bounding box diagonal
        min_coords = all_points.min(axis=0)
        max_coords = all_points.max(axis=0)
        diagonal = np.linalg.norm(max_coords - min_coords)

        return max(diagonal, 0.1)  # Avoid division by zero

    def _fuse_frame_cross_category(
        self,
        masks_per_camera: Dict[int, Dict[str, np.ndarray]],
        scores_per_camera: Dict[int, Dict[str, np.ndarray]],
        pointmaps: np.ndarray,
        confidences: np.ndarray,
        camera_poses: np.ndarray,
        camera_ids: np.ndarray,
    ) -> List[FusedInstance]:
        """
        Cross-category fusion: cluster masks by 3D overlap within compatible category groups.

        Category groups:
        - Group 0: person (never confused with objects)
        - Group 1: table, anesthesia (large equipment, SAM3 confuses these)
        - Group 2: tool, hand (small items, tools in hands)

        This fixes cross-view category inconsistency while preventing
        incorrect merges (e.g., person + table).
        """
        # Step 1: Collect ALL masks across ALL categories
        all_mask_infos = []

        for cam_idx, cam_id in enumerate(camera_ids):
            cam_id = int(cam_id)

            if cam_id not in masks_per_camera:
                continue

            for category in self.categories:
                if category not in masks_per_camera[cam_id]:
                    continue

                masks = masks_per_camera[cam_id][category]
                scores = scores_per_camera[cam_id][category]

                for mask_idx in range(len(masks)):
                    mask = masks[mask_idx]
                    score = scores[mask_idx]

                    # Extract 3D points
                    points, centroid = self.extract_mask_points(
                        mask,
                        pointmaps[cam_idx],
                        confidences[cam_idx],
                        camera_poses[cam_idx],
                    )

                    if centroid is not None:
                        # Unknown categories get unique group ID (prevents incorrect merging)
                        # Use hash for deterministic assignment
                        default_group = 100 + (hash(category) % 900)
                        cat_group = self.CATEGORY_GROUPS.get(category, default_group)

                        all_mask_infos.append({
                            'cam_idx': cam_idx,
                            'cam_id': cam_id,
                            'mask_idx': mask_idx,
                            'category': category,
                            'category_group': cat_group,
                            'centroid': centroid,
                            'points': points,
                            'score': score,
                        })

        if not all_mask_infos:
            return []

        # Step 2: Cluster masks by 3D IoU, but only within compatible category groups
        # Note: Pi3 depth is relative (not metric), so centroid distance thresholds
        # are meaningless. We rely on IoU (scale-invariant) + category constraints.
        # Compute adaptive voxel size based on scene scale for robust IoU.
        scene_scale = self._compute_scene_scale(all_mask_infos)
        voxel_size = scene_scale * 0.02  # 2% of scene extent

        labels = self.cluster_by_iou_with_category_constraint(
            all_mask_infos, iou_threshold=0.15, voxel_size=voxel_size
        )

        # Step 3: For each cluster, vote on category and create fused instance
        fused_instances = []
        n_clusters = labels.max() + 1

        for cluster_id in range(n_clusters):
            cluster_mask = labels == cluster_id
            cluster_infos = [all_mask_infos[i] for i in range(len(all_mask_infos)) if cluster_mask[i]]

            if not cluster_infos:
                continue

            # Vote on category (weighted by confidence score)
            category_votes = {}
            for info in cluster_infos:
                cat = info['category']
                score = info['score']
                if cat not in category_votes:
                    category_votes[cat] = 0.0
                category_votes[cat] += score

            # Pick category with highest weighted vote
            final_category = max(category_votes.keys(), key=lambda c: category_votes[c])

            # Build mask_indices and mask_categories
            camera_ids_list = []
            mask_indices = {}      # cam_id -> mask_idx
            mask_categories = {}   # cam_id -> original category

            for info in cluster_infos:
                cam_id = info['cam_id']
                if cam_id not in mask_indices:
                    camera_ids_list.append(cam_id)
                    mask_indices[cam_id] = info['mask_idx']
                    mask_categories[cam_id] = info['category']
                # If multiple masks from same camera in same cluster, keep highest score
                else:
                    existing_cat = mask_categories[cam_id]
                    existing_idx = mask_indices[cam_id]
                    existing_score = scores_per_camera[cam_id][existing_cat][existing_idx]
                    if info['score'] > existing_score:
                        mask_indices[cam_id] = info['mask_idx']
                        mask_categories[cam_id] = info['category']

            # Combine points from all views
            all_points = np.vstack([info['points'] for info in cluster_infos])

            # Subsample combined points
            if len(all_points) > self.max_points_per_mask:
                indices = np.random.choice(len(all_points), self.max_points_per_mask, replace=False)
                all_points = all_points[indices]

            # Compute fused centroid
            centroid = all_points.mean(axis=0)

            # Average score
            avg_score = np.mean([info['score'] for info in cluster_infos])

            fused_instances.append(FusedInstance(
                category=final_category,
                instance_id=cluster_id,
                camera_ids=camera_ids_list,
                mask_indices=mask_indices,
                mask_categories=mask_categories,  # Original SAM3 categories per camera
                centroid_world=centroid,
                points_world=all_points,
                score=float(avg_score),
            ))

        return fused_instances

    def _fuse_frame_per_category(
        self,
        masks_per_camera: Dict[int, Dict[str, np.ndarray]],
        scores_per_camera: Dict[int, Dict[str, np.ndarray]],
        pointmaps: np.ndarray,
        confidences: np.ndarray,
        camera_poses: np.ndarray,
        camera_ids: np.ndarray,
    ) -> List[FusedInstance]:
        """
        Original per-category fusion (kept for comparison).
        """
        fused_instances = []
        instance_counter = 0

        # Process each category separately
        for category in self.categories:
            # Collect all masks for this category across cameras
            mask_infos = []

            for cam_idx, cam_id in enumerate(camera_ids):
                cam_id = int(cam_id)

                if cam_id not in masks_per_camera:
                    continue

                if category not in masks_per_camera[cam_id]:
                    continue

                masks = masks_per_camera[cam_id][category]
                scores = scores_per_camera[cam_id][category]

                for mask_idx in range(len(masks)):
                    mask = masks[mask_idx]
                    score = scores[mask_idx]

                    # Extract 3D points
                    points, centroid = self.extract_mask_points(
                        mask,
                        pointmaps[cam_idx],
                        confidences[cam_idx],
                        camera_poses[cam_idx],
                    )

                    if centroid is not None:
                        mask_infos.append({
                            'cam_idx': cam_idx,
                            'cam_id': cam_id,
                            'mask_idx': mask_idx,
                            'centroid': centroid,
                            'points': points,
                            'score': score,
                        })

            if not mask_infos:
                continue

            # Cluster by 3D IoU
            iou_thresholds = {
                'person': 0.05,
                'table': 0.10,
                'anesthesia': 0.07,
                'tool': 0.05,
                'hand': 0.05,
            }
            labels = self.cluster_by_iou(mask_infos, iou_threshold=iou_thresholds.get(category, 0.07))

            # Group masks by cluster
            n_clusters = labels.max() + 1

            for cluster_id in range(n_clusters):
                cluster_mask = labels == cluster_id
                cluster_infos = [mask_infos[i] for i in range(len(mask_infos)) if cluster_mask[i]]

                camera_ids_list = [info['cam_id'] for info in cluster_infos]
                mask_indices = {info['cam_id']: info['mask_idx'] for info in cluster_infos}

                all_points = np.vstack([info['points'] for info in cluster_infos])

                if len(all_points) > self.max_points_per_mask:
                    indices = np.random.choice(len(all_points), self.max_points_per_mask, replace=False)
                    all_points = all_points[indices]

                centroid = all_points.mean(axis=0)
                avg_score = np.mean([info['score'] for info in cluster_infos])

                fused_instances.append(FusedInstance(
                    category=category,
                    instance_id=instance_counter,
                    camera_ids=camera_ids_list,
                    mask_indices=mask_indices,
                    centroid_world=centroid,
                    points_world=all_points,
                    score=float(avg_score),
                ))
                instance_counter += 1

        return fused_instances
