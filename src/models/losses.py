"""
Loss functions for scene graph prediction.

Hungarian matching for set prediction + classification losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Tuple, Optional
from collections import Counter
import numpy as np


def compute_class_weights(
    dataset,
    num_classes: int = 12,
    num_relations: int = 15,
    no_relation_weight: float = 0.0001,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute balanced class weights from training data.

    Uses linear inverse frequency (like LABRAD-OR):
    - Node weights: 1.0 / (log(count) + 1)
    - Relation weights: 1.0 / count
    - 'none' relation: fixed to no_relation_weight

    Args:
        dataset: Training dataset to compute weights from
        num_classes: Number of node classes
        num_relations: Number of relation classes
        no_relation_weight: Weight for 'none' relation

    Returns:
        node_weights: [num_classes] tensor
        rel_weights: [num_relations] tensor
    """
    node_counts = Counter()
    rel_counts = Counter()

    print("Computing class weights from training data...")
    for i in range(len(dataset)):
        sample = dataset[i]

        # Count node classes
        gt_classes = sample['gt_classes']
        for c in gt_classes.tolist():
            node_counts[c] += 1

        # Count relation classes (from relation matrix)
        gt_rel_matrix = sample['gt_rel_matrix']
        for r in gt_rel_matrix.flatten().tolist():
            rel_counts[r] += 1

    # Compute node weights: 1.0 / (log(count) + 1)
    node_weights = torch.ones(num_classes)
    for c in range(num_classes):
        count = node_counts.get(c, 1)  # Avoid division by zero
        node_weights[c] = 1.0 / (np.log(count) + 1)

    # Compute relation weights: 1.0 / count (linear inverse)
    rel_weights = torch.ones(num_relations)
    for r in range(num_relations):
        count = rel_counts.get(r, 1)  # Avoid division by zero
        rel_weights[r] = 1.0 / count

    # Override 'none' relation weight
    rel_weights[-1] = no_relation_weight

    # Normalize weights so they sum to num_classes (optional, for stability)
    # node_weights = node_weights / node_weights.sum() * num_classes
    # rel_weights = rel_weights / rel_weights.sum() * num_relations

    print(f"Node weights: min={node_weights.min():.4f}, max={node_weights.max():.4f}")
    print(f"Relation weights: min={rel_weights.min():.6f}, max={rel_weights.max():.4f}")

    return node_weights, rel_weights


def compute_proxy_pos_weight(
    dataset,
    num_relations: int = 15,
) -> torch.Tensor:
    """
    Compute pos_weight for proxy task BCE loss.

    pos_weight[i] = num_negatives[i] / num_positives[i]

    This balances the loss so rare relationships get more weight.
    """
    pos_counts = np.zeros(num_relations)
    neg_counts = np.zeros(num_relations)

    print("Computing proxy pos_weight from training data...")
    for i in range(len(dataset)):
        sample = dataset[i]
        hot_rels = sample.get('hot_rels', None)

        if hot_rels is not None:
            for r in range(num_relations):
                if hot_rels[r] > 0:
                    pos_counts[r] += 1
                else:
                    neg_counts[r] += 1

    # pos_weight = neg / pos (higher weight for rare positives)
    pos_weight = torch.ones(num_relations)
    for r in range(num_relations):
        if pos_counts[r] > 0:
            pos_weight[r] = neg_counts[r] / pos_counts[r]
        else:
            pos_weight[r] = 1.0  # Default if no positives

    print(f"Proxy pos_weight: min={pos_weight.min():.2f}, max={pos_weight.max():.2f}")

    return pos_weight


class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced classification.

    FL(p) = -α * (1-p)^γ * log(p)

    Down-weights easy examples, focuses learning on hard examples.
    Reference: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = 'mean',
    ):
        """
        Args:
            gamma: Focusing parameter. Higher = more focus on hard examples.
                   γ=0 is equivalent to CE. γ=2 is typical.
            alpha: Per-class weights [num_classes]. Optional.
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [N, C] logits
            targets: [N] class indices

        Returns:
            Focal loss (scalar if reduction='mean' or 'sum')
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # p_t = probability of correct class

        focal_weight = (1 - pt) ** self.gamma
        focal_loss = focal_weight * ce_loss

        # Apply class weights if provided
        if self.alpha is not None:
            alpha_t = self.alpha.to(inputs.device)[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# Category compatibility: SAM3 category → valid GT class indices
CATEGORY_COMPATIBILITY = {
    'person': [0, 2, 3, 4, 5, 6, 7],  # Patient, human_0-5
    'table': [9, 10, 11],              # instrument_table, operating_table, secondary_table
    'anesthesia': [1],                  # anesthesia_equipment
    'tool': [8],                        # instrument
    'hand': [],                         # No GT mapping
}


class HungarianMatcher:
    """
    Hungarian matching between predictions and GT.

    Finds optimal bipartite assignment based on:
    - Class probability cost
    - Category compatibility constraints
    """

    def __init__(self, cost_class: float = 1.0):
        self.cost_class = cost_class

    @torch.no_grad()
    def match(
        self,
        node_logits: torch.Tensor,       # [N, num_classes]
        gt_classes: torch.Tensor,         # [M]
        sam3_categories: List[str],       # [N] SAM3 category strings
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find optimal matching between N predictions and M GT objects.

        Returns:
            pred_indices: [K] indices of matched predictions
            gt_indices: [K] indices of matched GT objects
            K = min(N, M) after filtering incompatible pairs
        """
        N = node_logits.shape[0]
        M = gt_classes.shape[0]

        if N == 0 or M == 0:
            return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)

        # Class probability cost: 1 - P(gt_class)
        # [N, num_classes] -> [N, M]
        probs = F.softmax(node_logits, dim=-1)
        class_cost = 1 - probs[:, gt_classes]  # [N, M]

        # Category compatibility mask
        # Set cost = infinity for incompatible pairs
        compat_mask = torch.zeros(N, M, dtype=torch.bool, device=node_logits.device)

        for i, sam3_cat in enumerate(sam3_categories):
            valid_gt_classes = CATEGORY_COMPATIBILITY.get(sam3_cat, [])
            for j in range(M):
                gt_cls = gt_classes[j].item()
                if gt_cls in valid_gt_classes:
                    compat_mask[i, j] = True

        # Apply compatibility constraint
        cost_matrix = self.cost_class * class_cost
        cost_matrix[~compat_mask] = 1e6  # Large cost for incompatible pairs

        # Hungarian algorithm
        cost_np = cost_matrix.cpu().numpy()
        pred_indices, gt_indices = linear_sum_assignment(cost_np)

        # Filter out incompatible matches (cost = 1e6)
        valid = cost_np[pred_indices, gt_indices] < 1e5
        pred_indices = pred_indices[valid]
        gt_indices = gt_indices[valid]

        return (
            torch.tensor(pred_indices, dtype=torch.long, device=node_logits.device),
            torch.tensor(gt_indices, dtype=torch.long, device=node_logits.device),
        )


class SceneGraphLoss(nn.Module):
    """
    Loss for scene graph prediction.

    Components:
    - Node classification loss (CE with Hungarian matching)
    - Edge/relationship classification loss (CE)

    Matches 4D-OR: 15 relation classes (14 actual + 'none' at index 14)
    4D-OR sets 'none' weight to 0.0001
    """

    def __init__(
        self,
        num_classes: int = 12,
        num_relations: int = 15,  # 14 relations + 'none' (matches 4D-OR)
        cost_class: float = 1.0,
        node_weight: float = 1.0,
        edge_weight: float = 1.0,
        no_relation_weight: float = 0.0001,  # 4D-OR uses 0.0001 for 'none'
        use_focal_loss: bool = False,
        focal_gamma: float = 2.0,
        rel_class_weights: Optional[torch.Tensor] = None,  # Precomputed balanced weights
        use_proxy_task: bool = False,
        proxy_weight: float = 1.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_relations = num_relations
        self.node_weight = node_weight
        self.edge_weight = edge_weight
        self.use_focal_loss = use_focal_loss
        self.use_proxy_task = use_proxy_task
        self.proxy_weight = proxy_weight

        # Hungarian matcher
        self.matcher = HungarianMatcher(cost_class=cost_class)

        # Edge loss with class weighting
        if rel_class_weights is not None:
            # Use precomputed balanced weights
            edge_weights = rel_class_weights.clone()
        else:
            # Default: all 1.0 except 'none'
            edge_weights = torch.ones(num_relations)
            edge_weights[-1] = no_relation_weight
        self.register_buffer('edge_weights', edge_weights)

        # Focal loss for edges (optional)
        if use_focal_loss:
            self.edge_focal_loss = FocalLoss(gamma=focal_gamma, alpha=edge_weights)
        else:
            self.edge_focal_loss = None

        # Proxy task: compute pos_weight for BCE (handle class imbalance)
        # pos_weight will be set from training data if proxy task is enabled
        if use_proxy_task:
            self.register_buffer('proxy_pos_weight', torch.ones(num_relations))

    def set_proxy_pos_weight(self, pos_weight: torch.Tensor):
        """Set the proxy task pos_weight (computed from training data)."""
        self.proxy_pos_weight.copy_(pos_weight)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute losses.

        Args:
            outputs: Model outputs with node_logits [B, N, C], edge_logits [B, N, N, R]
            batch: Batch from dataloader with GT

        Returns:
            dict with 'total', 'node', 'edge' losses
        """
        node_logits = outputs['node_logits']  # [B, N, num_classes]
        edge_logits = outputs['edge_logits']  # [B, N, N, num_relations]
        mask = batch['mask']  # [B, N]

        B = node_logits.shape[0]

        total_node_loss = 0.0
        total_edge_loss = 0.0
        n_matched = 0
        n_edges = 0

        for b in range(B):
            # Get valid predictions for this sample
            valid_mask = mask[b]
            n_pred = valid_mask.sum().item()

            if n_pred == 0:
                continue

            pred_logits = node_logits[b, :n_pred]  # [n_pred, C]
            sam3_cats = batch['sam3_categories'][b]
            gt_classes = batch['gt_classes'][b].to(node_logits.device)
            gt_rel_matrix = batch['gt_rel_matrix'][b].to(node_logits.device)

            if len(gt_classes) == 0:
                continue

            # Hungarian matching
            pred_idx, gt_idx = self.matcher.match(pred_logits, gt_classes, sam3_cats)

            if len(pred_idx) == 0:
                continue

            # Node classification loss on matched pairs
            matched_logits = pred_logits[pred_idx]  # [K, C]
            matched_gt = gt_classes[gt_idx]  # [K]
            node_loss = F.cross_entropy(matched_logits, matched_gt)
            total_node_loss += node_loss
            n_matched += 1

            # Edge classification loss
            # Only compute for matched prediction pairs
            K = len(pred_idx)
            if K >= 2:
                # Get edge logits for matched pairs [K, K, R]
                matched_edge_logits = edge_logits[b, pred_idx][:, pred_idx]

                # Get GT relations for matched pairs [K, K]
                # Map from GT indices to our matched order
                matched_gt_rels = gt_rel_matrix[gt_idx][:, gt_idx]

                # Flatten for loss computation
                edge_logits_flat = matched_edge_logits.view(-1, self.num_relations)  # [K*K, R]
                gt_rels_flat = matched_gt_rels.view(-1)  # [K*K]

                # Encoding matches 4D-OR:
                # GT: 0-13 = actual relations (Assisting, Cementing, etc.), 14 = 'none'
                # Model predicts: same encoding (0-13 = actual, 14 = 'none')

                if self.use_focal_loss:
                    edge_loss = self.edge_focal_loss(edge_logits_flat, gt_rels_flat)
                else:
                    edge_loss = F.cross_entropy(
                        edge_logits_flat,
                        gt_rels_flat,
                        weight=self.edge_weights.to(edge_logits.device),
                    )
                total_edge_loss += edge_loss
                n_edges += 1

        # Average losses
        node_loss = total_node_loss / max(n_matched, 1)
        edge_loss = total_edge_loss / max(n_edges, 1)

        total_loss = self.node_weight * node_loss + self.edge_weight * edge_loss

        result = {
            'total': total_loss,
            'node': node_loss,
            'edge': edge_loss,
            'n_matched': n_matched,
        }

        # Proxy task loss (frame-level relationship existence)
        if self.use_proxy_task and 'proxy_logits' in outputs and 'hot_rels' in batch:
            proxy_logits = outputs['proxy_logits']  # [B, num_relations=15]
            hot_rels = batch['hot_rels'].to(proxy_logits.device)  # [B, num_relations=15]

            # Exclude 'none' from both proxy_logits and hot_rels (last element)
            proxy_logits = proxy_logits[:, :-1]  # [B, 14]
            hot_rels = hot_rels[:, :-1]  # [B, 14]

            # Also exclude 'none' from pos_weight
            pos_weight = self.proxy_pos_weight[:-1].to(proxy_logits.device)  # [14]

            proxy_loss = F.binary_cross_entropy_with_logits(
                proxy_logits,
                hot_rels,
                pos_weight=pos_weight,
            )

            total_loss = total_loss + self.proxy_weight * proxy_loss
            result['total'] = total_loss
            result['proxy'] = proxy_loss

        return result


class SceneGraphMetrics:
    """
    Compute evaluation metrics for scene graph prediction.

    Matches official 4D-OR evaluation (used by LABRAD-OR):
    - sklearn.metrics.classification_report with macro avg F1
    - Includes ALL relationship classes including 'none'
    - Main metric: results['macro avg']['f1-score']

    Reference: external/4D-OR/scene_graph_prediction/scene_graph_helpers/model/
               scene_graph_prediction_model.py lines 224-238
    """

    # Relationship names from 4D-OR (sorted alphabetically + 'none' appended)
    RELATION_NAMES = [
        'Assisting', 'Cementing', 'Cleaning', 'CloseTo', 'Cutting',
        'Drilling', 'Hammering', 'Holding', 'LyingOn', 'Operating',
        'Preparing', 'Sawing', 'Suturing', 'Touching', 'none',
    ]

    # Node class names from 4D-OR
    NODE_NAMES = [
        'Patient', 'anesthesia_equipment', 'human_0', 'human_1', 'human_2',
        'human_3', 'human_4', 'human_5', 'instrument', 'instrument_table',
        'operating_table', 'secondary_table',
    ]

    def __init__(self, num_classes: int = 12, num_relations: int = 15):
        """
        Args:
            num_classes: Number of object classes (12 in 4D-OR)
            num_relations: Number of relation classes including 'none' (15 in 4D-OR)
        """
        self.num_classes = num_classes
        self.num_relations = num_relations
        self.reset()

    def reset(self):
        # Collect all predictions and GT for sklearn classification_report
        # This matches 4D-OR: train_take_rel_preds, train_take_rel_gts
        self.all_rel_preds = []
        self.all_rel_gts = []

        # Also track node predictions for auxiliary metrics
        self.all_node_preds = []
        self.all_node_gts = []

    @torch.no_grad()
    def update(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict,
        matcher: HungarianMatcher,
    ):
        """Update metrics with batch predictions."""
        node_logits = outputs['node_logits']
        edge_logits = outputs['edge_logits']
        mask = batch['mask']

        B = node_logits.shape[0]

        for b in range(B):
            valid_mask = mask[b]
            n_pred = valid_mask.sum().item()

            if n_pred == 0:
                continue

            pred_logits = node_logits[b, :n_pred]
            sam3_cats = batch['sam3_categories'][b]
            gt_classes = batch['gt_classes'][b].to(node_logits.device)
            gt_rel_matrix = batch['gt_rel_matrix'][b].to(node_logits.device)

            if len(gt_classes) == 0:
                continue

            # Hungarian matching
            pred_idx, gt_idx = matcher.match(pred_logits, gt_classes, sam3_cats)

            if len(pred_idx) == 0:
                continue

            # Node predictions
            matched_logits = pred_logits[pred_idx]
            matched_gt = gt_classes[gt_idx]
            pred_classes = matched_logits.argmax(dim=-1)

            self.all_node_preds.extend(pred_classes.cpu().numpy().tolist())
            self.all_node_gts.extend(matched_gt.cpu().numpy().tolist())

            # Relationship predictions
            # 4D-OR evaluates on ALL edge pairs (not just diagonal-excluded)
            K = len(pred_idx)
            if K >= 2:
                matched_edge_logits = edge_logits[b, pred_idx][:, pred_idx]
                matched_gt_rels = gt_rel_matrix[gt_idx][:, gt_idx]

                pred_rels = matched_edge_logits.argmax(dim=-1)

                # Collect all pairs (matching 4D-OR which uses edge_indices)
                for i in range(K):
                    for j in range(K):
                        if i != j:  # Skip self-relations
                            self.all_rel_preds.append(pred_rels[i, j].item())
                            self.all_rel_gts.append(matched_gt_rels[i, j].item())

    def compute(self) -> Dict[str, float]:
        """
        Compute final metrics matching official 4D-OR evaluation.

        Uses sklearn.metrics.classification_report exactly like 4D-OR:
        ```
        results = classification_report(all_rel_gts, all_rel_preds,
                                        labels=list(range(len(self.relationNames))),
                                        target_names=self.relationNames, output_dict=True)
        macro_f1 = results['macro avg']['f1-score']
        ```
        """
        from sklearn.metrics import classification_report

        results = {}

        # === RELATIONSHIP METRICS (THE MAIN METRIC) ===
        if len(self.all_rel_gts) > 0:
            rel_preds = self.all_rel_preds
            rel_gts = self.all_rel_gts

            # Match 4D-OR exactly: use all classes including 'none'
            cls_report = classification_report(
                rel_gts,
                rel_preds,
                labels=list(range(self.num_relations)),
                target_names=self.RELATION_NAMES[:self.num_relations],
                output_dict=True,
                zero_division=0,
            )

            # Main metric: macro avg F1 (exactly what 4D-OR/LABRAD-OR report)
            results['rel_macro_f1'] = cls_report['macro avg']['f1-score']
            results['rel_macro_precision'] = cls_report['macro avg']['precision']
            results['rel_macro_recall'] = cls_report['macro avg']['recall']

            # Per-class P, R, F1 for detailed analysis
            for rel_name in self.RELATION_NAMES[:self.num_relations]:
                if rel_name in cls_report:
                    results[f'rel_{rel_name}_precision'] = cls_report[rel_name]['precision']
                    results[f'rel_{rel_name}_recall'] = cls_report[rel_name]['recall']
                    results[f'rel_{rel_name}_f1'] = cls_report[rel_name]['f1-score']
                    results[f'rel_{rel_name}_support'] = cls_report[rel_name]['support']
        else:
            results['rel_macro_f1'] = 0.0
            results['rel_macro_precision'] = 0.0
            results['rel_macro_recall'] = 0.0

        # === NODE METRICS (Auxiliary) ===
        if len(self.all_node_gts) > 0:
            node_cls_report = classification_report(
                self.all_node_gts,
                self.all_node_preds,
                labels=list(range(self.num_classes)),
                target_names=self.NODE_NAMES[:self.num_classes],
                output_dict=True,
                zero_division=0,
            )
            results['node_macro_f1'] = node_cls_report['macro avg']['f1-score']
            results['node_macro_precision'] = node_cls_report['macro avg']['precision']
            results['node_macro_recall'] = node_cls_report['macro avg']['recall']
            results['node_accuracy'] = node_cls_report['accuracy']

            # Per-class P, R, F1 for detailed analysis
            for node_name in self.NODE_NAMES[:self.num_classes]:
                if node_name in node_cls_report:
                    results[f'node_{node_name}_precision'] = node_cls_report[node_name]['precision']
                    results[f'node_{node_name}_recall'] = node_cls_report[node_name]['recall']
                    results[f'node_{node_name}_f1'] = node_cls_report[node_name]['f1-score']
                    results[f'node_{node_name}_support'] = node_cls_report[node_name]['support']
        else:
            results['node_macro_f1'] = 0.0
            results['node_macro_precision'] = 0.0
            results['node_macro_recall'] = 0.0
            results['node_accuracy'] = 0.0

        # Aliases for training script compatibility
        results['edge_f1'] = results['rel_macro_f1']
        results['node_f1'] = results['node_macro_f1']
        results['scene_graph_f1'] = results['rel_macro_f1']  # Main metric

        return results
