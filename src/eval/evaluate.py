#!/usr/bin/env python3
"""
Evaluate scene graph prediction model on train/validation/test set.

Matches LABRAD-OR evaluation procedure:
- sklearn.metrics.classification_report
- Per-take and aggregate metrics
- Macro F1 as primary metric

Usage:
    python -m src.eval.evaluate --checkpoint checkpoints/sage_or.pt --split test
    python -m src.eval.evaluate --checkpoint checkpoints/sage_or.pt --split validation
    python -m src.eval.evaluate --checkpoint checkpoints/sage_or.pt --split test --per-take
"""

import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import json
import sys
import numpy as np

from sklearn.metrics import classification_report, confusion_matrix

# Project root (src/eval/evaluate.py -> src/eval/ -> src/ -> project_root/)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.graph_transformer import SceneGraphPredictor
from src.models.losses import SceneGraphLoss, HungarianMatcher
from src.dataset.cached_scene_graph_dataset import CachedSceneGraphDataset, collate_fn


# Relationship names from 4D-OR (sorted alphabetically + 'none')
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


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


class DetailedMetrics:
    """
    Collect predictions for detailed evaluation matching LABRAD-OR.

    Tracks per-take predictions for breakdown analysis.
    """

    def __init__(self, num_classes: int = 12, num_relations: int = 15):
        self.num_classes = num_classes
        self.num_relations = num_relations
        self.reset()

    def reset(self):
        # Per-take collections (like LABRAD-OR)
        self.take_rel_preds = defaultdict(list)
        self.take_rel_gts = defaultdict(list)
        self.take_node_preds = defaultdict(list)
        self.take_node_gts = defaultdict(list)

        # Losses
        self.total_loss = 0.0
        self.total_node_loss = 0.0
        self.total_edge_loss = 0.0
        self.total_proxy_loss = 0.0
        self.n_batches = 0

    @torch.no_grad()
    def update(
        self,
        outputs: dict,
        batch: dict,
        matcher: HungarianMatcher,
        losses: dict = None,
    ):
        """Update metrics with batch predictions."""
        node_logits = outputs['node_logits']
        edge_logits = outputs['edge_logits']
        mask = batch['mask']
        take_ids = batch.get('take_ids', None)

        B = node_logits.shape[0]

        # Track losses
        if losses is not None:
            self.total_loss += losses['total'].item()
            self.total_node_loss += losses['node'].item() if isinstance(losses['node'], torch.Tensor) else losses['node']
            self.total_edge_loss += losses['edge'].item() if isinstance(losses['edge'], torch.Tensor) else losses['edge']
            if 'proxy' in losses:
                self.total_proxy_loss += losses['proxy'].item() if isinstance(losses['proxy'], torch.Tensor) else losses['proxy']
            self.n_batches += 1

        for b in range(B):
            valid_mask = mask[b]
            n_pred = valid_mask.sum().item()

            if n_pred == 0:
                continue

            pred_logits = node_logits[b, :n_pred]
            sam3_cats = batch['sam3_categories'][b]
            gt_classes = batch['gt_classes'][b].to(node_logits.device)
            gt_rel_matrix = batch['gt_rel_matrix'][b].to(node_logits.device)

            # Get take_id for per-take tracking
            take_id = take_ids[b] if take_ids is not None else 0

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

            self.take_node_preds[take_id].extend(pred_classes.cpu().numpy().tolist())
            self.take_node_gts[take_id].extend(matched_gt.cpu().numpy().tolist())

            # Relationship predictions
            K = len(pred_idx)
            if K >= 2:
                matched_edge_logits = edge_logits[b, pred_idx][:, pred_idx]
                matched_gt_rels = gt_rel_matrix[gt_idx][:, gt_idx]

                pred_rels = matched_edge_logits.argmax(dim=-1)

                for i in range(K):
                    for j in range(K):
                        if i != j:
                            self.take_rel_preds[take_id].append(pred_rels[i, j].item())
                            self.take_rel_gts[take_id].append(matched_gt_rels[i, j].item())

    def compute(self, per_take: bool = False) -> dict:
        """
        Compute metrics matching LABRAD-OR evaluation.

        Args:
            per_take: If True, print per-take breakdown

        Returns:
            Dictionary with all metrics
        """
        results = {}

        # Aggregate all predictions
        all_rel_preds = []
        all_rel_gts = []
        all_node_preds = []
        all_node_gts = []

        for take_id in sorted(self.take_rel_preds.keys()):
            all_rel_preds.extend(self.take_rel_preds[take_id])
            all_rel_gts.extend(self.take_rel_gts[take_id])
            all_node_preds.extend(self.take_node_preds[take_id])
            all_node_gts.extend(self.take_node_gts[take_id])

        # === LOSSES ===
        if self.n_batches > 0:
            results['loss'] = self.total_loss / self.n_batches
            results['node_loss'] = self.total_node_loss / self.n_batches
            results['edge_loss'] = self.total_edge_loss / self.n_batches
            if self.total_proxy_loss > 0:
                results['proxy_loss'] = self.total_proxy_loss / self.n_batches

        # === RELATIONSHIP METRICS ===
        if len(all_rel_gts) > 0:
            # Full classification report (like LABRAD-OR)
            rel_report = classification_report(
                all_rel_gts,
                all_rel_preds,
                labels=list(range(self.num_relations)),
                target_names=RELATION_NAMES[:self.num_relations],
                output_dict=True,
                zero_division=0,
            )

            # Main metrics
            results['rel_macro_f1'] = rel_report['macro avg']['f1-score']
            results['rel_macro_precision'] = rel_report['macro avg']['precision']
            results['rel_macro_recall'] = rel_report['macro avg']['recall']
            results['rel_weighted_f1'] = rel_report['weighted avg']['f1-score']
            results['rel_weighted_precision'] = rel_report['weighted avg']['precision']
            results['rel_weighted_recall'] = rel_report['weighted avg']['recall']

            # Per-class metrics
            for rel_name in RELATION_NAMES[:self.num_relations]:
                if rel_name in rel_report:
                    results[f'rel_{rel_name}_precision'] = rel_report[rel_name]['precision']
                    results[f'rel_{rel_name}_recall'] = rel_report[rel_name]['recall']
                    results[f'rel_{rel_name}_f1'] = rel_report[rel_name]['f1-score']
                    results[f'rel_{rel_name}_support'] = rel_report[rel_name]['support']

            # Store full report for printing
            results['_rel_report'] = rel_report
            results['_rel_report_str'] = classification_report(
                all_rel_gts,
                all_rel_preds,
                labels=list(range(self.num_relations)),
                target_names=RELATION_NAMES[:self.num_relations],
                zero_division=0,
            )

        # === NODE METRICS ===
        if len(all_node_gts) > 0:
            node_report = classification_report(
                all_node_gts,
                all_node_preds,
                labels=list(range(self.num_classes)),
                target_names=NODE_NAMES[:self.num_classes],
                output_dict=True,
                zero_division=0,
            )

            results['node_macro_f1'] = node_report['macro avg']['f1-score']
            results['node_macro_precision'] = node_report['macro avg']['precision']
            results['node_macro_recall'] = node_report['macro avg']['recall']
            results['node_weighted_f1'] = node_report['weighted avg']['f1-score']
            results['node_accuracy'] = node_report.get('accuracy', 0.0)

            # Per-class metrics
            for node_name in NODE_NAMES[:self.num_classes]:
                if node_name in node_report:
                    results[f'node_{node_name}_precision'] = node_report[node_name]['precision']
                    results[f'node_{node_name}_recall'] = node_report[node_name]['recall']
                    results[f'node_{node_name}_f1'] = node_report[node_name]['f1-score']
                    results[f'node_{node_name}_support'] = node_report[node_name]['support']

            results['_node_report_str'] = classification_report(
                all_node_gts,
                all_node_preds,
                labels=list(range(self.num_classes)),
                target_names=NODE_NAMES[:self.num_classes],
                zero_division=0,
            )

        # === PER-TAKE METRICS ===
        if per_take:
            results['_per_take'] = {}
            for take_id in sorted(self.take_rel_preds.keys()):
                take_rel_preds = self.take_rel_preds[take_id]
                take_rel_gts = self.take_rel_gts[take_id]

                if len(take_rel_gts) > 0:
                    take_report = classification_report(
                        take_rel_gts,
                        take_rel_preds,
                        labels=list(range(self.num_relations)),
                        target_names=RELATION_NAMES[:self.num_relations],
                        output_dict=True,
                        zero_division=0,
                    )
                    results['_per_take'][take_id] = {
                        'macro_f1': take_report['macro avg']['f1-score'],
                        'macro_precision': take_report['macro avg']['precision'],
                        'macro_recall': take_report['macro avg']['recall'],
                        'n_samples': len(take_rel_gts),
                        'report_str': classification_report(
                            take_rel_gts,
                            take_rel_preds,
                            labels=list(range(self.num_relations)),
                            target_names=RELATION_NAMES[:self.num_relations],
                            zero_division=0,
                        ),
                    }

        # Aliases for compatibility
        results['edge_f1'] = results.get('rel_macro_f1', 0.0)
        results['node_f1'] = results.get('node_macro_f1', 0.0)
        results['scene_graph_f1'] = results.get('rel_macro_f1', 0.0)

        return results


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: SceneGraphLoss,
    metrics: DetailedMetrics,
    device: torch.device,
    compute_loss: bool = True,
) -> dict:
    """Run evaluation on dataloader."""
    model.eval()
    metrics.reset()

    pbar = tqdm(dataloader, desc='Evaluating')
    for batch in pbar:
        visual = batch['visual_features'].to(device)
        geometric = batch['geometric_features'].to(device)
        category = batch['category_hints'].to(device)
        centroids = batch['centroids'].to(device)
        mask = batch['mask'].to(device)

        frame_embedding = batch.get('frame_embedding', None)
        if frame_embedding is not None:
            frame_embedding = frame_embedding.to(device)
        outputs = model(visual, geometric, category, centroids, mask, frame_embedding=frame_embedding)

        losses = None
        if compute_loss:
            losses = criterion(outputs, batch)

        metrics.update(outputs, batch, criterion.matcher, losses)

    return metrics.compute(per_take=True)


def print_results(results: dict, split: str):
    """Print evaluation results in a formatted way matching training output."""
    print("\n" + "=" * 80)
    print(f"EVALUATION RESULTS - {split.upper()}")
    print("=" * 80)

    # Summary line (like training)
    loss_str = f"Loss: {results.get('loss', 0):.4f}" if 'loss' in results else ""
    node_loss_str = f"Node: {results.get('node_loss', 0):.4f}" if 'node_loss' in results else ""
    edge_loss_str = f"Edge: {results.get('edge_loss', 0):.4f}" if 'edge_loss' in results else ""
    proxy_str = f", Proxy: {results.get('proxy_loss', 0):.4f}" if 'proxy_loss' in results else ""

    if loss_str:
        print(f"  {split.capitalize()} - {loss_str}, {node_loss_str}, {edge_loss_str}{proxy_str}")

    print(f"  {split.capitalize()} - Node F1: {results.get('node_macro_f1', 0):.4f}, Edge F1: {results.get('rel_macro_f1', 0):.4f}, SG F1: {results.get('rel_macro_f1', 0):.4f}")

    # Relationship Metrics (matching training format)
    print("  Relationship Metrics:")
    print(f"    {'Relation':<14} {'P':>6} {'R':>6} {'F1':>6}  {'Support':>7}")
    print("    " + "-" * 44)
    for rel_name in RELATION_NAMES:
        p = results.get(f'rel_{rel_name}_precision', 0.0)
        r = results.get(f'rel_{rel_name}_recall', 0.0)
        f1 = results.get(f'rel_{rel_name}_f1', 0.0)
        sup = int(results.get(f'rel_{rel_name}_support', 0))
        print(f"    {rel_name:<14} {p:>6.2f} {r:>6.2f} {f1:>6.2f}  {sup:>7}")

    # Macro averages for relationships
    print("    " + "-" * 44)
    print(f"    {'macro avg':<14} {results.get('rel_macro_precision', 0):>6.2f} {results.get('rel_macro_recall', 0):>6.2f} {results.get('rel_macro_f1', 0):>6.2f}")
    print(f"    {'weighted avg':<14} {results.get('rel_weighted_precision', 0):>6.2f} {results.get('rel_weighted_recall', 0):>6.2f} {results.get('rel_weighted_f1', 0):>6.2f}")

    # Node Metrics (matching training format)
    print("  Node Metrics:")
    print(f"    {'Node':<26} {'P':>6} {'R':>6} {'F1':>6}  {'Support':>7}")
    print("    " + "-" * 56)
    for node_name in NODE_NAMES:
        p = results.get(f'node_{node_name}_precision', 0.0)
        r = results.get(f'node_{node_name}_recall', 0.0)
        f1 = results.get(f'node_{node_name}_f1', 0.0)
        sup = int(results.get(f'node_{node_name}_support', 0))
        print(f"    {node_name:<26} {p:>6.2f} {r:>6.2f} {f1:>6.2f}  {sup:>7}")

    # Macro averages for nodes
    print("    " + "-" * 56)
    print(f"    {'macro avg':<26} {results.get('node_macro_precision', 0):>6.2f} {results.get('node_macro_recall', 0):>6.2f} {results.get('node_macro_f1', 0):>6.2f}")
    print(f"    {'accuracy':<26} {'':<6} {'':<6} {results.get('node_accuracy', 0):>6.2f}")

    # Per-take breakdown
    if '_per_take' in results and results['_per_take']:
        print("\n  Per-Take Breakdown:")
        print(f"    {'Take':<8} {'Macro P':>8} {'Macro R':>8} {'Macro F1':>8} {'Samples':>8}")
        print("    " + "-" * 44)
        for take_id, take_results in sorted(results['_per_take'].items()):
            print(f"    {take_id:<8} {take_results['macro_precision']:>8.4f} {take_results['macro_recall']:>8.4f} {take_results['macro_f1']:>8.4f} {take_results['n_samples']:>8}")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Evaluate scene graph model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--config', type=str, default=None, help='Path to config (auto-detected if not provided)')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'validation', 'test'])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch-size', type=int, default=None, help='Override batch size')
    parser.add_argument('--per-take', action='store_true', help='Show per-take breakdown')
    parser.add_argument('--save-json', type=str, default=None, help='Save results to JSON file')
    args = parser.parse_args()

    # Auto-detect config from checkpoint path if not provided
    if args.config is None:
        checkpoint_path = Path(args.checkpoint)
        # Extract experiment name from checkpoint directory
        # e.g., "checkpoints/base_model_42/best.pt" -> "base_model_42"
        experiment_name = checkpoint_path.parent.name

        # Try to find matching config
        config_candidates = [
            f"configs/{experiment_name}.yaml",
            f"configs/ablations/{experiment_name}.yaml",
        ]

        config_path = None
        for candidate in config_candidates:
            if Path(candidate).exists():
                config_path = candidate
                break

        if config_path is None:
            # Fall back to base config
            config_path = "configs/real-or.yaml"
            print(f"Warning: Could not find config for '{experiment_name}', using base config")

        args.config = config_path

    # Load config
    config = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print(f"Device: {device}")
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Split: {args.split}")

    # Check checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # Load dataset
    print(f"\nLoading {args.split} dataset...")
    dataset = CachedSceneGraphDataset(config, split=args.split)
    print(f"  {len(dataset)} frames")

    batch_size = args.batch_size or config['training']['batch_size']
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
    )

    # Load checkpoint
    print(f"\nLoading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # Build model
    print("\nBuilding model...")
    model = SceneGraphPredictor(config).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Load state dict
    model.load_state_dict(state_dict)

    # Print checkpoint info
    if 'epoch' in checkpoint:
        epoch = checkpoint['epoch']
        if 'best_metric' in checkpoint:
            print(f"  Loaded from epoch {epoch}, best_metric={checkpoint['best_metric']}")
        elif 'best_f1' in checkpoint:
            print(f"  Loaded from epoch {epoch}, best_f1={checkpoint['best_f1']}")
        else:
            print(f"  Loaded from epoch {epoch}")
    else:
        print("  Loaded checkpoint")

    # Loss function (for computing losses during eval)
    # Note: Disable proxy task for evaluation since we don't compute it
    criterion = SceneGraphLoss(
        num_classes=config['model']['node_head']['num_classes'],
        num_relations=config['model']['edge_head']['num_relations'],
        use_proxy_task=False,  # Disabled for evaluation
    )

    # Metrics
    metrics = DetailedMetrics(
        num_classes=config['model']['node_head']['num_classes'],
        num_relations=config['model']['edge_head']['num_relations'],
    )

    # Evaluate
    print(f"\nEvaluating on {args.split}...")
    results = evaluate(model, dataloader, criterion, metrics, device, compute_loss=True)

    # Print results
    print_results(results, args.split)

    # Save to JSON if requested
    if args.save_json:
        # Filter out non-serializable items
        json_results = {k: v for k, v in results.items() if not k.startswith('_')}
        with open(args.save_json, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"\nResults saved to {args.save_json}")

    return results


if __name__ == '__main__':
    main()
