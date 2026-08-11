#!/usr/bin/env python3
"""
Train SAGE-OR scene graph prediction model.

Usage:
    python -m src.training.train_scene_graph --config configs/real-or.yaml
    python -m src.training.train_scene_graph --config configs/real-or.yaml --resume checkpoints/latest.pt
    python -m src.training.train_scene_graph --config configs/real-or.yaml --eval_every 1

Arguments:
    --config        Path to config YAML file
    --resume        Resume training from checkpoint
    --device        Device to use (default: cuda)
    --eval_every    Evaluate on test set every N epochs (default: 10)

Pre-computed feature cache is recommended (see src/dataset/feature_cache_generator.py).
Falls back to on-the-fly extraction if cache not found (slow).

Note: For ease of use, this script evaluates on the test set.
"""

import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from pathlib import Path
from tqdm import tqdm
import json
import time
import sys
import random
import numpy as np

# Project root is three levels up from src/training/train_scene_graph.py
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.graph_transformer import SceneGraphPredictor
from src.models.losses import SceneGraphLoss, SceneGraphMetrics, compute_class_weights, compute_proxy_pos_weight
from src.dataset.scene_graph_dataset import SceneGraphDataset, collate_fn


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def worker_init_fn(worker_id):
    """Initialize each DataLoader worker with a unique but deterministic seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_seed(seed: int):
    """
    Set random seed for reproducibility across:
    - Python random
    - NumPy
    - PyTorch CPU/GPU
    - cuDNN (makes it deterministic but slower)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make cuDNN deterministic (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Set random seed to {seed} (deterministic mode enabled)")


def apply_augmentation(
    visual_features: torch.Tensor,
    geometric_features: torch.Tensor,
    mask: torch.Tensor,
    config: dict,
) -> tuple:
    """
    Apply data augmentation to input features during training.

    Args:
        visual_features: [B, N, D_vis] visual features
        geometric_features: [B, N, D_geo] geometric features
        mask: [B, N] validity mask
        config: augmentation config dict

    Returns:
        Augmented visual features, geometric features, and mask
    """
    aug_cfg = config.get('augmentation', {})
    if not aug_cfg.get('enabled', False):
        return visual_features, geometric_features, mask

    device = visual_features.device
    B, N, D_vis = visual_features.shape
    D_geo = geometric_features.shape[2]

    # Clone to avoid modifying original
    visual_aug = visual_features.clone()
    geometric_aug = geometric_features.clone()
    mask_aug = mask.clone()  # Clone mask for potential modification

    # === Feature Dropout ===
    dropout_cfg = aug_cfg.get('feature_dropout', {})
    if dropout_cfg.get('enabled', False):
        # Visual feature dropout
        visual_drop_prob = dropout_cfg.get('visual_drop_prob', 0.1)
        visual_drop_mask = (torch.rand(B, N, D_vis, device=device) > visual_drop_prob).float()
        visual_aug = visual_aug * visual_drop_mask

        # Geometric feature dropout
        geometric_drop_prob = dropout_cfg.get('geometric_drop_prob', 0.1)
        geometric_drop_mask = (torch.rand(B, N, D_geo, device=device) > geometric_drop_prob).float()
        geometric_aug = geometric_aug * geometric_drop_mask

    # === Node Dropout (OPTIONAL) ===
    node_dropout_cfg = aug_cfg.get('node_dropout', {})
    if node_dropout_cfg.get('enabled', False):
        drop_prob = node_dropout_cfg.get('drop_prob', 0.1)
        # Create dropout mask [B, N]
        node_drop_mask = (torch.rand(B, N, device=device) > drop_prob).float()
        # Combine with validity mask
        combined_mask = mask_aug.float() * node_drop_mask
        # Zero out dropped nodes in features
        visual_aug = visual_aug * combined_mask.unsqueeze(-1)
        geometric_aug = geometric_aug * combined_mask.unsqueeze(-1)
        # Update mask to reflect dropped nodes
        mask_aug = combined_mask.bool()

    # === Subgraph Sampling ===
    subgraph_cfg = aug_cfg.get('subgraph_sampling', {})
    if subgraph_cfg.get('enabled', False):
        min_keep_ratio = subgraph_cfg.get('min_keep_ratio', 0.5)  # Keep at least 50% of nodes
        max_keep_ratio = subgraph_cfg.get('max_keep_ratio', 0.9)  # Keep at most 90% of nodes

        for b in range(B):
            # Get valid nodes for this sample
            valid_mask = mask_aug[b]  # [N]
            n_valid = valid_mask.sum().item()

            if n_valid > 1:  # Need at least 2 nodes for meaningful sampling
                # Randomly decide how many to keep
                keep_ratio = torch.empty(1).uniform_(min_keep_ratio, max_keep_ratio).item()
                num_keep = max(1, int(n_valid * keep_ratio))

                # Get indices of valid nodes
                valid_indices = torch.where(valid_mask)[0]

                # Randomly select which nodes to drop
                perm = torch.randperm(n_valid, device=device)
                drop_idx = perm[num_keep:]  # Indices within valid nodes to drop

                # Get actual node indices to drop
                drop_node_indices = valid_indices[drop_idx]

                # Update mask - mark dropped nodes as invalid
                mask_aug[b, drop_node_indices] = False

                # Zero out features for dropped nodes (cleaner, though mask handles it)
                visual_aug[b, drop_node_indices] = 0
                geometric_aug[b, drop_node_indices] = 0

    return visual_aug, geometric_aug, mask_aug


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: SceneGraphLoss,
    device: torch.device,
    epoch: int,
    config: dict,
) -> dict:
    model.train()
    total_loss = 0
    total_node_loss = 0
    total_edge_loss = 0
    total_proxy_loss = 0

    # Diagnostic accumulators
    total_grad_norm = 0
    total_node_grad_norm = 0
    total_edge_grad_norm = 0
    total_proxy_grad_norm = 0
    total_visual_edge_grad_norm = 0
    total_visual_feat_norm = 0
    total_geometric_feat_norm = 0

    n_batches = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    for batch in pbar:
        # Move to device
        visual = batch['visual_features'].to(device)
        geometric = batch['geometric_features'].to(device)
        category = batch['category_hints'].to(device)
        centroids = batch['centroids'].to(device)
        mask = batch['mask'].to(device)

        # Apply data augmentation (training only)
        visual, geometric, mask = apply_augmentation(visual, geometric, mask, config)

        # Feature statistics (after augmentation)
        with torch.no_grad():
            total_visual_feat_norm += visual[mask.bool()].norm(dim=-1).mean().item()
            total_geometric_feat_norm += geometric[mask.bool()].norm(dim=-1).mean().item()

        # Frame embedding (global scene context)
        frame_embedding = batch.get('frame_embedding', None)
        if frame_embedding is not None:
            frame_embedding = frame_embedding.to(device)

        # Forward
        outputs = model(visual, geometric, category, centroids, mask, frame_embedding)

        # Loss
        losses = criterion(outputs, batch)

        # Backward
        optimizer.zero_grad()
        losses['total'].backward()

        # Compute gradient norms BEFORE clipping
        with torch.no_grad():
            # Total gradient norm
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            total_grad_norm += total_norm

            # Node head gradient norm
            node_grad_norm = 0
            for p in model.node_head.parameters():
                if p.grad is not None:
                    node_grad_norm += p.grad.data.norm(2).item() ** 2
            node_grad_norm = node_grad_norm ** 0.5
            total_node_grad_norm += node_grad_norm

            # Edge head gradient norm
            edge_grad_norm = 0
            for p in model.edge_head.parameters():
                if p.grad is not None:
                    edge_grad_norm += p.grad.data.norm(2).item() ** 2
            edge_grad_norm = edge_grad_norm ** 0.5
            total_edge_grad_norm += edge_grad_norm

            # Proxy head gradient norm (if exists)
            if hasattr(model, 'proxy_head') and model.proxy_head is not None:
                proxy_grad_norm = 0
                for p in model.proxy_head.parameters():
                    if p.grad is not None:
                        proxy_grad_norm += p.grad.data.norm(2).item() ** 2
                proxy_grad_norm = proxy_grad_norm ** 0.5
                total_proxy_grad_norm += proxy_grad_norm

            # Visual edge projection gradient norm (if exists)
            if hasattr(model, 'visual_edge_proj') and model.visual_edge_proj is not None:
                visual_edge_grad_norm = 0
                for p in model.visual_edge_proj.parameters():
                    if p.grad is not None:
                        visual_edge_grad_norm += p.grad.data.norm(2).item() ** 2
                visual_edge_grad_norm = visual_edge_grad_norm ** 0.5
                total_visual_edge_grad_norm += visual_edge_grad_norm

        # Gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Track
        total_loss += losses['total'].item()
        total_node_loss += losses['node'].item() if isinstance(losses['node'], torch.Tensor) else losses['node']
        total_edge_loss += losses['edge'].item() if isinstance(losses['edge'], torch.Tensor) else losses['edge']
        if 'proxy' in losses:
            total_proxy_loss += losses['proxy'].item() if isinstance(losses['proxy'], torch.Tensor) else losses['proxy']
        n_batches += 1

        pbar.set_postfix({
            'loss': f"{losses['total'].item():.4f}",
            'node': f"{losses['node'].item() if isinstance(losses['node'], torch.Tensor) else losses['node']:.4f}",
        })

    result = {
        'loss': total_loss / n_batches,
        'node_loss': total_node_loss / n_batches,
        'edge_loss': total_edge_loss / n_batches,
        # Diagnostics
        'grad_norm': total_grad_norm / n_batches,
        'node_grad_norm': total_node_grad_norm / n_batches,
        'edge_grad_norm': total_edge_grad_norm / n_batches,
        'visual_feat_norm': total_visual_feat_norm / n_batches,
        'geometric_feat_norm': total_geometric_feat_norm / n_batches,
    }
    if total_proxy_loss > 0:
        result['proxy_loss'] = total_proxy_loss / n_batches
        result['proxy_grad_norm'] = total_proxy_grad_norm / n_batches
    if total_visual_edge_grad_norm > 0:
        result['visual_edge_grad_norm'] = total_visual_edge_grad_norm / n_batches

    return result


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: SceneGraphLoss,
    metrics: SceneGraphMetrics,
    device: torch.device,
    epoch: int,
) -> dict:
    model.eval()
    metrics.reset()

    total_loss = 0
    n_batches = 0

    # FPS tracking
    total_frames = 0
    total_inference_time = 0.0

    # GPU memory tracking (reset peak stats before validation)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Test]')
    for batch in pbar:
        visual = batch['visual_features'].to(device)
        geometric = batch['geometric_features'].to(device)
        category = batch['category_hints'].to(device)
        centroids = batch['centroids'].to(device)
        mask = batch['mask'].to(device)

        # Frame embedding (global scene context)
        frame_embedding = batch.get('frame_embedding', None)
        if frame_embedding is not None:
            frame_embedding = frame_embedding.to(device)

        # Forward (timed for FPS calculation)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        inference_start = time.time()

        outputs = model(visual, geometric, category, centroids, mask, frame_embedding)

        if device.type == 'cuda':
            torch.cuda.synchronize()
        inference_time = time.time() - inference_start

        # Track FPS
        batch_size = visual.shape[0]
        total_frames += batch_size
        total_inference_time += inference_time

        losses = criterion(outputs, batch)
        total_loss += losses['total'].item()
        n_batches += 1

        # Update metrics
        metrics.update(outputs, batch, criterion.matcher)

    # Compute final metrics
    metric_results = metrics.compute()
    metric_results['loss'] = total_loss / n_batches

    # Add FPS metric
    if total_inference_time > 0:
        metric_results['fps'] = total_frames / total_inference_time
    else:
        metric_results['fps'] = 0.0

    # Add GPU memory metric (zero overhead - just reads a counter)
    # Use memory_reserved() to match nvidia-smi (includes model + cache + activations)
    if device.type == 'cuda':
        total_memory_bytes = torch.cuda.memory_reserved(device)
        metric_results['peak_memory_mb'] = total_memory_bytes / (1024 ** 2)
    else:
        metric_results['peak_memory_mb'] = 0.0

    return metric_results


def main():
    parser = argparse.ArgumentParser(description='Train scene graph model')
    parser.add_argument('--config', type=str, default='configs/real-or.yaml')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--eval_every', type=int, default=10,
                        help='Evaluate on test set every N epochs (default: 10)')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Set random seed for reproducibility
    seed = config['training'].get('seed', 42)
    set_seed(seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}")
    print(f"Config: {args.config}")

    # Paths (project_root is defined at module level)
    experiment_name = config['logging'].get('experiment_name', 'default')
    output_dir = project_root / config['logging']['output_dir'] / experiment_name
    checkpoint_dir = project_root / config['logging']['checkpoint_dir'] / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"Experiment: {experiment_name}")
    print(f"Outputs: {output_dir}")
    print(f"Checkpoints: {checkpoint_dir}")

    # Check for feature cache
    feature_cache_dir = project_root / config['data'].get('feature_cache_dir', 'caches/features')
    use_cached = feature_cache_dir.exists() and any(feature_cache_dir.rglob('*.npz'))

    if use_cached:
        print(f"Using cached features from {feature_cache_dir}")
        from src.dataset.cached_scene_graph_dataset import CachedSceneGraphDataset, collate_fn as cached_collate_fn

        train_dataset = CachedSceneGraphDataset(config, split='train')
        test_dataset = CachedSceneGraphDataset(config, split='test')
        collate = cached_collate_fn
        num_workers = config['training'].get('num_workers', 4)
    else:
        print("No feature cache found, using on-the-fly extraction (slow)")
        print("To generate cache: python -m src.dataset.feature_cache_generator --config configs/real-or.yaml")
        from src.features.dinov3 import DINOv3Extractor

        print("Loading DINOv3 extractor...")
        extractor = DINOv3Extractor(device=str(device))

        train_dataset = SceneGraphDataset(config, split='train', extractor=extractor)
        test_dataset = SceneGraphDataset(config, split='test', extractor=extractor)
        collate = collate_fn
        num_workers = 0  # Must be 0 for CUDA extractor

    print(f"Train: {len(train_dataset)} frames")
    print(f"Test: {len(test_dataset)} frames")
    print(f"DataLoader num_workers: {num_workers}")

    # Dataloaders with deterministic shuffling
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate,
        generator=g,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
    )

    # Model
    print("Building model...")
    model = SceneGraphPredictor(config).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Loss and metrics
    use_balanced = config['training'].get('balanced_weights', False)
    proxy_config = config['training'].get('proxy_task', {})
    use_proxy = proxy_config.get('enabled', False)
    proxy_weight = proxy_config.get('weight', 1.0)

    # Compute balanced class weights if enabled
    rel_class_weights = None
    if use_balanced:
        print("Computing balanced class weights from training data...")

        _, rel_class_weights = compute_class_weights(
            train_dataset,
            num_classes=config['model']['node_head']['num_classes'],
            num_relations=config['model']['edge_head']['num_relations'],
            no_relation_weight=config['training']['loss_weights'].get('no_relation', 0.0001),
        )
        print("Relation weights per class:")
        rel_names = ['Assisting', 'Cementing', 'Cleaning', 'CloseTo', 'Cutting',
                     'Drilling', 'Hammering', 'Holding', 'LyingOn', 'Operating',
                     'Preparing', 'Sawing', 'Suturing', 'Touching', 'none']
        for i, name in enumerate(rel_names):
            print(f"  {name}: {rel_class_weights[i]:.6f}")

    criterion = SceneGraphLoss(
        num_classes=config['model']['node_head']['num_classes'],
        num_relations=config['model']['edge_head']['num_relations'],
        node_weight=config['training']['loss_weights']['node'],
        edge_weight=config['training']['loss_weights']['edge'],
        no_relation_weight=config['training']['loss_weights'].get('no_relation', 0.0001),
        rel_class_weights=rel_class_weights,
        use_proxy_task=use_proxy,
        proxy_weight=proxy_weight,
    )

    # Compute proxy pos_weight if proxy task is enabled
    if use_proxy:
        proxy_pos_weight = compute_proxy_pos_weight(
            train_dataset,
            num_relations=config['model']['edge_head']['num_relations'],
        )
        criterion.set_proxy_pos_weight(proxy_pos_weight)

    metrics = SceneGraphMetrics(
        num_classes=config['model']['node_head']['num_classes'],
        num_relations=config['model']['edge_head']['num_relations'],
    )

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=float(config['training']['learning_rate']),
        weight_decay=float(config['training']['weight_decay']),
    )

    # Scheduler: Linear warmup + Cosine decay (NO restarts)
    warmup_epochs = config['training']['scheduler'].get('warmup_epochs', 10)
    total_epochs = config['training']['num_epochs']
    min_lr = float(config['training']['scheduler']['min_lr'])

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_epochs,
        eta_min=min_lr,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )

    # Resume from checkpoint
    start_epoch = 0
    best_f1 = 0.0

    if args.resume:
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        best_f1 = checkpoint.get('best_f1', 0.0)

    # Early stopping setup
    early_stopping_patience = config['training']['early_stopping']['patience']
    early_stopping_metric = config['training']['early_stopping']['metric']
    epochs_without_improvement = 0

    # Training loop
    print("\nStarting training...")
    print(f"Early stopping: patience={early_stopping_patience}, metric={early_stopping_metric}")
    history = {'train': [], 'test': []}

    for epoch in range(start_epoch, config['training']['num_epochs']):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, epoch, config)
        history['train'].append(train_metrics)

        scheduler.step()

        # Print training metrics
        print(f"\nEpoch {epoch}:")
        proxy_str = f", Proxy: {train_metrics['proxy_loss']:.4f}" if 'proxy_loss' in train_metrics else ""
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, Node: {train_metrics['node_loss']:.4f}, Edge: {train_metrics['edge_loss']:.4f}{proxy_str}")

        # Evaluate on test set every N epochs
        test_metrics = None
        if epoch % args.eval_every == 0 or epoch == config['training']['num_epochs'] - 1:
            test_metrics = validate(model, test_loader, criterion, metrics, device, epoch)
            history['test'].append({'epoch': epoch, **test_metrics})
            fps_str = f", FPS: {test_metrics['fps']:.1f}" if 'fps' in test_metrics else ""
            mem_str = f", Mem: {test_metrics['peak_memory_mb']:.0f}MB" if 'peak_memory_mb' in test_metrics else ""
            print(f"  Test  - Loss: {test_metrics['loss']:.4f}, Node F1: {test_metrics['node_f1']:.4f}, Edge F1: {test_metrics['edge_f1']:.4f}, SG F1: {test_metrics['scene_graph_f1']:.4f}{fps_str}{mem_str}")

        # Print per-relationship metrics (P, R, F1) - only when test metrics available
        if test_metrics is not None:
            rel_names = [
                'Assisting', 'Cementing', 'Cleaning', 'CloseTo', 'Cutting',
                'Drilling', 'Hammering', 'Holding', 'LyingOn', 'Operating',
                'Preparing', 'Sawing', 'Suturing', 'Touching', 'none',
            ]
            print("  Relationship Metrics:")
            print(f"    {'Relation':<12} {'P':>6} {'R':>6} {'F1':>6} {'Support':>8}")
            print("    " + "-" * 40)
            for name in rel_names:
                p = test_metrics.get(f'rel_{name}_precision', 0.0)
                r = test_metrics.get(f'rel_{name}_recall', 0.0)
                f1 = test_metrics.get(f'rel_{name}_f1', 0.0)
                sup = test_metrics.get(f'rel_{name}_support', 0)
                print(f"    {name:<12} {p:>6.2f} {r:>6.2f} {f1:>6.2f} {sup:>8.0f}")

            # Print per-node metrics (P, R, F1)
            node_names = [
                'Patient', 'anesthesia_equipment', 'human_0', 'human_1', 'human_2',
                'human_3', 'human_4', 'human_5', 'instrument', 'instrument_table',
                'operating_table', 'secondary_table',
            ]
            print("  Node Metrics:")
            print(f"    {'Node':<20} {'P':>6} {'R':>6} {'F1':>6} {'Support':>8}")
            print("    " + "-" * 48)
            for name in node_names:
                p = test_metrics.get(f'node_{name}_precision', 0.0)
                r = test_metrics.get(f'node_{name}_recall', 0.0)
                f1 = test_metrics.get(f'node_{name}_f1', 0.0)
                sup = test_metrics.get(f'node_{name}_support', 0)
                print(f"    {name:<20} {p:>6.2f} {r:>6.2f} {f1:>6.2f} {sup:>8.0f}")

        # Save checkpoint (only update best when we have test metrics)
        is_best = False
        if test_metrics is not None:
            current_metric = test_metrics[early_stopping_metric]
            is_best = current_metric > best_f1
            if is_best:
                best_f1 = current_metric
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

        checkpoint = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_f1': best_f1,
            'config': config,
        }

        torch.save(checkpoint, checkpoint_dir / 'latest.pt')
        if is_best:
            torch.save(checkpoint, checkpoint_dir / 'best.pt')
            print(f"  New best! SG F1: {best_f1:.4f}")

        if (epoch + 1) % config['logging']['save_every'] == 0:
            torch.save(checkpoint, checkpoint_dir / f'epoch_{epoch:03d}.pt')

        # Save history
        with open(output_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

        # Early stopping check
        if epochs_without_improvement >= early_stopping_patience:
            print(f"\nEarly stopping triggered! No improvement for {early_stopping_patience} epochs.")
            break

    print("\nTraining complete!")
    print(f"Total epochs: {epoch + 1}")
    print(f"Best F1: {best_f1:.4f}")


if __name__ == '__main__':
    main()
