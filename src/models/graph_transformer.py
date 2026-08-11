"""
Graph Transformer for Scene Graph Prediction.

Takes fused instances with visual features and 3D positions,
predicts object classes and pairwise relationships.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List


class PositionalEncoding3D(nn.Module):
    """Encode 3D positions into feature vectors."""

    def __init__(self, hidden_dim: int, max_freq: int = 10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_freq = max_freq

        # Learnable projection from sinusoidal features
        self.proj = nn.Linear(3 * 2 * max_freq, hidden_dim)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions: [B, N, 3] 3D coordinates

        Returns:
            encoding: [B, N, hidden_dim]
        """
        B, N, _ = positions.shape

        # Sinusoidal encoding
        freqs = torch.arange(self.max_freq, device=positions.device, dtype=positions.dtype)
        freqs = (2.0 ** freqs) * math.pi

        # [B, N, 3, max_freq]
        pos_expanded = positions.unsqueeze(-1) * freqs
        # [B, N, 3 * 2 * max_freq]
        encoding = torch.cat([pos_expanded.sin(), pos_expanded.cos()], dim=-1)
        encoding = encoding.view(B, N, -1)

        return self.proj(encoding)


class EdgeFeatureEncoder(nn.Module):
    """Encode pairwise edge features (relative position, distance)."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        # Input: relative position (3) + distance (1) + relative bbox (6)
        self.proj = nn.Sequential(
            nn.Linear(10, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        centroids: torch.Tensor,
        bbox_sizes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            centroids: [B, N, 3] node 3D positions
            bbox_sizes: [B, N, 3] optional bounding box sizes

        Returns:
            edge_features: [B, N, N, hidden_dim]
        """
        B, N, _ = centroids.shape

        # Relative positions [B, N, N, 3]
        rel_pos = centroids.unsqueeze(2) - centroids.unsqueeze(1)

        # Distances [B, N, N, 1]
        dist = rel_pos.norm(dim=-1, keepdim=True)

        # Normalized relative position
        rel_pos_norm = rel_pos / (dist + 1e-8)

        if bbox_sizes is not None:
            # Relative bbox sizes [B, N, N, 6]
            bbox_i = bbox_sizes.unsqueeze(2).expand(-1, -1, N, -1)
            bbox_j = bbox_sizes.unsqueeze(1).expand(-1, N, -1, -1)
            edge_input = torch.cat([rel_pos_norm, dist, bbox_i, bbox_j], dim=-1)
        else:
            # Pad to expected size
            edge_input = torch.cat([
                rel_pos_norm,
                dist,
                torch.zeros(B, N, N, 6, device=centroids.device)
            ], dim=-1)

        return self.proj(edge_input)


class EdgeAwareAttention(nn.Module):
    """Multi-head attention with edge feature bias."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Edge feature to attention bias
        self.edge_bias = nn.Linear(hidden_dim, num_heads)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, N, hidden_dim]
            edge_features: [B, N, N, hidden_dim] optional
            mask: [B, N] True for valid nodes

        Returns:
            output: [B, N, hidden_dim]
        """
        B, N, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores [B, num_heads, N, N]
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # Add edge bias if provided
        if edge_features is not None:
            # [B, N, N, num_heads] -> [B, num_heads, N, N]
            edge_bias = self.edge_bias(edge_features).permute(0, 3, 1, 2)
            attn = attn + edge_bias

        # Mask invalid nodes
        if mask is not None:
            # [B, 1, 1, N]
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(~mask_expanded, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)  # [B, num_heads, N, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, N, self.hidden_dim)

        return self.out_proj(out)


class TransformerLayer(nn.Module):
    """Single transformer layer with edge-aware attention."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = EdgeAwareAttention(hidden_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention with residual
        x = x + self.attention(self.norm1(x), edge_features, mask)
        # FFN with residual
        x = x + self.ffn(self.norm2(x))
        return x


class GraphTransformer(nn.Module):
    """
    Graph Transformer encoder.

    Takes node features and positions, outputs contextualized node embeddings.
    """

    def __init__(self, config: Dict):
        super().__init__()
        cfg = config['model']

        # Dimensions
        visual_dim = cfg['visual_feat_dim']
        geometric_dim = cfg['geometric_feat_dim']
        category_dim = cfg['category_hint_dim']
        hidden_dim = cfg['transformer']['hidden_dim']
        num_heads = cfg['transformer']['num_heads']
        num_layers = cfg['transformer']['num_layers']
        dropout = cfg['transformer']['dropout']
        use_edge_features = cfg['transformer']['use_edge_features']

        # Ablation flags
        self.use_category_hints = cfg.get('use_category_hints', True)
        self.use_visual_features = cfg.get('use_visual_features', True)
        self.geometric_dim = geometric_dim  # Store for slicing in forward

        # Input projection (concatenate enabled features)
        input_dim = 0
        if self.use_visual_features:
            input_dim += visual_dim
        if geometric_dim > 0:
            input_dim += geometric_dim
        if self.use_category_hints:
            input_dim += category_dim
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        # Positional encoding from 3D positions
        self.pos_encoder = PositionalEncoding3D(hidden_dim)

        # Edge feature encoder
        self.use_edge_features = use_edge_features
        if use_edge_features:
            self.edge_encoder = EdgeFeatureEncoder(hidden_dim)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        visual_features: torch.Tensor,     # [B, N, visual_dim]
        geometric_features: torch.Tensor,  # [B, N, geometric_dim]
        category_hints: torch.Tensor,      # [B, N, category_dim]
        centroids: torch.Tensor,           # [B, N, 3]
        mask: Optional[torch.Tensor] = None,  # [B, N] valid nodes
    ) -> torch.Tensor:
        """
        Returns:
            node_embeddings: [B, N, hidden_dim]
        """
        # Concatenate input features based on what's enabled
        inputs = []

        if self.use_visual_features:
            inputs.append(visual_features)
        if self.geometric_dim > 0:
            # Slice geometric features to desired dimension (supports variable geometry)
            inputs.append(geometric_features[:, :, :self.geometric_dim])
        if self.use_category_hints:
            inputs.append(category_hints)

        x = torch.cat(inputs, dim=-1) if len(inputs) > 1 else inputs[0]
        x = self.input_proj(x)

        # Add positional encoding
        x = x + self.pos_encoder(centroids)

        # Compute edge features
        edge_features = None
        if self.use_edge_features:
            edge_features = self.edge_encoder(centroids)

        # Transformer layers
        for layer in self.layers:
            x = layer(x, edge_features, mask)

        return self.norm(x)


class EdgeTransformer(nn.Module):
    """Small transformer over edges for relationship prediction with cross-edge reasoning."""

    def __init__(self, input_dim: int, hidden_dim: int, num_relations: int, num_layers: int = 2, num_heads: int = 4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_relations),
        )

    def forward(self, edge_features: torch.Tensor, mask: Optional[torch.Tensor] = None, return_features: bool = False) -> torch.Tensor:
        """
        Args:
            edge_features: [B, N, N, input_dim] pairwise features
            mask: [B, N] valid nodes
            return_features: If True, return features before classification (for late fusion)

        Returns:
            edge_logits: [B, N, N, num_relations] if return_features=False
            edge_features: [B, N, N, hidden_dim] if return_features=True
        """
        B, N, _, D = edge_features.shape

        # Flatten edges [B, N*N, D]
        edges_flat = edge_features.view(B, N * N, D)
        edges_flat = self.input_proj(edges_flat)

        # Create edge mask from node mask [B, N*N]
        if mask is not None:
            # Valid edge = both nodes valid
            edge_mask = mask.unsqueeze(2) & mask.unsqueeze(1)  # [B, N, N]
            edge_mask = edge_mask.view(B, N * N)
            # Transformer expects True for positions to IGNORE
            attn_mask = ~edge_mask
        else:
            attn_mask = None

        # Transformer over edges
        edges_out = self.transformer(edges_flat, src_key_padding_mask=attn_mask)

        # Return features before classification for late fusion
        if return_features:
            return edges_out.view(B, N, N, -1)

        # Predict relationships
        edge_logits = self.output_head(edges_out)

        # Reshape back [B, N, N, num_relations]
        return edge_logits.view(B, N, N, -1)


class SceneGraphPredictor(nn.Module):
    """
    Full scene graph prediction model.

    GraphTransformer encoder + node classifier + edge transformer classifier.
    """

    def __init__(self, config: Dict):
        super().__init__()
        cfg = config['model']
        hidden_dim = cfg['transformer']['hidden_dim']
        visual_dim = cfg['visual_feat_dim']
        category_dim = cfg['category_hint_dim']

        # Ablation flags
        self.use_visual_features = cfg.get('use_visual_features', True)

        # Encoder
        self.encoder = GraphTransformer(config)

        # Node classification head (MLP - sufficient with contextualized embeddings)
        node_hidden = cfg['node_head']['hidden_dim']
        num_classes = cfg['node_head']['num_classes']
        self.node_head = nn.Sequential(
            nn.Linear(hidden_dim, node_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(node_hidden, num_classes),
        )

        # Edge classification head (Transformer for cross-edge reasoning)
        edge_hidden = cfg['edge_head'].get('hidden_dim', 256)
        num_relations = cfg['edge_head']['num_relations']
        edge_num_layers = cfg['edge_head'].get('num_layers', 2)
        edge_num_heads = cfg['edge_head'].get('num_heads', 4)
        self.use_object_types = cfg['edge_head'].get('use_object_types', False)
        self.use_node_class_priors = cfg['edge_head'].get('use_node_class_priors', False)
        self.use_visual_edge_features = cfg['edge_head'].get('use_visual_edge_features', False)
        self.use_frame_features = cfg['edge_head'].get('use_frame_features', False)
        self.num_classes = num_classes  # Store for forward pass

        # Frame-level feature projection (global scene context -> edge features)
        # Provides scene-level context that per-object crops lack
        frame_feat_dim = cfg.get('frame_feat_dim', 1024)  # DINOv3 output
        frame_proj_dim = cfg['edge_head'].get('frame_proj_dim', 256)
        if self.use_frame_features:
            self.frame_proj = nn.Sequential(
                nn.Linear(frame_feat_dim, frame_proj_dim),
                nn.LayerNorm(frame_proj_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            )
        else:
            self.frame_proj = None


        # Visual edge feature projection (raw visual -> edge-specific representation)
        # This lets the edge head see "what does this tool look like" directly
        visual_edge_dim = cfg['edge_head'].get('visual_edge_dim', 512)
        if self.use_visual_edge_features:
            self.visual_edge_proj = nn.Sequential(
                nn.Linear(visual_dim * 3, visual_edge_dim),  # [visual_i, visual_j, diff] -> 512
                nn.LayerNorm(visual_edge_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(visual_edge_dim, visual_edge_dim),
                nn.LayerNorm(visual_edge_dim),  # Final norm to match node embedding scale
            )
        else:
            self.visual_edge_proj = None

        # Normalization layers for edge features to ensure equal contribution
        # Without this, node embeddings (~68 norm) drown out visual (~8 norm)
        self.edge_node_norm = nn.LayerNorm(hidden_dim * 3)
        self.edge_probs_norm = nn.LayerNorm(num_classes * 2) if self.use_node_class_priors else None

        # Pairwise geometric features for edge classification (CRITICAL for spatial relationships!)
        # These are explicit spatial features: distance, relative position, height diff, size ratio, overlap
        self.use_pairwise_geometry = cfg['edge_head'].get('use_pairwise_geometry', False)  # Read from edge_head section
        pairwise_geo_dim = 128  # Dimension for projected pairwise geometry

        # Gradient rescaling to prevent geometry from being drowned out by visual features
        self.pairwise_geo_grad_scale = cfg['edge_head'].get('pairwise_geo_grad_scale', 1.0)

        if self.use_pairwise_geometry:
            # Project raw pairwise geometric features to learned embedding
            # Input: 11 features (distance_norm, direction[3], height_diff_norm, size_ratio[3], vol_ratio, iou_3d, dist_to_size)
            self.pairwise_geo_proj = nn.Sequential(
                nn.Linear(11, pairwise_geo_dim),
                nn.LayerNorm(pairwise_geo_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(pairwise_geo_dim, pairwise_geo_dim),
                nn.LayerNorm(pairwise_geo_dim),
            )

            # Register gradient hook to rescale geometry gradients if needed
            if self.pairwise_geo_grad_scale != 1.0:
                for param in self.pairwise_geo_proj.parameters():
                    param.register_hook(lambda grad: grad * self.pairwise_geo_grad_scale)
        else:
            self.pairwise_geo_proj = None

        # Edge input: node_i, node_j, node_i - node_j, [optional: type/visual/frame/geometry features]
        edge_input_dim = hidden_dim * 3
        if self.use_object_types:
            edge_input_dim += category_dim * 2  # SAM3 categories (5 classes each)
        if self.use_node_class_priors:
            edge_input_dim += num_classes * 2  # Predicted node classes (12 classes each)
        if self.use_visual_edge_features:
            edge_input_dim += visual_edge_dim  # Raw visual features for edge
        if self.use_frame_features:
            edge_input_dim += frame_proj_dim  # Global scene context
        if self.use_pairwise_geometry:
            edge_input_dim += pairwise_geo_dim  # Pairwise geometric features

        # Edge classifier: choose between Transformer (complex) or GCN (simple, like LABRAD-OR)
        edge_classifier_type = cfg['edge_head'].get('classifier_type', 'transformer')  # 'transformer' or 'gcn'

        if edge_classifier_type == 'gcn':
            from .gcn_edge_classifier import GCNEdgeClassifier
            dropout = cfg['transformer'].get('dropout', 0.1)
            self.edge_head = GCNEdgeClassifier(
                input_dim=edge_input_dim,
                hidden_dim=edge_hidden,
                num_relations=num_relations,
                num_layers=edge_num_layers,  # Usually 2-3 for GCN
                dropout=dropout,
            )
        else:  # transformer (default)
            self.edge_head = EdgeTransformer(
                input_dim=edge_input_dim,
                hidden_dim=edge_hidden,
                num_relations=num_relations,
                num_layers=edge_num_layers,
                num_heads=edge_num_heads,
            )


        # Proxy task head (frame-level relationship existence prediction)
        self.use_proxy_task = config.get('training', {}).get('proxy_task', {}).get('enabled', False)
        if self.use_proxy_task:
            self.proxy_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, num_relations),
            )
        else:
            self.proxy_head = None

    def _compute_bbox_iou_3d(
        self,
        cent_i: torch.Tensor,
        bbox_i: torch.Tensor,
        cent_j: torch.Tensor,
        bbox_j: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute 3D axis-aligned bounding box IoU.

        Args:
            cent_i: [B, N, N, 3] centroids of objects i
            bbox_i: [B, N, N, 3] bbox sizes of objects i
            cent_j: [B, N, N, 3] centroids of objects j
            bbox_j: [B, N, N, 3] bbox sizes of objects j

        Returns:
            iou: [B, N, N, 1] intersection-over-union in [0, 1]
        """
        # Compute bbox corners (min and max)
        min_i = cent_i - bbox_i / 2
        max_i = cent_i + bbox_i / 2
        min_j = cent_j - bbox_j / 2
        max_j = cent_j + bbox_j / 2

        # Intersection box
        inter_min = torch.max(min_i, min_j)
        inter_max = torch.min(max_i, max_j)
        inter_size = (inter_max - inter_min).clamp(min=0)
        inter_vol = inter_size.prod(dim=-1, keepdim=True)

        # Individual volumes
        vol_i = bbox_i.prod(dim=-1, keepdim=True)
        vol_j = bbox_j.prod(dim=-1, keepdim=True)

        # Union volume
        union_vol = vol_i + vol_j - inter_vol

        # IoU
        iou = inter_vol / union_vol.clamp(min=1e-8)
        return iou

    def _compute_pairwise_geometry(
        self,
        geometric_features: torch.Tensor,
        centroids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute pairwise geometric features for edges.

        Pi3 reconstruction scales are consistent across frames (CV=13.77%),
        so no normalization is needed. Uses raw distances.

        Args:
            geometric_features: [B, N, 13] per-object features
                - [0-2]: centroid (x, y, z)
                - [3-5]: bbox size (w, h, d)
            centroids: [B, N, 3] 3D centroids (redundant but passed for clarity)

        Returns:
            pairwise_features: [B, N, N, 11] with:
                - [0]: 3D distance (raw Euclidean distance)
                - [1-3]: direction (normalized relative position)
                - [4]: height difference (z_j - z_i, raw)
                - [5-7]: size ratio (log-scale, bbox_j / bbox_i)
                - [8]: volume ratio (log-scale, vol_j / vol_i)
                - [9]: 3D IoU (continuous overlap measure)
                - [10]: distance-to-size ratio (distance / mean_object_size)
        """
        B, N, _ = centroids.shape

        # Extract bbox sizes from geometric features (indices 3-5)
        bbox_sizes = geometric_features[:, :, 3:6]  # [B, N, 3]

        # Pairwise centroids [B, N, N, 3]
        cent_i = centroids.unsqueeze(2).expand(-1, -1, N, -1)
        cent_j = centroids.unsqueeze(1).expand(-1, N, -1, -1)

        # Relative position vector [B, N, N, 3]
        rel_pos = cent_j - cent_i

        # 3D Euclidean distance [B, N, N, 1]
        # Raw distance - no normalization needed (Pi3 scales consistent, CV=13.77%)
        distance = rel_pos.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        # Direction (normalized relative position)
        direction = rel_pos / distance

        # Height difference (z_j - z_i) [B, N, N, 1]
        # Raw height - no normalization needed (Pi3 scales consistent)
        height_diff = cent_j[:, :, :, 2:3] - cent_i[:, :, :, 2:3]

        # Pairwise bbox sizes [B, N, N, 3]
        bbox_i = bbox_sizes.unsqueeze(2).expand(-1, -1, N, -1)
        bbox_j = bbox_sizes.unsqueeze(1).expand(-1, N, -1, -1)

        # Size ratio (log-scale, frame-invariant)
        # log(x+1) maps: 1.0→0, 2.7→1, 7.4→2, 20→3, 54→4, 148→5
        size_ratio_log = torch.log(bbox_j / bbox_i.clamp(min=0.01) + 1.0)

        # Volume ratio (log-scale, frame-invariant)
        vol_i = bbox_i.prod(dim=-1, keepdim=True)
        vol_j = bbox_j.prod(dim=-1, keepdim=True)
        volume_ratio_log = torch.log(vol_j / vol_i.clamp(min=1e-6) + 1.0)

        # 3D IoU (continuous overlap, frame-invariant)
        iou_3d = self._compute_bbox_iou_3d(cent_i, bbox_i, cent_j, bbox_j)

        # Distance relative to object sizes (frame-invariant context)
        mean_size = (bbox_i.mean(dim=-1, keepdim=True) + bbox_j.mean(dim=-1, keepdim=True)) / 2
        distance_to_size = distance / mean_size.clamp(min=0.01)

        # Concatenate all features [B, N, N, 11]
        pairwise_features = torch.cat([
            distance,           # 1 - raw distance (no normalization)
            direction,          # 3 - normalized direction
            height_diff,        # 1 - raw height difference (no normalization)
            size_ratio_log,     # 3 - log-scale size ratios
            volume_ratio_log,   # 1 - log-scale volume ratio
            iou_3d,            # 1 - continuous 3D IoU
            distance_to_size,  # 1 - distance / mean object size
        ], dim=-1)

        return pairwise_features

    @torch.no_grad()
    def get_feature_statistics(
        self,
        geometric_features: torch.Tensor,
        centroids: torch.Tensor,
    ) -> dict:
        """
        Compute statistics of pairwise geometric features for debugging.

        Returns dict with min/max/mean/std for each feature component.
        """
        pairwise_geo = self._compute_pairwise_geometry(geometric_features, centroids)

        # Exclude diagonal (self-edges) for meaningful statistics
        B, N, _, _ = pairwise_geo.shape
        # Create mask for non-diagonal elements
        eye_mask = torch.eye(N, dtype=torch.bool, device=pairwise_geo.device)

        stats = {}
        feature_names = ['distance', 'direction_x', 'direction_y', 'direction_z',
                        'height_diff', 'size_ratio_x', 'size_ratio_y', 'size_ratio_z',
                        'volume_ratio', 'iou_3d', 'distance_to_size']

        for i, name in enumerate(feature_names):
            # Extract feature i for all batches, then mask out diagonal
            feat = pairwise_geo[..., i]  # [B, N, N]
            # Mask out diagonal elements
            feat_masked = feat[:, ~eye_mask]  # [B, N*(N-1)]
            stats[name] = {
                'min': feat_masked.min().item(),
                'max': feat_masked.max().item(),
                'mean': feat_masked.mean().item(),
                'std': feat_masked.std().item(),
            }

        # Check direction norm (should be ~1.0)
        direction = pairwise_geo[..., 1:4]  # [B, N, N, 3]
        # Reshape to [B, N*N, 3] then mask
        direction_flat = direction.reshape(B, N*N, 3)
        mask_flat = (~eye_mask).reshape(-1)  # [N*N]
        direction_masked = direction_flat[:, mask_flat, :]  # [B, N*(N-1), 3]
        direction_norm = direction_masked.norm(dim=-1)
        stats['rel_pos_norm'] = {
            'min': direction_norm.min().item(),
            'max': direction_norm.max().item(),
            'mean': direction_norm.mean().item(),
        }

        # IoU statistics (continuous overlap measure)
        iou = pairwise_geo[..., 9]  # [B, N, N] - iou_3d is at index 9
        iou_masked = iou[:, ~eye_mask]  # [B, N*(N-1)]
        stats['iou_mean'] = iou_masked.mean().item()
        stats['overlap_pct'] = (iou_masked > 0.1).float().mean().item() * 100  # % with IoU > 0.1

        return stats

    def forward(
        self,
        visual_features: torch.Tensor,
        geometric_features: torch.Tensor,
        category_hints: torch.Tensor,
        centroids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        frame_embedding: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
            dict with:
                - node_logits: [B, N, num_classes]
                - edge_logits: [B, N, N, num_relations]
                - node_embeddings: [B, N, hidden_dim]
                - proxy_logits: [B, num_relations] (if proxy task enabled)
        """
        # Ablation: zero out visual features if disabled
        # This affects both encoder input and edge head (if use_visual_edge_features)
        if not self.use_visual_features:
            visual_features = torch.zeros_like(visual_features)

        # Encode nodes with Graph Transformer
        node_embeddings = self.encoder(
            visual_features, geometric_features, category_hints, centroids, mask
        )

        # Node classification (MLP)
        node_logits = self.node_head(node_embeddings)

        # Edge classification (Transformer over edges)
        B, N, D = node_embeddings.shape

        # Pairwise features [B, N, N, hidden_dim * 3]
        # IMPORTANT: Normalize each component so they contribute equally
        node_i = node_embeddings.unsqueeze(2).expand(-1, -1, N, -1)
        node_j = node_embeddings.unsqueeze(1).expand(-1, N, -1, -1)
        node_edge_feat = torch.cat([node_i, node_j, node_i - node_j], dim=-1)  # [B, N, N, hidden_dim*3]
        node_edge_feat = self.edge_node_norm(node_edge_feat)  # Normalize to unit scale

        edge_features = [node_edge_feat]

        # Add object type information (SAM3 category hints) for strong priors
        if self.use_object_types:
            cat_i = category_hints.unsqueeze(2).expand(-1, -1, N, -1)  # [B, N, N, cat_dim]
            cat_j = category_hints.unsqueeze(1).expand(-1, N, -1, -1)  # [B, N, N, cat_dim]
            edge_features.extend([cat_i, cat_j])

        # Add predicted node class probabilities as priors (detached - no gradient flow)
        # This tells the edge head: "node i is probably human_0, node j is probably instrument"
        if self.use_node_class_priors:
            node_probs = F.softmax(node_logits, dim=-1).detach()  # [B, N, num_classes]
            probs_i = node_probs.unsqueeze(2).expand(-1, -1, N, -1)  # [B, N, N, num_classes]
            probs_j = node_probs.unsqueeze(1).expand(-1, N, -1, -1)  # [B, N, N, num_classes]
            probs_edge = torch.cat([probs_i, probs_j], dim=-1)  # [B, N, N, num_classes*2]
            probs_edge = self.edge_probs_norm(probs_edge)  # Normalize
            edge_features.append(probs_edge)

        # Add raw visual features to edge head (bypasses transformer compression)
        # This lets edge head see "what does this tool look like" directly
        if self.visual_edge_proj is not None:
            visual_i = visual_features.unsqueeze(2).expand(-1, -1, N, -1)  # [B, N, N, visual_dim]
            visual_j = visual_features.unsqueeze(1).expand(-1, N, -1, -1)  # [B, N, N, visual_dim]
            visual_pair = torch.cat([visual_i, visual_j, visual_i - visual_j], dim=-1)  # [B, N, N, visual_dim*3]
            visual_edge = self.visual_edge_proj(visual_pair)  # [B, N, N, visual_edge_dim] (already normed)

            edge_features.append(visual_edge)

        # Add frame-level features (global scene context) via late fusion
        # Frame embedding is shared across all edges in the scene
        if self.frame_proj is not None and frame_embedding is not None:
            frame_feat = self.frame_proj(frame_embedding)  # [B, frame_proj_dim]
            # Expand to all edges: [B, N, N, frame_proj_dim]
            frame_feat_expanded = frame_feat.unsqueeze(1).unsqueeze(1).expand(-1, N, N, -1)
            edge_features.append(frame_feat_expanded)

        # Add pairwise geometric features (distance, relative position, size ratio, etc.)
        # CRITICAL for spatial relationships like "LyingOn", "Touching", "CloseTo"
        if self.pairwise_geo_proj is not None:
            pairwise_geo = self._compute_pairwise_geometry(geometric_features, centroids)  # [B, N, N, 11]
            pairwise_geo_emb = self.pairwise_geo_proj(pairwise_geo)  # [B, N, N, pairwise_geo_dim]
            edge_features.append(pairwise_geo_emb)

        edge_features = torch.cat(edge_features, dim=-1)

        # Edge classification
        edge_logits = self.edge_head(edge_features, mask)

        outputs = {
            'node_logits': node_logits,
            'edge_logits': edge_logits,
            'node_embeddings': node_embeddings,
        }

        # Proxy task: frame-level relationship existence prediction
        if self.proxy_head is not None:
            # Global pooling over valid nodes -> frame representation
            if mask is not None:
                # Masked mean pooling
                mask_expanded = mask.unsqueeze(-1).float()  # [B, N, 1]
                frame_features = (node_embeddings * mask_expanded).sum(dim=1)  # [B, D]
                frame_features = frame_features / mask_expanded.sum(dim=1).clamp(min=1)
            else:
                frame_features = node_embeddings.mean(dim=1)  # [B, D]

            proxy_logits = self.proxy_head(frame_features)  # [B, num_relations]
            outputs['proxy_logits'] = proxy_logits

        return outputs


def build_model(config: Dict) -> SceneGraphPredictor:
    """Build model from config."""
    return SceneGraphPredictor(config)
