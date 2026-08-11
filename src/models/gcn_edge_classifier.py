"""
Simple GCN-based edge classifier, following LABRAD-OR architecture.

Much simpler than 4-layer Edge Transformer - just message passing + MLPs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeGCNLayer(nn.Module):
    """
    Single GCN layer that updates both nodes and edges.

    For each edge (i, j):
    - Input: [node_i, edge_feature, node_j]
    - Output: [updated_node_i, updated_edge, updated_node_j]
    - Aggregate to update all nodes and edges
    """

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim

        # MLP to process [node_i, edge, node_j] -> [new_node_i, new_edge, new_node_j]
        self.message_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, node_dim * 2 + edge_dim),
        )

        # MLP to update nodes after aggregation
        self.node_update = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, node_dim),
        )

    def forward(self, node_features: torch.Tensor, edge_features: torch.Tensor, mask=None):
        """
        Args:
            node_features: [B, N, node_dim]
            edge_features: [B, N, N, edge_dim]
            mask: [B, N] optional validity mask

        Returns:
            updated_nodes: [B, N, node_dim]
            updated_edges: [B, N, N, edge_dim]
        """
        B, N, node_dim = node_features.shape
        _, _, _, edge_dim = edge_features.shape

        # Pairwise node features [B, N, N, node_dim]
        node_i = node_features.unsqueeze(2).expand(-1, -1, N, -1)
        node_j = node_features.unsqueeze(1).expand(-1, N, -1, -1)

        # Concatenate [B, N, N, node_dim*2 + edge_dim]
        triplet = torch.cat([node_i, edge_features, node_j], dim=-1)

        # Process through MLP [B, N, N, node_dim*2 + edge_dim]
        messages = self.message_mlp(triplet)

        # Split into node and edge updates
        msg_i = messages[:, :, :, :node_dim]  # [B, N, N, node_dim]
        msg_edge = messages[:, :, :, node_dim:(node_dim + edge_dim)]  # [B, N, N, edge_dim]
        msg_j = messages[:, :, :, (node_dim + edge_dim):]  # [B, N, N, node_dim]

        # Aggregate messages: sum over neighbors
        # For node i: aggregate all messages from edges (i, j) where j is a neighbor
        aggregated_i = msg_i.mean(dim=2)  # [B, N, node_dim] - mean over j
        aggregated_j = msg_j.mean(dim=1)  # [B, N, node_dim] - mean over i
        aggregated_nodes = (aggregated_i + aggregated_j) / 2  # [B, N, node_dim]

        # Update nodes
        updated_nodes = node_features + self.node_update(aggregated_nodes)

        # Update edges directly from messages
        updated_edges = edge_features + msg_edge

        # Apply mask if provided
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()  # [B, N, 1]
            updated_nodes = updated_nodes * mask_expanded

        return updated_nodes, updated_edges


class GCNEdgeClassifier(nn.Module):
    """
    GCN-based edge classifier following LABRAD-OR architecture.

    Much simpler than Edge Transformer:
    - 2-3 GCN layers (not 4 transformer layers)
    - Message passing (not attention)
    - ~10x fewer parameters
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_relations: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_relations = num_relations
        self.num_layers = num_layers

        # Project edge features to hidden dim
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # GCN layers (node_dim = hidden_dim, edge_dim = hidden_dim)
        self.gcn_layers = nn.ModuleList([
            EdgeGCNLayer(
                node_dim=hidden_dim,
                edge_dim=hidden_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_relations),
        )

    def forward(self, edge_features: torch.Tensor, mask=None):
        """
        Args:
            edge_features: [B, N, N, input_dim] edge features
            mask: [B, N] optional node validity mask

        Returns:
            logits: [B, N, N, num_relations] relationship predictions
        """
        B, N, _, _ = edge_features.shape

        # Project to hidden dim
        edge_hidden = self.input_proj(edge_features)  # [B, N, N, hidden_dim]

        # Initialize node features (mean pool from edges)
        # For each node i, average over all edges (i, j)
        node_hidden = edge_hidden.mean(dim=2)  # [B, N, hidden_dim]

        # Apply GCN layers
        for i, gcn_layer in enumerate(self.gcn_layers):
            node_hidden, edge_hidden = gcn_layer(node_hidden, edge_hidden, mask)

            # ReLU between layers (not on last layer)
            if i < self.num_layers - 1:
                node_hidden = F.relu(node_hidden)
                edge_hidden = F.relu(edge_hidden)

        # Classify edges
        logits = self.classifier(edge_hidden)  # [B, N, N, num_relations]

        return logits
