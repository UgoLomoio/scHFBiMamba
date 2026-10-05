"""
GCN classifier on the cell-cell kNN graph (classifier ablation arm).

Pure-PyTorch implementation (no torch_geometric dependency) following the
codebase style. Operates transductively: one forward pass classifies all
graph nodes; the loss is applied to training nodes only.

Architecture:
  Input: (N, 350) latent features + normalized sparse adjacency (N, N)
    |
    +-- GCN layers: H' = ReLU(Â H W) with dropout
    |
    Output: (N, 13) multi-label logits
"""
import torch
import torch.nn as nn

from config import N_CLASSES


class GCNLayer(nn.Module):
    """Single GCN layer: H' = Â H W with Â the normalized adjacency."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, in_dim) node features
            adj_norm: (N, N) normalized adjacency (torch sparse COO)
        Returns:
            (N, out_dim)
        """
        return torch.sparse.mm(adj_norm, self.linear(x))


class GCNClassifier(nn.Module):
    """
    Multi-layer GCN classifier for multi-label cell classification.

    Same output contract as BiMambaClassifier/MLPClassifier: (N, 13) logits.
    """

    def __init__(self, input_dim: int = 350, hidden_dim: int = 128,
                 n_layers: int = 2, dropout: float = 0.1,
                 n_outputs: int = N_CLASSES):
        super().__init__()
        self.config = {
            'input_dim': input_dim, 'hidden_dim': hidden_dim,
            'n_layers': n_layers, 'dropout': dropout, 'n_outputs': n_outputs,
        }

        dims = [input_dim] + [hidden_dim] * (n_layers - 1) + [n_outputs]
        self.layers = nn.ModuleList([
            GCNLayer(dims[i], dims[i + 1]) for i in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, input_dim) latent features for all graph nodes
            adj_norm: (N, N) normalized adjacency (torch sparse COO)
        Returns:
            (N, 13) multi-label logits
        """
        for i, layer in enumerate(self.layers):
            x = layer(x, adj_norm)
            if i < len(self.layers) - 1:
                x = torch.relu(x)
                x = self.dropout(x)
        return x
