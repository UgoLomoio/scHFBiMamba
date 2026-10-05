"""
GAT classifier on the cell-cell kNN graph (classifier ablation arm).

Pure-PyTorch multi-head graph attention over a sparse edge list (no
torch_geometric / torch_scatter dependency). Same graph and output contract
as the GCN arm: one transductive forward pass yields (N, 13) logits.

Architecture per layer:
  e_ij   = LeakyReLU(a^T [W h_i || W h_j])   for each edge (i, j)
  alpha  = softmax_j(e_ij) per target node i (scatter softmax)
  h_i'   = sum_j alpha_ij W h_j              (mean over heads, last layer)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import N_CLASSES


class GATLayer(nn.Module):
    """Single multi-head GAT layer over a sparse edge list."""

    def __init__(self, in_dim: int, out_dim: int, n_heads: int = 4,
                 dropout: float = 0.1, concat: bool = True):
        super().__init__()
        self.n_heads = n_heads
        self.out_dim = out_dim
        self.concat = concat

        self.linear = nn.Linear(in_dim, n_heads * out_dim, bias=False)
        self.attn_src = nn.Parameter(torch.empty(n_heads, out_dim))
        self.attn_dst = nn.Parameter(torch.empty(n_heads, out_dim))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, in_dim) node features
            edge_index: (2, E) edges (source, target), including self-loops
        Returns:
            (N, n_heads * out_dim) if concat else (N, out_dim)
        """
        n = x.shape[0]
        h = self.linear(x).view(n, self.n_heads, self.out_dim)  # (N, H, D)

        src, dst = edge_index[0], edge_index[1]
        # Attention logits per edge and head
        e = (h[src] * self.attn_src).sum(-1) + (h[dst] * self.attn_dst).sum(-1)
        e = self.leaky_relu(e)  # (E, H)

        # Scatter softmax over incoming edges per target node
        e_max = torch.full((n, self.n_heads), float('-inf'), device=x.device)
        e_max.scatter_reduce_(0, dst.unsqueeze(-1).expand_as(e), e,
                              reduce='amax', include_self=True)
        alpha = torch.exp(e - e_max[dst])
        denom = torch.zeros(n, self.n_heads, device=x.device)
        denom.index_add_(0, dst, alpha)
        alpha = alpha / (denom[dst] + 1e-16)
        alpha = self.dropout(alpha)

        # Weighted message passing
        messages = h[src] * alpha.unsqueeze(-1)  # (E, H, D)
        out = torch.zeros_like(h)
        out.index_add_(0, dst, messages)

        if self.concat:
            return out.reshape(n, self.n_heads * self.out_dim)
        return out.mean(dim=1)


class GATClassifier(nn.Module):
    """
    Multi-layer GAT classifier for multi-label cell classification.

    Same output contract as the other classifier arms: (N, 13) logits.
    """

    def __init__(self, input_dim: int = 350, hidden_dim: int = 128,
                 n_layers: int = 2, n_heads: int = 4, dropout: float = 0.1,
                 n_outputs: int = N_CLASSES):
        super().__init__()
        self.config = {
            'input_dim': input_dim, 'hidden_dim': hidden_dim,
            'n_layers': n_layers, 'n_heads': n_heads, 'dropout': dropout,
            'n_outputs': n_outputs,
        }

        self.layers = nn.ModuleList()
        head_dim = max(hidden_dim // n_heads, 1)
        dims = [input_dim] + [hidden_dim] * (n_layers - 1)
        for i in range(n_layers - 1):
            self.layers.append(
                GATLayer(dims[i], head_dim, n_heads=n_heads,
                         dropout=dropout, concat=True))
        head_out = head_dim * n_heads if n_layers > 1 else input_dim
        self.out_layer = GATLayer(head_out, n_outputs, n_heads=1,
                                  dropout=dropout, concat=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, input_dim) latent features for all graph nodes
            edge_index: (2, E) edges including self-loops
        Returns:
            (N, 13) multi-label logits
        """
        for layer in self.layers:
            x = layer(x, edge_index)
            x = F.elu(x)
            x = self.dropout(x)
        return self.out_layer(x, edge_index)
