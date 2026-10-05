"""
Cell-cell kNN graph construction from autoencoder latent representations.

Used by the GCN/GAT classifier ablation arms. The graph is built over the
pooled train+val+test latent features (transductive, features only): test
cells participate in the graph topology, but their labels are never used
during training.

Sparse representation throughout: the real dataset is far too large for a
dense N x N adjacency matrix.
"""
import numpy as np
import scipy.sparse as sp
import torch
from typing import Tuple


def build_knn_graph(features: np.ndarray, k: int = 15,
                    metric: str = "euclidean") -> sp.csr_matrix:
    """
    Build a symmetric kNN graph over cells from latent features.

    Args:
        features: (N, D) latent representations (e.g. 350-dim DAE latent)
        k: number of nearest neighbors (excluding self)
        metric: distance metric passed to sklearn NearestNeighbors

    Returns:
        (N, N) scipy CSR adjacency, symmetrized (union of kNN edges),
        without self-loops (self-loops are added by the model-specific
        normalization step)
    """
    from sklearn.neighbors import NearestNeighbors

    n = features.shape[0]
    k = min(k, n - 1)
    if k < 1:
        raise ValueError(f"kNN graph needs at least 2 cells, got {n}")

    print(f"  Building kNN graph: N={n}, k={k}, metric={metric}")
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric, algorithm='auto')
    nn.fit(features)
    distances, indices = nn.kneighbors(features)

    # Drop the first neighbor (the cell itself)
    rows = np.repeat(np.arange(n), k)
    cols = indices[:, 1:].ravel()
    data = np.ones(len(rows), dtype=np.float32)

    adj = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    # Symmetrize: union of directed kNN edges
    adj = adj.maximum(adj.T)
    print(f"  Graph edges: {adj.nnz:,} (symmetric)")
    return adj


def normalize_adjacency_gcn(adj: sp.csr_matrix) -> torch.Tensor:
    """
    Symmetric normalization with self-loops: D^-1/2 (A + I) D^-1/2 (GCN).

    Args:
        adj: (N, N) scipy sparse adjacency without self-loops

    Returns:
        torch sparse COO tensor (N, N)
    """
    adj_hat = adj + sp.identity(adj.shape[0], dtype=np.float32, format='csr')
    degree = np.asarray(adj_hat.sum(axis=1)).ravel()
    with np.errstate(divide='ignore'):
        d_inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree), 0.0)
    d_mat = sp.diags(d_inv_sqrt.astype(np.float32))
    adj_norm = d_mat @ adj_hat @ d_mat
    return scipy_to_torch_sparse(adj_norm)


def adjacency_to_edge_index(adj: sp.csr_matrix,
                            add_self_loops: bool = True) -> torch.Tensor:
    """
    Convert scipy adjacency to a (2, E) edge index tensor (GAT).

    Args:
        adj: (N, N) scipy sparse adjacency
        add_self_loops: whether to append self-loop edges

    Returns:
        (2, E) int64 tensor
    """
    coo = adj.tocoo()
    row = torch.from_numpy(coo.row.astype(np.int64))
    col = torch.from_numpy(coo.col.astype(np.int64))
    edge_index = torch.stack([row, col], dim=0)
    if add_self_loops:
        n = adj.shape[0]
        loops = torch.arange(n, dtype=torch.int64)
        edge_index = torch.cat([edge_index, torch.stack([loops, loops], dim=0)],
                               dim=1)
    return edge_index


def scipy_to_torch_sparse(mat: sp.csr_matrix) -> torch.Tensor:
    """Convert a scipy sparse matrix to a coalesced torch sparse COO tensor."""
    coo = mat.tocoo()
    indices = torch.from_numpy(
        np.stack([coo.row, coo.col], axis=0).astype(np.int64))
    values = torch.from_numpy(coo.data.astype(np.float32))
    return torch.sparse_coo_tensor(indices, values, coo.shape).coalesce()
