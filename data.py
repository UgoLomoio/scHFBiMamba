"""
Data loading, memmap conversion, label encoding, and streaming for HF Mamba pipeline.
Handles the CVD_dataset from HuggingFace (mruzjurado/CVD_dataset).
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional, Dict

from config import (
    ALL_CLASSES, N_CLASSES, SPECIES_CLASSES, CELLTYPE_CLASSES,
    DISEASE_CLASSES, N_GENES, MT_GENE_PREFIX, DataConfig
)


# ─── Label encoding ──────────────────────────────────────────────────────────

def parse_label(label_str: str) -> Tuple[str, str, str]:
    """
    Parse a label string like "Human-HFpEF_Cardiomyocytes" into
    (species, disease, cell_type).

    Format: "Species-Disease_CellType"
    The first '-' separates species from the rest.
    The first '_' separates disease from cell_type.
    """
    # Remove quotes if present
    label_str = label_str.strip().strip('"').strip("'")
    # Split on first '-' -> species and rest
    parts = label_str.split('-', 1)
    species = parts[0]
    rest = parts[1] if len(parts) > 1 else ""
    # Split rest on first '_' -> disease and cell_type
    parts2 = rest.split('_', 1)
    disease = parts2[0]
    cell_type = parts2[1] if len(parts2) > 1 else ""
    return species, disease, cell_type


def label_to_multihot(label_str: str) -> np.ndarray:
    """
    Convert a label string to 13-dim multi-hot vector.
    Format: [species(2), cell_type(7), disease(4)]
    Each cell gets exactly 3 ones (one per category).
    """
    species, disease, cell_type = parse_label(label_str)
    vec = np.zeros(N_CLASSES, dtype=np.float32)

    # Species
    if species in SPECIES_CLASSES:
        vec[SPECIES_CLASSES.index(species)] = 1.0

    # Cell type
    if cell_type in CELLTYPE_CLASSES:
        vec[2 + CELLTYPE_CLASSES.index(cell_type)] = 1.0

    # Disease
    if disease in DISEASE_CLASSES:
        vec[9 + DISEASE_CLASSES.index(disease)] = 1.0

    return vec


def load_labels(label_path: str) -> np.ndarray:
    """
    Load label CSV and return multi-hot encoded array (N, 13).
    First row is a header (skip it).
    """
    df = pd.read_csv(label_path, header=0)
    labels = df.values.flatten().tolist()
    multihot = np.array([label_to_multihot(l) for l in labels], dtype=np.float32)
    return multihot


def load_label_strings(label_path: str) -> List[str]:
    """Load raw label strings (for grouping in DXG analysis)."""
    df = pd.read_csv(label_path, header=0)
    return df.values.flatten().tolist()


# ─── Gene names ──────────────────────────────────────────────────────────────

def load_gene_names(data_path: str) -> List[str]:
    """Load gene names from the header of a data CSV."""
    # Read only the header
    df_header = pd.read_csv(data_path, nrows=0, index_col=0)
    return df_header.columns.tolist()


def get_mt_gene_indices(gene_names: List[str]) -> np.ndarray:
    """Return indices of mitochondrial genes (to exclude from SHAP)."""
    mt_indices = []
    for i, g in enumerate(gene_names):
        if g.startswith(MT_GENE_PREFIX) or g.startswith('MT-') or g.startswith('mt-'):
            mt_indices.append(i)
    return np.array(mt_indices, dtype=np.int64)


# ─── Memmap conversion ───────────────────────────────────────────────────────

def csv_to_memmap(csv_path: str, cache_dir: str, name: str,
                  dtype: np.dtype = np.float32) -> Tuple[np.ndarray, List[str]]:
    """
    Convert a large CSV to a memory-mapped .npy file for efficient streaming.
    The CSV has a header row with gene names and first column with cell IDs.

    Returns:
        memmap: (N_cells, N_genes) memory-mapped array
        gene_names: list of gene name strings
    """
    os.makedirs(cache_dir, exist_ok=True)
    npy_path = os.path.join(cache_dir, f"{name}.npy")
    genes_path = os.path.join(cache_dir, f"{name}_genes.txt")

    # If already cached, load
    if os.path.exists(npy_path) and os.path.exists(genes_path):
        print(f"Loading cached memmap: {npy_path}")
        memmap = np.load(npy_path, mmap_mode='r')
        with open(genes_path, 'r') as f:
            gene_names = f.read().strip().split('\n')
        return memmap, gene_names

    # Otherwise, convert
    print(f"Converting {csv_path} to memmap...")
    # First pass: count rows and get gene names
    df_header = pd.read_csv(csv_path, nrows=0, index_col=0)
    gene_names = df_header.columns.tolist()
    n_genes = len(gene_names)

    # Count total rows (excluding header)
    n_rows = sum(1 for _ in open(csv_path)) - 1
    print(f"  {n_rows} cells x {n_genes} genes")

    # Create memmap
    memmap = np.lib.format.open_memmap(npy_path, mode='w+', dtype=dtype,
                                       shape=(n_rows, n_genes))

    # Stream CSV in chunks
    # NOTE: do not pass dtype= to read_csv — pandas >= 2.x would also try to
    # cast the index (cell ID) column. Cast values after reading instead.
    chunksize = 5000
    reader = pd.read_csv(csv_path, index_col=0, chunksize=chunksize,
                         low_memory=False)
    offset = 0
    for chunk in reader:
        n = len(chunk)
        memmap[offset:offset + n] = chunk.values.astype(dtype, copy=False)
        offset += n
        if offset % 20000 == 0:
            print(f"  Processed {offset}/{n_rows} rows...")

    memmap.flush()
    del memmap

    # Save gene names
    with open(genes_path, 'w') as f:
        f.write('\n'.join(gene_names))

    # Reload in read mode
    memmap = np.load(npy_path, mmap_mode='r')
    print(f"  Done. Cached to {npy_path}")
    return memmap, gene_names


# ─── PyTorch Dataset ─────────────────────────────────────────────────────────

class HFDataset(Dataset):
    """
    Dataset for HF Mamba training/val/test.
    Loads from memmap + pre-computed labels.
    """
    def __init__(self, features: np.ndarray, labels: np.ndarray,
                 device: torch.device, use_dae: bool = False,
                 dae_features: Optional[np.ndarray] = None):
        """
        Args:
            features: (N, N_genes) memmap or array of expression values
            labels: (N, 13) multi-hot encoded labels
            device: torch device
            use_dae: if True, use dae_features instead of raw features
            dae_features: (N, latent_dim) pre-computed DAE latent features
        """
        self.features = dae_features if use_dae else features
        self.labels = labels
        self.device = device
        self.use_dae = use_dae

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(np.asarray(self.features[idx], dtype=np.float32))
        y = torch.tensor(self.labels[idx])
        return x, y


def create_dataloader(features: np.ndarray, labels: np.ndarray,
                      batch_size: int, shuffle: bool, device: torch.device,
                      use_dae: bool = False, dae_features: Optional[np.ndarray] = None,
                      num_workers: int = 4, pin_memory: bool = True) -> DataLoader:
    """Create a DataLoader from features and labels."""
    dataset = HFDataset(features, labels, device, use_dae, dae_features)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=pin_memory,
                      drop_last=False)


# ─── Balanced test subset for SHAP ───────────────────────────────────────────

def sample_balanced_test_subset(
    test_features: np.ndarray,
    test_labels_str: List[str],
    n_per_celltype: int = 200,
    seed: int = 1111
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Sample a balanced subset of test cells: n_per_celltype cells per cell type.
    Returns features, multi-hot labels, and label strings for the subset.
    """
    rng = np.random.RandomState(seed)

    # Group cell indices by cell type
    celltype_indices: Dict[str, List[int]] = {}
    for i, label_str in enumerate(test_labels_str):
        _, _, cell_type = parse_label(label_str)
        if cell_type not in celltype_indices:
            celltype_indices[cell_type] = []
        celltype_indices[cell_type].append(i)

    # Sample
    selected_indices = []
    for ct, indices in celltype_indices.items():
        n_sample = min(n_per_celltype, len(indices))
        sampled = rng.choice(indices, size=n_sample, replace=False)
        selected_indices.extend(sampled)

    selected_indices = sorted(selected_indices)
    subset_features = np.asarray(test_features[selected_indices], dtype=np.float32)
    subset_labels_str = [test_labels_str[i] for i in selected_indices]
    subset_labels = np.array([label_to_multihot(l) for l in subset_labels_str],
                             dtype=np.float32)

    return subset_features, subset_labels, subset_labels_str


# ─── Cross-validation ────────────────────────────────────────────────────────

def stratified_kfold_indices(labels_str: List[str], k: int, seed: int = 1234
                             ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Stratified k-fold split over label strings.

    The full label string ("Species-Disease_CellType") uniquely determines the
    13-dim multi-hot label, so stratifying by it preserves the joint
    species/celltype/disease distribution in every fold.

    Args:
        labels_str: list of label strings for the pooled samples
        k: number of folds (k >= 2; k=1 is handled by the caller as the
           existing train/val split)
        seed: shuffle seed

    Returns:
        list of k (train_idx, val_idx) index arrays
    """
    if k < 2:
        raise ValueError("stratified_kfold_indices requires k >= 2")

    rng = np.random.RandomState(seed)
    labels_arr = np.asarray(labels_str)
    fold_val = [[] for _ in range(k)]

    for cls in np.unique(labels_arr):
        idx = np.where(labels_arr == cls)[0]
        rng.shuffle(idx)
        if len(idx) < k:
            print(f"  Warning: class '{cls}' has {len(idx)} samples < k={k}; "
                  f"it will be absent from some validation folds")
        for i, sample_idx in enumerate(idx):
            fold_val[i % k].append(sample_idx)

    folds = []
    all_idx = np.arange(len(labels_str))
    for f in range(k):
        val_idx = np.sort(np.array(fold_val[f], dtype=np.int64))
        train_idx = np.setdiff1d(all_idx, val_idx)
        folds.append((train_idx, val_idx))
    return folds


# ─── Full data preparation ───────────────────────────────────────────────────

def prepare_data(data_cfg: DataConfig, device: torch.device,
                 use_dae: bool = False) -> Dict:
    """
    Prepare all data splits. Converts CSVs to memmap if needed.
    Returns dict with train/val/test features, labels, dataloaders, gene_names.
    """
    cache_dir = data_cfg.cache_dir
    os.makedirs(cache_dir, exist_ok=True)

    # Convert CSVs to memmap
    print("Preparing training data...")
    train_features, gene_names = csv_to_memmap(
        data_cfg.train_path, cache_dir, "train")
    print("Preparing validation data...")
    val_features, _ = csv_to_memmap(
        data_cfg.val_path, cache_dir, "val")
    print("Preparing test data...")
    test_features, _ = csv_to_memmap(
        data_cfg.test_path, cache_dir, "test")

    # Load labels
    print("Loading labels...")
    train_labels = load_labels(data_cfg.train_labels_path)
    val_labels = load_labels(data_cfg.val_labels_path)
    test_labels = load_labels(data_cfg.test_labels_path)
    test_labels_str = load_label_strings(data_cfg.test_labels_path)

    # MT gene indices
    mt_indices = get_mt_gene_indices(gene_names)

    print(f"Train: {train_features.shape}, Val: {val_features.shape}, "
          f"Test: {test_features.shape}")
    print(f"Labels: train={train_labels.shape}, val={val_labels.shape}, "
          f"test={test_labels.shape}")
    print(f"MT genes: {len(mt_indices)} (excluded from SHAP)")

    return {
        'train_features': train_features,
        'val_features': val_features,
        'test_features': test_features,
        'train_labels': train_labels,
        'val_labels': val_labels,
        'test_labels': test_labels,
        'test_labels_str': test_labels_str,
        'gene_names': gene_names,
        'mt_indices': mt_indices,
    }
