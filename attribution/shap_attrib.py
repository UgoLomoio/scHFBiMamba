"""
SHAP attribution for HF Mamba classifiers.

Uses shap.GradientExplainer to compute Shapley value attribution scores.
All models use the composite model wrapper (encoder + classifier) so that
attributions are computed at the GENE level (16,545 genes) for all models,
regardless of whether the encoder is a Dense DAE or BiMamba AE.

Protocol (matching paper):
  1. Background: 1000 cells randomly sampled from training set (seed=1111)
  2. Test set: balanced subset (200 cells per cell type)
  3. Compute SHAP values for each of 13 output classes
  4. Exclude mitochondrial genes
  5. Z-transform per cell (done in dxg.py)
"""
import os
import numpy as np
import torch
from typing import Tuple, Optional, List, Dict

from config import N_GENES, N_CLASSES, ALL_CLASSES, MT_GENE_PREFIX
from data import label_to_multihot, parse_label


def prepare_shap_background(train_features: np.ndarray,
                            background_size: int = 1000,
                            seed: int = 1111) -> np.ndarray:
    """
    Sample background cells from training data for SHAP.
    Matches paper: 1000 random cells, seed=1111.
    """
    rng = np.random.RandomState(seed)
    n = len(train_features)
    n_sample = min(background_size, n)
    indices = rng.choice(n, size=n_sample, replace=False)
    background = np.asarray(train_features[indices], dtype=np.float32)
    print(f"  SHAP background: {background.shape} (seed={seed})")
    return background


def compute_shap_attributions_composite(
    composite_model: torch.nn.Module,
    test_features: np.ndarray,
    train_features: np.ndarray,
    device: torch.device,
    background_size: int = 1000,
    seed: int = 1111,
    batch_size: int = 50,
    mt_indices: Optional[np.ndarray] = None,
    nsamples: int = 100,
) -> np.ndarray:
    """
    Compute SHAP attribution scores at the GENE level for all 13 classes,
    using the composite model (encoder + classifier).

    The composite model takes (B, 16545) raw gene expression and outputs
    (B, 13) classification probabilities. SHAP values are computed for
    each input gene, giving gene-level attributions for all models.

    Args:
        composite_model: CompositeModel (encoder + classifier)
        test_features: (N_test, N_genes) test set expression values
        train_features: (N_train, N_genes) training set for background sampling
        device: torch device
        background_size: number of background cells (paper: 1000)
        seed: random seed for background sampling (paper: 1111)
        batch_size: batch size for SHAP computation
        mt_indices: indices of MT genes to zero out in attributions
        nsamples: number of samples for SHAP expectation estimation

    Returns:
        shap_values: (N_classes, N_test, N_genes) SHAP attribution scores
    """
    import shap

    composite_model.eval()

    # Prepare background (raw gene expression)
    background = prepare_shap_background(train_features, background_size, seed)
    test_input = np.asarray(test_features, dtype=np.float32)

    print(f"  SHAP test input: {test_input.shape}")
    print(f"  SHAP background: {background.shape}")

    # Convert to tensors
    background_t = torch.from_numpy(background).float().to(device)
    test_t = torch.from_numpy(test_input).float().to(device)

    # Build SHAP GradientExplainer on the composite model
    print(f"  Building SHAP GradientExplainer on composite model...")
    explainer = shap.GradientExplainer(composite_model, background_t)

    # Compute SHAP values for all 13 output classes
    print(f"  Computing SHAP values for {len(test_input)} cells, {N_CLASSES} classes...")

    shap_values_list = []
    n_test = len(test_input)

    for cls_idx in range(N_CLASSES):
        print(f"    Class {cls_idx}/{N_CLASSES} ({ALL_CLASSES[cls_idx]})...")

        cls_shap = []
        for i in range(0, n_test, batch_size):
            batch = test_t[i:i + batch_size]

            # Use a wrapper to get single-class output
            class _SingleOutput(torch.nn.Module):
                def __init__(self, model, idx):
                    super().__init__()
                    self.model = model
                    self.idx = idx
                def forward(self, x):
                    return self.model(x)[:, self.idx].unsqueeze(-1)

            wrapper = _SingleOutput(composite_model, cls_idx).to(device)
            explainer_cls = shap.GradientExplainer(wrapper, background_t)
            sv = explainer_cls.shap_values(batch, nsamples=nsamples)

            if isinstance(sv, list):
                cls_shap.append(sv[0])
            elif isinstance(sv, np.ndarray):
                if sv.ndim == 3:
                    cls_shap.append(sv[:, :, 0])
                else:
                    cls_shap.append(sv)
            else:
                cls_shap.append(np.array(sv))

        cls_shap = np.concatenate(cls_shap, axis=0)  # (N_test, N_genes)
        shap_values_list.append(cls_shap)

    shap_values = np.array(shap_values_list)  # (N_classes, N_test, N_genes)

    # Zero out MT gene attributions (per paper)
    if mt_indices is not None and len(mt_indices) > 0:
        shap_values[:, :, mt_indices] = 0.0
        print(f"  Zeroed out {len(mt_indices)} MT gene attributions")

    print(f"  SHAP values shape: {shap_values.shape}")
    return shap_values


def compute_shap_attributions(
    model: torch.nn.Module,
    test_features: np.ndarray,
    train_features: np.ndarray,
    device: torch.device,
    background_size: int = 1000,
    seed: int = 1111,
    batch_size: int = 100,
    use_dae: bool = False,
    dae_model: Optional[torch.nn.Module] = None,
    mt_indices: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Legacy SHAP attribution (for standalone classifier without composite model).
    Kept for backward compatibility.

    If use_dae=True, attributes to DAE latent dims (not genes).
    For gene-level attribution with encoders, use compute_shap_attributions_composite.
    """
    import shap

    model.eval()

    background = prepare_shap_background(train_features, background_size, seed)

    if use_dae and dae_model is not None:
        dae_model.eval()
        with torch.no_grad():
            bg_t = torch.from_numpy(background).to(device)
            background = dae_model.get_latent(bg_t).cpu().numpy()
            test_t = torch.from_numpy(
                np.asarray(test_features, dtype=np.float32)
            ).to(device)
            test_input = dae_model.get_latent(test_t).cpu().numpy()
    else:
        test_input = np.asarray(test_features, dtype=np.float32)

    background_t = torch.from_numpy(background).float().to(device)
    test_t = torch.from_numpy(test_input).float().to(device)

    explainer = shap.GradientExplainer(model, background_t)

    shap_values_list = []
    n_test = len(test_input)

    for cls_idx in range(N_CLASSES):
        print(f"    Class {cls_idx}/{N_CLASSES} ({ALL_CLASSES[cls_idx]})...")

        cls_shap = []
        for i in range(0, n_test, batch_size):
            batch = test_t[i:i + batch_size]
            sv = explainer.shap_values(batch, nsamples=100)

            if isinstance(sv, list):
                cls_shap.append(sv[cls_idx])
            elif sv.ndim == 3:
                cls_shap.append(sv[:, :, cls_idx])
            else:
                cls_shap.append(sv)

        cls_shap = np.concatenate(cls_shap, axis=0)
        shap_values_list.append(cls_shap)

    shap_values = np.array(shap_values_list)

    if not use_dae and mt_indices is not None and len(mt_indices) > 0:
        shap_values[:, :, mt_indices] = 0.0
        print(f"  Zeroed out {len(mt_indices)} MT gene attributions")

    print(f"  SHAP values shape: {shap_values.shape}")
    return shap_values


def save_shap_results(shap_values: np.ndarray, test_labels_str: List[str],
                      gene_names: List[str], output_dir: str,
                      model_name: str = ""):
    """Save SHAP results: raw values, gene names, and label strings."""
    os.makedirs(output_dir, exist_ok=True)

    prefix = f"{model_name}_" if model_name else ""

    np.save(os.path.join(output_dir, f'{prefix}shap_values.npy'), shap_values)

    with open(os.path.join(output_dir, f'{prefix}shap_gene_names.txt'), 'w') as f:
        f.write('\n'.join(gene_names))

    with open(os.path.join(output_dir, f'{prefix}shap_test_labels.txt'), 'w') as f:
        f.write('\n'.join(test_labels_str))

    print(f"  Saved SHAP results to {output_dir}/{prefix}shap_values.npy")
    print(f"  Shape: {shap_values.shape}")
