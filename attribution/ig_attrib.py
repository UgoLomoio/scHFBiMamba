"""
Integrated Gradients attribution for HF Mamba classifiers.

Uses Captum's IntegratedGradients to compute path-attribution scores from
a baseline (zero expression) to the actual expression for each gene and
each of the 13 output classes.

All models use the composite model wrapper (encoder + classifier) so that
attributions are computed at the GENE level (16,545 genes) for all models.

Integrated Gradients satisfies the completeness axiom:
  sum(attributions) = f(x) - f(baseline)
"""
import os
import numpy as np
import torch
from typing import Optional, List, Dict

from config import N_GENES, N_CLASSES, ALL_CLASSES, MT_GENE_PREFIX


class SingleOutputWrapper(torch.nn.Module):
    """
    Wrapper to extract a single output class from a model.
    Captum's IntegratedGradients requires a scalar or (B,) output.
    """

    def __init__(self, model: torch.nn.Module, class_idx: int):
        super().__init__()
        self.model = model
        self.class_idx = class_idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the output for a single class. (B, L) -> (B,)"""
        out = self.model(x)  # (B, 13)
        return out[:, self.class_idx]


def compute_ig_attributions_composite(
    composite_model: torch.nn.Module,
    test_features: np.ndarray,
    device: torch.device,
    n_steps: int = 200,
    baseline: str = "zero",
    batch_size: int = 50,
    mt_indices: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute Integrated Gradients attribution scores at the GENE level for
    all 13 classes, using the composite model (encoder + classifier).

    Args:
        composite_model: CompositeModel (encoder + classifier)
        test_features: (N_test, N_genes) test set expression values
        device: torch device
        n_steps: number of interpolation steps (default 200)
        baseline: "zero" (zero expression) or "mean" (mean expression)
        batch_size: batch size for IG computation
        mt_indices: indices of MT genes to zero out

    Returns:
        ig_values: (N_classes, N_test, N_genes) IG attribution scores
    """
    from captum.attr import IntegratedGradients

    composite_model.eval()

    test_input = np.asarray(test_features, dtype=np.float32)
    n_test = len(test_input)
    seq_len = test_input.shape[1]
    print(f"  IG test input: {test_input.shape}")
    print(f"  Steps: {n_steps}, Baseline: {baseline}")

    # Prepare baseline
    if baseline == "zero":
        baseline_data = np.zeros((1, seq_len), dtype=np.float32)
    elif baseline == "mean":
        baseline_data = test_input.mean(axis=0, keepdims=True)
    else:
        baseline_data = np.zeros((1, seq_len), dtype=np.float32)

    baseline_t = torch.from_numpy(baseline_data).to(device)

    # Compute IG for each class
    ig_values_list = []

    for cls_idx in range(N_CLASSES):
        print(f"    Class {cls_idx}/{N_CLASSES} ({ALL_CLASSES[cls_idx]})...")

        wrapper = SingleOutputWrapper(composite_model, cls_idx).to(device)
        ig = IntegratedGradients(wrapper)

        cls_ig = []
        for i in range(0, n_test, batch_size):
            batch = torch.from_numpy(
                test_input[i:i + batch_size]
            ).float().to(device)

            attributions = ig.attribute(
                batch,
                baselines=baseline_t.expand(len(batch), -1),
                n_steps=n_steps,
                method='gausslegendre',
            )

            cls_ig.append(attributions.cpu().numpy())

        cls_ig = np.concatenate(cls_ig, axis=0)  # (N_test, N_genes)
        ig_values_list.append(cls_ig)

    ig_values = np.array(ig_values_list)  # (N_classes, N_test, N_genes)

    # Zero out MT gene attributions (per paper)
    if mt_indices is not None and len(mt_indices) > 0:
        ig_values[:, :, mt_indices] = 0.0
        print(f"  Zeroed out {len(mt_indices)} MT gene attributions")

    print(f"  IG values shape: {ig_values.shape}")
    return ig_values


def compute_ig_attributions(
    model: torch.nn.Module,
    test_features: np.ndarray,
    device: torch.device,
    n_steps: int = 200,
    baseline: str = "zero",
    batch_size: int = 50,
    use_dae: bool = False,
    dae_model: Optional[torch.nn.Module] = None,
    mt_indices: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Legacy IG attribution (for standalone classifier without composite model).
    Kept for backward compatibility.

    If use_dae=True, attributes to DAE latent dims (not genes).
    For gene-level attribution with encoders, use compute_ig_attributions_composite.
    """
    from captum.attr import IntegratedGradients

    model.eval()

    if use_dae and dae_model is not None:
        dae_model.eval()
        with torch.no_grad():
            test_t = torch.from_numpy(
                np.asarray(test_features, dtype=np.float32)
            ).to(device)
            test_input = dae_model.get_latent(test_t).cpu().numpy()
    else:
        test_input = np.asarray(test_features, dtype=np.float32)

    n_test = len(test_input)
    seq_len = test_input.shape[1]

    if baseline == "zero":
        baseline_data = np.zeros((1, seq_len), dtype=np.float32)
    elif baseline == "mean":
        baseline_data = test_input.mean(axis=0, keepdims=True)
    else:
        baseline_data = np.zeros((1, seq_len), dtype=np.float32)

    baseline_t = torch.from_numpy(baseline_data).to(device)

    ig_values_list = []

    for cls_idx in range(N_CLASSES):
        print(f"    Class {cls_idx}/{N_CLASSES} ({ALL_CLASSES[cls_idx]})...")

        wrapper = SingleOutputWrapper(model, cls_idx).to(device)
        ig = IntegratedGradients(wrapper)

        cls_ig = []
        for i in range(0, n_test, batch_size):
            batch = torch.from_numpy(
                test_input[i:i + batch_size]
            ).float().to(device)

            attributions = ig.attribute(
                batch,
                baselines=baseline_t.expand(len(batch), -1),
                n_steps=n_steps,
                method='gausslegendre',
            )

            cls_ig.append(attributions.cpu().numpy())

        cls_ig = np.concatenate(cls_ig, axis=0)
        ig_values_list.append(cls_ig)

    ig_values = np.array(ig_values_list)

    if not use_dae and mt_indices is not None and len(mt_indices) > 0:
        ig_values[:, :, mt_indices] = 0.0
        print(f"  Zeroed out {len(mt_indices)} MT gene attributions")

    print(f"  IG values shape: {ig_values.shape}")
    return ig_values


def save_ig_results(ig_values: np.ndarray, test_labels_str: List[str],
                    gene_names: List[str], output_dir: str,
                    model_name: str = ""):
    """Save IG results."""
    os.makedirs(output_dir, exist_ok=True)

    prefix = f"{model_name}_" if model_name else ""

    np.save(os.path.join(output_dir, f'{prefix}ig_values.npy'), ig_values)

    with open(os.path.join(output_dir, f'{prefix}ig_gene_names.txt'), 'w') as f:
        f.write('\n'.join(gene_names))

    with open(os.path.join(output_dir, f'{prefix}ig_test_labels.txt'), 'w') as f:
        f.write('\n'.join(test_labels_str))

    print(f"  Saved IG results to {output_dir}/{prefix}ig_values.npy")
    print(f"  Shape: {ig_values.shape}")
