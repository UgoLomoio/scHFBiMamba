"""
ROC and Precision-Recall curves for HF Mamba ablation models.

Computes per-category (species, celltype, disease) one-vs-rest ROC and PR
curves for all 4 models, with macro-averaged AUC and Average Precision.

Output:
  - 6 figures (3 categories × 2 curve types) with all 4 models overlaid
  - JSON with AUC and AP values per model per category
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from typing import Dict, List, Tuple

# NumPy 2.x removed np.trapz (renamed to np.trapezoid); keep 1.x compatibility
try:
    from numpy import trapezoid as _trapezoid
except ImportError:  # NumPy < 2.0
    from numpy import trapz as _trapezoid

from config import (
    SPECIES_IDX, CELLTYPE_IDX, DISEASE_IDX, ABLATION_MODELS
)

# Phylo color palette for models
MODEL_COLORS = {
    'M0': '#808080',  # gray
    'M1': '#0279EE',  # blue
    'M2': '#FF9400',  # orange
    'M3': '#75A025',  # green
    'M4': '#FD9BED',  # pink
}

FIGURE_DPI = 300


def compute_roc_pr_for_category(
    probs: np.ndarray,
    labels: np.ndarray,
    cat_start: int,
    cat_end: int,
) -> Dict:
    """
    Compute one-vs-rest ROC and PR curves for a category.

    Args:
        probs: (N, 13) sigmoid probabilities
        labels: (N, 13) multi-hot ground truth
        cat_start: start index of category
        cat_end: end index (exclusive)

    Returns:
        dict with macro-averaged ROC AUC, PR AP, and per-class curves
    """
    n_classes = cat_end - cat_start
    cat_probs = probs[:, cat_start:cat_end]
    cat_labels = labels[:, cat_start:cat_end]

    # Per-class ROC
    all_fpr = []
    all_tpr = []
    roc_aucs = []
    for c in range(n_classes):
        # Skip if only one class present
        if cat_labels[:, c].sum() == 0 or cat_labels[:, c].sum() == len(cat_labels):
            roc_aucs.append(0.5)
            all_fpr.append(np.array([0.0, 1.0]))
            all_tpr.append(np.array([0.0, 1.0]))
            continue
        fpr, tpr, _ = roc_curve(cat_labels[:, c], cat_probs[:, c])
        all_fpr.append(fpr)
        all_tpr.append(tpr)
        roc_aucs.append(auc(fpr, tpr))

    # Macro-average ROC
    # Interpolate all curves to common FPR grid
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.zeros_like(mean_fpr)
    for c in range(n_classes):
        mean_tpr += np.interp(mean_fpr, all_fpr[c], all_tpr[c])
    mean_tpr /= n_classes
    macro_roc_auc = auc(mean_fpr, mean_tpr)

    # Per-class PR
    all_precision = []
    all_recall = []
    aps = []
    for c in range(n_classes):
        # Skip if only one class present
        if cat_labels[:, c].sum() == 0 or cat_labels[:, c].sum() == len(cat_labels):
            aps.append(0.0)
            all_precision.append(np.array([1.0, 0.0]))
            all_recall.append(np.array([0.0, 1.0]))
            continue
        precision, recall, _ = precision_recall_curve(cat_labels[:, c], cat_probs[:, c])
        all_precision.append(precision)
        all_recall.append(recall)
        aps.append(average_precision_score(cat_labels[:, c], cat_probs[:, c]))

    # Macro-average PR
    mean_recall = np.linspace(0, 1, 100)
    mean_precision = np.zeros_like(mean_recall)
    for c in range(n_classes):
        # PR curves go from high recall to low; reverse for interpolation
        sorted_idx = np.argsort(all_recall[c])
        mean_precision += np.interp(
            mean_recall, all_recall[c][sorted_idx], all_precision[c][sorted_idx])
    mean_precision /= n_classes
    macro_ap = _trapezoid(mean_precision, mean_recall)

    return {
        'macro_roc_auc': float(macro_roc_auc),
        'macro_ap': float(macro_ap),
        'per_class_roc_auc': [float(a) for a in roc_aucs],
        'per_class_ap': [float(a) for a in aps],
        'macro_fpr': mean_fpr.tolist(),
        'macro_tpr': mean_tpr.tolist(),
        'macro_precision': mean_precision.tolist(),
        'macro_recall': mean_recall.tolist(),
    }


def compute_all_roc_pr(
    ablation_dir: str,
    model_ids: List[str] = None,
) -> Dict:
    """
    Compute ROC and PR curves for all models and all categories.

    Args:
        ablation_dir: directory with model subdirectories
        model_ids: list of model IDs (default: all)

    Returns:
        dict: {model_id: {category: {roc_auc, ap, curves}}}
    """
    if model_ids is None:
        model_ids = list(ABLATION_MODELS.keys())

    categories = [
        ('species', SPECIES_IDX[0], SPECIES_IDX[1]),
        ('celltype', CELLTYPE_IDX[0], CELLTYPE_IDX[1]),
        ('disease', DISEASE_IDX[0], DISEASE_IDX[1]),
    ]

    all_results = {}

    for model_id in model_ids:
        probs_path = os.path.join(ablation_dir, model_id, 'test_probs.npy')
        if not os.path.exists(probs_path):
            print(f"  Warning: {probs_path} not found, skipping {model_id}")
            continue

        probs = np.load(probs_path)

        # Load labels
        from data import load_labels
        # Try to find test labels in the model dir or ablation dir
        labels_path = os.path.join(ablation_dir, model_id, 'test_labels.npy')
        if os.path.exists(labels_path):
            labels = np.load(labels_path)
        else:
            # Load from data directory (fallback)
            print(f"  Warning: test labels not found at {labels_path}")
            print(f"  Please ensure test labels are available")
            continue

        all_results[model_id] = {}
        for cat_name, cat_start, cat_end in categories:
            result = compute_roc_pr_for_category(probs, labels, cat_start, cat_end)
            all_results[model_id][cat_name] = result
            print(f"  {model_id} {cat_name}: ROC AUC={result['macro_roc_auc']:.4f}, "
                  f"AP={result['macro_ap']:.4f}")

    return all_results


def plot_roc_curves(all_results: Dict, output_dir: str):
    """Plot ROC curves for all models, one figure per category."""
    os.makedirs(output_dir, exist_ok=True)
    categories = ['species', 'celltype', 'disease']

    for cat in categories:
        fig, ax = plt.subplots(figsize=(7, 6))

        for model_id in sorted(all_results.keys()):
            if cat not in all_results[model_id]:
                continue
            r = all_results[model_id][cat]
            ax.plot(r['macro_fpr'], r['macro_tpr'],
                    color=MODEL_COLORS.get(model_id, '#000000'),
                    lw=2, label=f'{model_id} (AUC={r["macro_roc_auc"]:.3f})')

        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curves: {cat.capitalize()}')
        ax.legend(loc='lower right', fontsize=9)

        for fmt in ['svg', 'png']:
            path = os.path.join(output_dir, f'roc_{cat}.{fmt}')
            fig.savefig(path, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved roc_{cat}.svg/png")


def plot_pr_curves(all_results: Dict, output_dir: str):
    """Plot PR curves for all models, one figure per category."""
    os.makedirs(output_dir, exist_ok=True)
    categories = ['species', 'celltype', 'disease']

    for cat in categories:
        fig, ax = plt.subplots(figsize=(7, 6))

        for model_id in sorted(all_results.keys()):
            if cat not in all_results[model_id]:
                continue
            r = all_results[model_id][cat]
            ax.plot(r['macro_recall'], r['macro_precision'],
                    color=MODEL_COLORS.get(model_id, '#000000'),
                    lw=2, label=f'{model_id} (AP={r["macro_ap"]:.3f})')

        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'Precision-Recall Curves: {cat.capitalize()}')
        ax.legend(loc='lower left', fontsize=9)

        for fmt in ['svg', 'png']:
            path = os.path.join(output_dir, f'pr_{cat}.{fmt}')
            fig.savefig(path, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved pr_{cat}.svg/png")


def run_roc_pr_analysis(ablation_dir: str, output_dir: str,
                        model_ids: List[str] = None):
    """
    Full ROC/PR analysis: compute curves, plot, and save results.
    """
    print("\n=== ROC / PR Curve Analysis ===")
    os.makedirs(output_dir, exist_ok=True)

    all_results = compute_all_roc_pr(ablation_dir, model_ids)

    # Save results JSON
    with open(os.path.join(output_dir, 'roc_pr_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    # Plot
    plot_roc_curves(all_results, output_dir)
    plot_pr_curves(all_results, output_dir)

    print(f"ROC/PR results saved to {output_dir}/")
    return all_results
