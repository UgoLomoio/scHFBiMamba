"""
DeLong's test for comparing correlated ROC AUCs.

Implements the DeLong et al. (1988) algorithm for testing whether two
ROC curves (from models evaluated on the same test set) have significantly
different AUCs.

Reference: DeLong ER, DeLong DM, Clarke-Pearson DL.
  "Comparing the areas under two or more correlated receiver operating
  characteristic curves: a nonparametric approach." Biometrics. 1988.

The algorithm:
  1. Compute placement values (structural components) for each model
  2. Compute covariance matrix between AUC estimators
  3. Z-test: Z = (AUC1 - AUC2) / sqrt(var1 + var2 - 2*cov12)
  4. Two-sided p-value from standard normal
"""
import os
import json
import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, List, Tuple, Optional

from config import (
    SPECIES_IDX, CELLTYPE_IDX, DISEASE_IDX,
    ABLATION_MODELS, ABLATION_MODEL_PAIRS
)


def _compute_placement_values(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Compute placement values for the AUC estimator.

    For each sample, the placement value is:
      - For positive samples: (rank of score among all) / n_neg - 0.5/n_neg
        Simplified: fraction of negative samples with lower score
      - For negative samples: (rank of score among all) / n_pos - 0.5/n_pos
        Simplified: fraction of positive samples with higher score

    These are the structural components whose average gives the AUC.

    Args:
        scores: (N,) predicted scores for one class
        labels: (N,) binary labels (0 or 1)

    Returns:
        (N,) placement values
    """
    n = len(scores)
    n_pos = labels.sum()
    n_neg = n - n_pos

    if n_pos == 0 or n_neg == 0:
        return np.zeros(n)

    # Compute ranks (average rank for ties)
    order = np.argsort(scores, kind='mergesort')
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)

    # Handle ties: assign average rank
    # Find groups of tied scores
    sorted_scores = scores[order]
    i = 0
    while i < n:
        j = i
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        if j > i + 1:
            avg_rank = (ranks[order[i]] + ranks[order[j - 1]]) / 2
            for k in range(i, j):
                ranks[order[k]] = avg_rank
        i = j

    # Placement values
    pv = np.zeros(n)
    pos_mask = labels == 1
    neg_mask = labels == 0

    # For positive samples: (rank - n_pos) / n_neg = fraction of neg below
    pv[pos_mask] = (ranks[pos_mask] - n_pos) / n_neg
    # For negative samples: (n_neg + 1 - rank) / n_pos = fraction of pos above
    pv[neg_mask] = (n_neg + 1 - ranks[neg_mask]) / n_pos

    return pv


def compute_auc_and_variance(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """
    Compute AUC and its variance using the DeLong method.

    Args:
        scores: (N,) predicted scores
        labels: (N,) binary labels

    Returns:
        (auc, variance)
    """
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.5, 0.0

    pv = _compute_placement_values(scores, labels)

    # AUC = mean of placement values for positive samples
    auc = pv[labels == 1].mean()

    # Variance components
    # V10 = variance of placement values for positive samples
    # V01 = variance of placement values for negative samples
    v10 = pv[labels == 1].var(ddof=1) if n_pos > 1 else 0.0
    v01 = pv[labels == 0].var(ddof=1) if n_neg > 1 else 0.0

    # DeLong variance
    variance = v10 / n_pos + v01 / n_neg

    return float(auc), float(variance)


def compute_covariance(scores1: np.ndarray, scores2: np.ndarray,
                       labels: np.ndarray) -> float:
    """
    Compute covariance between two AUC estimators (same test set).

    Args:
        scores1: (N,) scores from model 1
        scores2: (N,) scores from model 2
        labels: (N,) binary labels

    Returns:
        covariance between AUC1 and AUC2
    """
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.0

    pv1 = _compute_placement_values(scores1, labels)
    pv2 = _compute_placement_values(scores2, labels)

    # Covariance components
    pos_mask = labels == 1
    neg_mask = labels == 0

    # Cov10: covariance of placement values for positive samples
    cov10 = np.cov(pv1[pos_mask], pv2[pos_mask], ddof=1)[0, 1] if n_pos > 1 else 0.0
    # Cov01: covariance of placement values for negative samples
    cov01 = np.cov(pv1[neg_mask], pv2[neg_mask], ddof=1)[0, 1] if n_neg > 1 else 0.0

    covariance = cov10 / n_pos + cov01 / n_neg
    return float(covariance)


def delong_test(scores1: np.ndarray, scores2: np.ndarray,
                labels: np.ndarray) -> Dict:
    """
    Perform DeLong's test comparing two ROC AUCs.

    Args:
        scores1: (N,) scores from model 1
        scores2: (N,) scores from model 2
        labels: (N,) binary labels

    Returns:
        dict with AUC1, AUC2, difference, Z-statistic, p-value
    """
    auc1, var1 = compute_auc_and_variance(scores1, labels)
    auc2, var2 = compute_auc_and_variance(scores2, labels)
    cov12 = compute_covariance(scores1, scores2, labels)

    diff = auc1 - auc2
    se = np.sqrt(var1 + var2 - 2 * cov12)

    if se > 0:
        z = diff / se
        p_value = 2 * norm.sf(np.abs(z))
    else:
        z = 0.0
        p_value = 1.0

    return {
        'auc1': float(auc1),
        'auc2': float(auc2),
        'auc_diff': float(diff),
        'se': float(se),
        'z_statistic': float(z),
        'p_value': float(p_value),
    }


def run_delong_for_all_pairs(
    ablation_dir: str,
    output_dir: str,
    model_ids: List[str] = None,
) -> pd.DataFrame:
    """
    Run DeLong's test for all model pairs across all categories.

    Args:
        ablation_dir: directory with model subdirectories containing test_probs.npy
        output_dir: where to save results
        model_ids: list of model IDs (default: all)

    Returns:
        DataFrame with DeLong test results
    """
    print("\n=== DeLong's Test ===")
    os.makedirs(output_dir, exist_ok=True)

    if model_ids is None:
        model_ids = list(ABLATION_MODELS.keys())

    categories = [
        ('species', SPECIES_IDX[0], SPECIES_IDX[1]),
        ('celltype', CELLTYPE_IDX[0], CELLTYPE_IDX[1]),
        ('disease', DISEASE_IDX[0], DISEASE_IDX[1]),
    ]

    # Load probabilities and labels for all models
    probs = {}
    labels = None
    for model_id in model_ids:
        probs_path = os.path.join(ablation_dir, model_id, 'test_probs.npy')
        if not os.path.exists(probs_path):
            print(f"  Warning: {probs_path} not found, skipping {model_id}")
            continue
        probs[model_id] = np.load(probs_path)

        if labels is None:
            labels_path = os.path.join(ablation_dir, model_id, 'test_labels.npy')
            if os.path.exists(labels_path):
                labels = np.load(labels_path)

    if labels is None or len(probs) < 2:
        print("  Not enough models or labels for DeLong test")
        return pd.DataFrame()

    # Run DeLong for each pair × category
    results = []
    available_models = sorted(probs.keys())

    for i, m1 in enumerate(available_models):
        for m2 in available_models[i + 1:]:
            for cat_name, cat_start, cat_end in categories:
                n_classes = cat_end - cat_start

                # Macro-average: average DeLong across one-vs-rest classes
                class_results = []
                for c in range(n_classes):
                    cls_idx = cat_start + c
                    binary_labels = labels[:, cls_idx]
                    if binary_labels.sum() == 0 or binary_labels.sum() == len(binary_labels):
                        continue

                    result = delong_test(
                        probs[m1][:, cls_idx],
                        probs[m2][:, cls_idx],
                        binary_labels
                    )
                    class_results.append(result)

                if not class_results:
                    continue

                # Average across classes
                avg_auc1 = np.mean([r['auc1'] for r in class_results])
                avg_auc2 = np.mean([r['auc2'] for r in class_results])
                # Fisher's method for combining p-values
                from scipy.stats import combine_pvalues
                p_vals = [r['p_value'] for r in class_results]
                _, combined_p = combine_pvalues(p_vals, method='fisher')

                results.append({
                    'model_1': m1,
                    'model_2': m2,
                    'category': cat_name,
                    'auc_1': avg_auc1,
                    'auc_2': avg_auc2,
                    'auc_diff': avg_auc1 - avg_auc2,
                    'p_value': combined_p,
                    'n_classes': len(class_results),
                })
                print(f"  {m1} vs {m2} ({cat_name}): "
                      f"AUC={avg_auc1:.4f} vs {avg_auc2:.4f}, p={combined_p:.4e}")

    df = pd.DataFrame(results)

    if len(df) > 0:
        # Multiple testing correction (BH-FDR)
        from statsmodels.stats.multitest import multipletests
        _, df['p_adj'], _, _ = multipletests(df['p_value'].values, method='fdr_bh')
        df.to_csv(os.path.join(output_dir, 'delong_results.csv'), index=False)
        print(f"  Saved to {output_dir}/delong_results.csv")

    return df
