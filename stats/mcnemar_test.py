"""
McNemar's test for comparing classification accuracy between model pairs.

For each pair of models evaluated on the same test set, constructs a 2×2
contingency table:
  - Both correct
  - Model A correct, Model B wrong
  - Model A wrong, Model B correct
  - Both wrong

McNemar's test focuses on the discordant pairs (A correct/B wrong vs
A wrong/B correct) to test whether the models have significantly different
accuracy.

Uses statsmodels for the test implementation.
"""
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

from config import (
    ALL_CLASSES, SPECIES_IDX, CELLTYPE_IDX, DISEASE_IDX,
    ABLATION_MODELS, ABLATION_MODEL_PAIRS
)
from losses import onehot_predict


def compute_correctness_per_category(
    probs: np.ndarray,
    labels: np.ndarray,
    cat_start: int,
    cat_end: int,
) -> np.ndarray:
    """
    Compute per-cell correctness for a category.

    A cell is "correct" if the predicted class (argmax within category)
    matches the true class.

    Args:
        probs: (N, 13) sigmoid probabilities
        labels: (N, 13) multi-hot ground truth
        cat_start: start index of category
        cat_end: end index (exclusive)

    Returns:
        (N,) boolean array: True if correct
    """
    # Get predicted class (argmax within category)
    pred_idx = np.argmax(probs[:, cat_start:cat_end], axis=1)
    # Get true class
    true_idx = np.argmax(labels[:, cat_start:cat_end], axis=1)
    return pred_idx == true_idx


def mcnemar_test(correct_a: np.ndarray, correct_b: np.ndarray,
                 correction: bool = True) -> Dict:
    """
    Perform McNemar's test on two correctness arrays.

    Args:
        correct_a: (N,) boolean — whether model A predicted correctly
        correct_b: (N,) boolean — whether model B predicted correctly
        correction: whether to use continuity correction (default True)

    Returns:
        dict with statistic, p-value, and contingency table
    """
    from statsmodels.stats.contingency_tables import mcnemar

    # Build 2×2 contingency table
    # [[both_correct, a_correct_b_wrong],
    #  [a_wrong_b_correct, both_wrong]]
    both_correct = (correct_a & correct_b).sum()
    a_correct_b_wrong = (correct_a & ~correct_b).sum()
    a_wrong_b_correct = (~correct_a & correct_b).sum()
    both_wrong = (~correct_a & ~correct_b).sum()

    table = [[int(both_correct), int(a_correct_b_wrong)],
             [int(a_wrong_b_correct), int(both_wrong)]]

    # McNemar's test
    result = mcnemar(table, exact=False, correction=correction)

    return {
        'statistic': float(result.statistic),
        'p_value': float(result.pvalue),
        'contingency_table': table,
        'accuracy_a': float(correct_a.mean()),
        'accuracy_b': float(correct_b.mean()),
        'accuracy_diff': float(correct_a.mean() - correct_b.mean()),
        'n_discordant': int(a_correct_b_wrong + a_wrong_b_correct),
    }


def run_mcnemar_for_all_pairs(
    ablation_dir: str,
    output_dir: str,
    model_ids: List[str] = None,
    correction: bool = True,
) -> pd.DataFrame:
    """
    Run McNemar's test for all model pairs across all categories.

    Args:
        ablation_dir: directory with model subdirectories
        output_dir: where to save results
        model_ids: list of model IDs (default: all)
        correction: use continuity correction

    Returns:
        DataFrame with McNemar test results
    """
    print("\n=== McNemar's Test ===")
    os.makedirs(output_dir, exist_ok=True)

    if model_ids is None:
        model_ids = list(ABLATION_MODELS.keys())

    categories = [
        ('species', SPECIES_IDX[0], SPECIES_IDX[1]),
        ('celltype', CELLTYPE_IDX[0], CELLTYPE_IDX[1]),
        ('disease', DISEASE_IDX[0], DISEASE_IDX[1]),
    ]

    # Load probabilities and labels
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
        print("  Not enough models or labels for McNemar test")
        return pd.DataFrame()

    # Compute correctness per model per category
    correctness = {}
    for model_id, model_probs in probs.items():
        correctness[model_id] = {}
        for cat_name, cat_start, cat_end in categories:
            correctness[model_id][cat_name] = compute_correctness_per_category(
                model_probs, labels, cat_start, cat_end)

    # Run McNemar for each pair × category
    results = []
    available_models = sorted(probs.keys())

    for i, m1 in enumerate(available_models):
        for m2 in available_models[i + 1:]:
            for cat_name, _, _ in categories:
                result = mcnemar_test(
                    correctness[m1][cat_name],
                    correctness[m2][cat_name],
                    correction=correction
                )

                results.append({
                    'model_1': m1,
                    'model_2': m2,
                    'category': cat_name,
                    'accuracy_1': result['accuracy_a'],
                    'accuracy_2': result['accuracy_b'],
                    'accuracy_diff': result['accuracy_diff'],
                    'chi2': result['statistic'],
                    'p_value': result['p_value'],
                    'n_discordant': result['n_discordant'],
                })
                print(f"  {m1} vs {m2} ({cat_name}): "
                      f"Acc={result['accuracy_a']:.4f} vs {result['accuracy_b']:.4f}, "
                      f"p={result['p_value']:.4e}")

    df = pd.DataFrame(results)

    if len(df) > 0:
        # Multiple testing correction (BH-FDR)
        from statsmodels.stats.multitest import multipletests
        _, df['p_adj'], _, _ = multipletests(df['p_value'].values, method='fdr_bh')
        df.to_csv(os.path.join(output_dir, 'mcnemar_results.csv'), index=False)
        print(f"  Saved to {output_dir}/mcnemar_results.csv")

    return df
