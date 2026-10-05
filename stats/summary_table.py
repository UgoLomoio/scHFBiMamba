"""
Summary table combining DeLong and McNemar test results across all model pairs.

Produces:
  - Combined CSV with all statistical test results
  - Heatmap figure of -log10(p-value) for all pairs × categories
  - Summary table figure (publication-formatted)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional

from config import ABLATION_MODEL_PAIRS

FIGURE_DPI = 300


def create_summary_table(
    delong_df: pd.DataFrame,
    mcnemar_df: pd.DataFrame,
    output_dir: str,
) -> pd.DataFrame:
    """
    Merge DeLong and McNemar results into a single summary table.

    Args:
        delong_df: DeLong test results
        mcnemar_df: McNemar test results
        output_dir: where to save

    Returns:
        Combined DataFrame
    """
    rows = []

    # Build all pair × category combinations
    for m1, m2 in ABLATION_MODEL_PAIRS:
        for cat in ['species', 'celltype', 'disease']:
            row = {
                'model_1': m1,
                'model_2': m2,
                'category': cat,
            }

            # DeLong results
            if len(delong_df) > 0:
                d_match = delong_df[
                    (delong_df['model_1'] == m1) &
                    (delong_df['model_2'] == m2) &
                    (delong_df['category'] == cat)
                ]
                if len(d_match) > 0:
                    row['delong_auc_1'] = d_match.iloc[0]['auc_1']
                    row['delong_auc_2'] = d_match.iloc[0]['auc_2']
                    row['delong_auc_diff'] = d_match.iloc[0]['auc_diff']
                    row['delong_p_value'] = d_match.iloc[0]['p_value']
                    row['delong_p_adj'] = d_match.iloc[0].get('p_adj', d_match.iloc[0]['p_value'])

            # McNemar results
            if len(mcnemar_df) > 0:
                m_match = mcnemar_df[
                    (mcnemar_df['model_1'] == m1) &
                    (mcnemar_df['model_2'] == m2) &
                    (mcnemar_df['category'] == cat)
                ]
                if len(m_match) > 0:
                    row['mcnemar_acc_1'] = m_match.iloc[0]['accuracy_1']
                    row['mcnemar_acc_2'] = m_match.iloc[0]['accuracy_2']
                    row['mcnemar_acc_diff'] = m_match.iloc[0]['accuracy_diff']
                    row['mcnemar_p_value'] = m_match.iloc[0]['p_value']
                    row['mcnemar_p_adj'] = m_match.iloc[0].get('p_adj', m_match.iloc[0]['p_value'])

            rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(output_dir, 'stats_summary.csv'), index=False)
    print(f"  Saved summary to {output_dir}/stats_summary.csv")
    return summary


def plot_pvalue_heatmap(
    delong_df: pd.DataFrame,
    mcnemar_df: pd.DataFrame,
    output_dir: str,
):
    """
    Plot heatmap of -log10(p-value) for all model pairs × categories.

    Two heatmaps side by side: DeLong and McNemar.
    """
    os.makedirs(output_dir, exist_ok=True)

    categories = ['species', 'celltype', 'disease']
    pair_labels = [f"{m1} vs {m2}" for m1, m2 in ABLATION_MODEL_PAIRS]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax_idx, (df, title, p_col) in enumerate([
        (delong_df, "DeLong's Test (ROC AUC)", 'p_value'),
        (mcnemar_df, "McNemar's Test (Accuracy)", 'p_value'),
    ]):
        ax = axes[ax_idx]

        if len(df) == 0:
            ax.text(0.5, 0.5, f'No data for {title}', ha='center', va='center')
            ax.set_title(title)
            continue

        # Build matrix: rows = pairs, cols = categories
        matrix = np.full((len(ABLATION_MODEL_PAIRS), len(categories)), np.nan)

        for i, (m1, m2) in enumerate(ABLATION_MODEL_PAIRS):
            for j, cat in enumerate(categories):
                match = df[
                    (df['model_1'] == m1) &
                    (df['model_2'] == m2) &
                    (df['category'] == cat)
                ]
                if len(match) > 0:
                    p = match.iloc[0][p_col]
                    if p > 0:
                        matrix[i, j] = -np.log10(p)

        # Data-driven colorbar limits (min→max of the plotted values)
        if np.isnan(matrix).all():
            vmin, vmax = 0.0, 1.0
        else:
            vmin = float(np.nanmin(matrix))
            vmax = float(np.nanmax(matrix))
            if vmax <= vmin:
                vmax = vmin + 1e-6
        sns.heatmap(
            matrix, annot=True, fmt='.2f', cmap='YlOrRd',
            xticklabels=categories, yticklabels=pair_labels,
            ax=ax, linewidths=0.5, cbar_kws={'label': '-log10(p-value)'},
            vmin=vmin, vmax=vmax,
        )
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Category')
        ax.set_ylabel('Model Pair')

    plt.tight_layout()
    for fmt in ['svg', 'png']:
        path = os.path.join(output_dir, f'stats_pvalues_heatmap.{fmt}')
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved stats_pvalues_heatmap.svg/png")


def run_stats_summary(
    delong_df: pd.DataFrame,
    mcnemar_df: pd.DataFrame,
    output_dir: str,
):
    """
    Generate summary table and p-value heatmap.
    """
    print("\n=== Statistical Test Summary ===")
    os.makedirs(output_dir, exist_ok=True)

    summary = create_summary_table(delong_df, mcnemar_df, output_dir)
    plot_pvalue_heatmap(delong_df, mcnemar_df, output_dir)

    return summary
