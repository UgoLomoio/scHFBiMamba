"""
Figure generation for HF Mamba publication.
All figures saved as SVG (user preference) + PNG.

Generates figures matching the original paper's structure:
  F1-F13: Paper equivalents (workflow, training curves, beeswarm, DXG/GSEA)
  F14-F16: New ablation figures
  F17: Koenig validation
  F18: ROC/PR curves
  F19: Stats p-value heatmap
  F20-F23: UMAP projections
  F24-F36: Per-label explainability
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from typing import Dict, List, Optional

from config import (
    ALL_CLASSES, SPECIES_CLASSES, CELLTYPE_CLASSES, DISEASE_CLASSES,
    SPECIES_IDX, CELLTYPE_IDX, DISEASE_IDX, ABLATION_MODELS
)


# ─── Style ───────────────────────────────────────────────────────────────────
FIGURE_DPI = 300
FIGURE_FORMAT = ['svg', 'png']

# Phylo color palette
PHYLO_COLORS = ['#000000', '#ECE9E2', '#FAF9F3', '#E9ED4C', '#FF9400',
                '#75A025', '#FD9BED', '#0279EE']

# Colorblind-friendly palette for model comparison
MODEL_COLORS = {
    'M1': '#0279EE',  # blue
    'M2': '#FF9400',  # orange
    'M3': '#75A025',  # green
    'M4': '#FD9BED',  # pink
}

# Set font: keep only families actually installed (avoids findfont warnings,
# e.g. "Font family 'Arimo' not found" on systems without the MS core fonts).
from matplotlib import font_manager as _font_manager
_installed_fonts = {f.name for f in _font_manager.fontManager.ttflist}
_preferred_fonts = ['Liberation Sans', 'Arimo', 'DejaVu Sans']
_available_fonts = [f for f in _preferred_fonts if f in _installed_fonts]
matplotlib.rcParams['font.family'] = _available_fonts or ['DejaVu Sans']
matplotlib.rcParams['svg.fonttype'] = 'none'  # keep SVG text editable


def save_figure(fig, output_dir: str, name: str):
    """Save figure in all configured formats."""
    os.makedirs(output_dir, exist_ok=True)
    for fmt in FIGURE_FORMAT:
        path = os.path.join(output_dir, f"{name}.{fmt}")
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {name}.{FIGURE_FORMAT}")


# ─── F1: Workflow schematic ──────────────────────────────────────────────────

def fig_workflow_schematic(output_dir: str):
    """F1: Workflow schematic (paper Fig 1a equivalent)."""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis('off')

    boxes = [
        (0.5, 1.5, 2, 1, 'snRNA-seq\n(Human + Mouse\nHF + Healthy)'),
        (3.5, 1.5, 2, 1, 'Preprocessing\n& Normalization\n(16,545 genes)'),
        (6.5, 1.5, 2, 1, 'Encoder\n(Dense DAE or\nBiMamba AE\n→350 latent)'),
        (9.5, 1.5, 2, 1, 'Classifier\n(BiMamba or MLP\n→ 13 outputs)'),
        (12.5, 2.8, 1.2, 0.8, 'SHAP\nAttribution'),
        (12.5, 1.2, 1.2, 0.8, 'IG\nAttribution'),
    ]

    for i, (x, y, w, h, label) in enumerate(boxes):
        color = PHYLO_COLORS[i % len(PHYLO_COLORS)]
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor='black', alpha=0.7)
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, label, ha='center', va='center',
                fontsize=8, fontweight='bold')

    for x_start in [2.5, 5.5, 8.5]:
        ax.annotate('', xy=(x_start + 1, 2), xytext=(x_start, 2),
                     arrowprops=dict(arrowstyle='->', lw=1.5))
    ax.annotate('', xy=(12.5, 3.2), xytext=(11.5, 2.3),
                 arrowprops=dict(arrowstyle='->', lw=1.5))
    ax.annotate('', xy=(12.5, 1.6), xytext=(11.5, 1.7),
                 arrowprops=dict(arrowstyle='->', lw=1.5))

    box = FancyBboxPatch((12.5, 0.1), 1.2, 0.6, boxstyle="round,pad=0.1",
                         facecolor=PHYLO_COLORS[4], edgecolor='black', alpha=0.7)
    ax.add_patch(box)
    ax.text(13.1, 0.4, 'DXG + GSEA', ha='center', va='center',
            fontsize=7, fontweight='bold')
    ax.annotate('', xy=(13.1, 0.7), xytext=(13.1, 1.1),
                 arrowprops=dict(arrowstyle='->', lw=1.5))

    ax.set_title('Workflow: HF Subtyping with Bidirectional Mamba Networks',
                 fontsize=12, fontweight='bold')
    save_figure(fig, output_dir, 'F01_workflow_schematic')


# ─── F3: Training curves ─────────────────────────────────────────────────────

def fig_training_curves(history_path: str, output_dir: str, model_name: str = ""):
    """F3: F1-score and precision/recall per epoch (paper Fig 2c-d)."""
    import pickle
    with open(history_path, 'rb') as f:
        history = pickle.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    epochs = history['epoch']

    # F1-score
    ax = axes[0]
    if 'val_f1_mean' in history:
        ax.plot(epochs, history['val_f1_mean'], label='Val F1 (mean)', color=PHYLO_COLORS[7])
    if 'val_f1_species' in history:
        ax.plot(epochs, history['val_f1_species'], label='Val F1 (species)', color=PHYLO_COLORS[4])
    if 'val_f1_celltype' in history:
        ax.plot(epochs, history['val_f1_celltype'], label='Val F1 (celltype)', color=PHYLO_COLORS[5])
    if 'val_f1_disease' in history:
        ax.plot(epochs, history['val_f1_disease'], label='Val F1 (disease)', color=PHYLO_COLORS[6])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1-Score')
    ax.set_title(f'{model_name}: F1-Score')
    ax.legend(fontsize=7)
    ax.set_ylim([0.5, 1.01])

    # Precision/Recall
    ax = axes[1]
    if 'val_precision_mean' in history:
        ax.plot(epochs, history['val_precision_mean'], label='Val Precision', color=PHYLO_COLORS[4])
    if 'val_recall_mean' in history:
        ax.plot(epochs, history['val_recall_mean'], label='Val Recall', color=PHYLO_COLORS[7])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.set_title(f'{model_name}: Precision & Recall')
    ax.legend(fontsize=7)
    ax.set_ylim([0.5, 1.01])

    # Loss
    ax = axes[2]
    ax.plot(epochs, history['loss'], label='Train Loss', color=PHYLO_COLORS[0])
    ax.plot(epochs, history['val_loss'], label='Val Loss', color=PHYLO_COLORS[4])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Macro F1-Loss')
    ax.set_title(f'{model_name}: Training Loss')
    ax.legend(fontsize=7)

    plt.tight_layout()
    save_figure(fig, output_dir, f'F03_training_curves_{model_name}')


# ─── F4: Classification rates comparison ─────────────────────────────────────

def fig_classification_rates(ablation_dir: str, output_dir: str):
    """F4: Correct classification rates across models (paper Fig 2e)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    categories = [
        ('species', SPECIES_CLASSES, 'Species'),
        ('celltype', CELLTYPE_CLASSES, 'Cell Type'),
        ('disease', DISEASE_CLASSES, 'Disease'),
    ]

    for ax_idx, (cat, classes, title) in enumerate(categories):
        ax = axes[ax_idx]
        width = 0.2
        x = np.arange(len(classes))

        for i, model_id in enumerate(['M1', 'M2', 'M3', 'M4']):
            metrics_path = os.path.join(ablation_dir, model_id, 'test_metrics.json')
            if not os.path.exists(metrics_path):
                continue
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)

            f1_per = metrics.get('f1_per_class', [])
            if cat == 'species':
                vals = f1_per[SPECIES_IDX[0]:SPECIES_IDX[1]]
            elif cat == 'celltype':
                vals = f1_per[CELLTYPE_IDX[0]:CELLTYPE_IDX[1]]
            else:
                vals = f1_per[DISEASE_IDX[0]:DISEASE_IDX[1]]

            ax.bar(x + i * width, vals, width, label=model_id,
                   color=MODEL_COLORS[model_id], alpha=0.8)

        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('F1-Score')
        ax.set_title(title)
        ax.set_ylim([0, 1.05])
        ax.legend(fontsize=7)

    plt.suptitle('Classification Performance: BiMamba Ablation', fontweight='bold')
    plt.tight_layout()
    save_figure(fig, output_dir, 'F04_classification_rates')


# ─── F6: Beeswarm plot ───────────────────────────────────────────────────────

def fig_beeswarm(shap_values: np.ndarray, test_expression: np.ndarray,
                 gene_names: List[str], class_idx: int,
                 class_name: str, output_dir: str, top_n: int = 20):
    """
    F6: Beeswarm plot of top predictors (paper Fig 3b,e,h).
    Fixed version: proper y-axis label placement.
    """
    # Mean absolute SHAP per gene for this class
    mean_abs_shap = np.mean(np.abs(shap_values[class_idx]), axis=0)
    top_indices = np.argsort(mean_abs_shap)[-top_n:][::-1]

    fig, ax = plt.subplots(figsize=(7, 8))

    for rank, g_idx in enumerate(top_indices):
        shap_vals = shap_values[class_idx, :, g_idx]
        expr_vals = test_expression[:, g_idx]

        # Jitter y positions
        y_jitter = np.random.normal(rank, 0.12, size=len(shap_vals))

        # Color by expression (data-driven colorbar limits)
        expr_min = float(np.min(expr_vals))
        expr_max = float(np.max(expr_vals))
        if expr_max <= expr_min:
            expr_max = expr_min + 1e-6
        scatter = ax.scatter(shap_vals, y_jitter, c=expr_vals, cmap='coolwarm',
                            s=12, alpha=0.6, edgecolors='none',
                            vmin=expr_min, vmax=expr_max)

    # Fixed: use set_yticklabels directly
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([gene_names[i] for i in top_indices], fontsize=8)
    ax.set_xlabel('SHAP value')
    ax.set_title(f'Top {top_n} predictors: {class_name}', fontweight='bold')
    plt.colorbar(scatter, ax=ax, label='Expression', shrink=0.5)

    plt.tight_layout()
    save_figure(fig, output_dir, f'F06_beeswarm_{class_name}')


# ─── F10: DXG vs DEG Venn diagram ────────────────────────────────────────────

def fig_dxg_deg_venn(dxg_genes: set, deg_genes: set, comparison_name: str,
                     output_dir: str):
    """F10: Venn diagram of DXG vs DEG overlap (paper Fig 4b)."""
    from matplotlib_venn import venn2

    fig, ax = plt.subplots(figsize=(6, 5))
    venn2([dxg_genes, deg_genes], set_labels=['DXG', 'DEG'],
          ax=ax, set_colors=(PHYLO_COLORS[7], PHYLO_COLORS[4]))
    ax.set_title(f'DXG vs DEG overlap: {comparison_name}', fontweight='bold')
    save_figure(fig, output_dir, f'F10_venn_{comparison_name}')


# ─── F14: Ablation heatmap ───────────────────────────────────────────────────

def fig_ablation_heatmap(ablation_dir: str, output_dir: str):
    """F14: Ablation heatmap: 4 models × metrics."""
    metrics_names = ['f1_mean', 'f1_species', 'f1_celltype', 'f1_disease',
                     'precision_mean', 'recall_mean']
    model_ids = ['M1', 'M2', 'M3', 'M4']

    data_matrix = np.zeros((len(model_ids), len(metrics_names)))

    for i, model_id in enumerate(model_ids):
        metrics_path = os.path.join(ablation_dir, model_id, 'test_metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            for j, mn in enumerate(metrics_names):
                data_matrix[i, j] = metrics.get(mn, 0.0)

    fig, ax = plt.subplots(figsize=(10, 4))
    # Data-driven colorbar limits (min→max of the plotted values)
    vmin = float(np.nanmin(data_matrix))
    vmax = float(np.nanmax(data_matrix))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    sns.heatmap(data_matrix, annot=True, fmt='.4f', cmap='YlOrRd',
                xticklabels=metrics_names, yticklabels=model_ids,
                ax=ax, vmin=vmin, vmax=vmax, linewidths=0.5)
    ax.set_title('Ablation Matrix: Model × Metrics', fontweight='bold')
    plt.tight_layout()
    save_figure(fig, output_dir, 'F14_ablation_heatmap')


# ─── F15: SHAP vs IG comparison ──────────────────────────────────────────────

def fig_shap_vs_ig(shap_values: np.ndarray, ig_values: np.ndarray,
                   gene_names: List[str], class_idx: int, class_name: str,
                   output_dir: str, top_n: int = 20):
    """F15: SHAP vs IG attribution comparison."""
    mean_shap = np.mean(np.abs(shap_values[class_idx]), axis=0)
    mean_ig = np.mean(np.abs(ig_values[class_idx]), axis=0)

    top_shap = set(np.argsort(mean_shap)[-top_n:])
    top_ig = set(np.argsort(mean_ig)[-top_n:])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter plot
    ax = axes[0]
    ax.scatter(mean_shap, mean_ig, s=5, alpha=0.3, color=PHYLO_COLORS[7])
    for g_idx in list(top_shap | top_ig)[:10]:
        ax.annotate(gene_names[g_idx], (mean_shap[g_idx], mean_ig[g_idx]),
                    fontsize=6, alpha=0.7)
    ax.set_xlabel('Mean |SHAP|')
    ax.set_ylabel('Mean |IG|')
    ax.set_title(f'Attribution correlation: {class_name}')

    # Venn diagram
    ax = axes[1]
    from matplotlib_venn import venn2
    venn2([top_shap, top_ig], set_labels=['SHAP top-20', 'IG top-20'],
          ax=ax, set_colors=(PHYLO_COLORS[7], PHYLO_COLORS[4]))
    ax.set_title(f'Top-{top_n} gene overlap: {class_name}')

    plt.tight_layout()
    save_figure(fig, output_dir, f'F15_shap_vs_ig_{class_name}')


# ─── F13: GSEA F1 across cell types ──────────────────────────────────────────

def fig_gsea_f1_across_celltypes(gsea_f1_path: str, output_dir: str):
    """F13: F1-scores for GSEA across cell types (paper Fig 4e-f)."""
    if not os.path.exists(gsea_f1_path):
        print(f"  GSEA F1 file not found: {gsea_f1_path}")
        print("  (Expected when no attribution files were available — DXG/GSEA was "
              "skipped. Run run_ablation.py --skip_training, then rerun "
              "run_analysis.py.)")
        return

    with open(gsea_f1_path, 'r') as f:
        all_f1 = json.load(f)

    # Aggregate by cell type and method
    rows = []
    for model_method, comps in all_f1.items():
        for comp_name, methods in comps.items():
            parts = comp_name.split('_')
            cell_type = parts[1] if len(parts) > 1 else ''
            disease = parts[2] if len(parts) > 2 else ''

            for method, f1_data in methods.items():
                rows.append({
                    'model_method': model_method,
                    'cell_type': cell_type,
                    'disease': disease,
                    'method': method,
                    'f1': f1_data['f1'],
                })

    df = pd.DataFrame(rows)
    if df.empty:
        print("  No GSEA F1 data to plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, disease in enumerate(['HFpEF', 'HFrEF']):
        ax = axes[ax_idx]
        # FIXED: was df[df['disease == disease'], now correct
        sub = df[df['disease'] == disease]

        if sub.empty:
            ax.text(0.5, 0.5, f'No data for {disease}', ha='center')
            continue

        pivot = sub.pivot_table(values='f1', index='cell_type',
                                columns='model_method', aggfunc='mean')
        pivot.plot(kind='bar', ax=ax, width=0.8)
        ax.set_title(f'GSEA F1: {disease}')
        ax.set_ylabel('F1-Score')
        ax.set_xlabel('Cell Type')
        ax.legend(fontsize=6, title='Model/Method')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    save_figure(fig, output_dir, 'F13_gsea_f1_celltypes')


# ─── Confusion matrix ────────────────────────────────────────────────────────

def fig_confusion_matrix(cm: np.ndarray, class_names: List[str],
                         title: str, output_dir: str, name: str):
    """Plot a confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, linewidths=0.5)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    save_figure(fig, output_dir, name)


# ─── F18: ROC/PR curves ──────────────────────────────────────────────────────

def fig_roc_pr_curves(ablation_dir: str, output_dir: str):
    """F18: ROC and PR curves for all models across categories."""
    from stats.roc_pr import run_roc_pr_analysis
    run_roc_pr_analysis(ablation_dir, output_dir)


# ─── F19: Stats p-value heatmap ──────────────────────────────────────────────

def fig_stats_pvalues(ablation_dir: str, output_dir: str):
    """F19: DeLong and McNemar p-value heatmap."""
    from stats.delong_test import run_delong_for_all_pairs
    from stats.mcnemar_test import run_mcnemar_for_all_pairs
    from stats.summary_table import run_stats_summary

    delong_df = run_delong_for_all_pairs(ablation_dir, output_dir)
    mcnemar_df = run_mcnemar_for_all_pairs(ablation_dir, output_dir)
    run_stats_summary(delong_df, mcnemar_df, output_dir)


# ─── Generate all figures ────────────────────────────────────────────────────

def generate_all_figures(ablation_dir: str, analysis_dir: str,
                         output_dir: str, go_keywords: dict = None,
                         test_features=None, test_labels_str=None,
                         gene_names=None, test_labels_path: str = ""):
    """
    Generate all publication figures.
    Called by run_analysis.py after all analyses are complete.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("Generating figures...")

    # F1: Workflow schematic
    try:
        fig_workflow_schematic(output_dir)
    except Exception as e:
        print(f"  F1 failed: {e}")

    # F3: Training curves for each model
    for model_id in ABLATION_MODELS:
        history_path = os.path.join(ablation_dir, model_id, 'classifier_history.pkl')
        if os.path.exists(history_path):
            try:
                fig_training_curves(history_path, output_dir, model_id)
            except Exception as e:
                print(f"  F3 ({model_id}) failed: {e}")

    # F4: Classification rates
    try:
        fig_classification_rates(ablation_dir, output_dir)
    except Exception as e:
        print(f"  F4 failed: {e}")

    # F14: Ablation heatmap
    try:
        fig_ablation_heatmap(ablation_dir, output_dir)
    except Exception as e:
        print(f"  F14 failed: {e}")

    # F13: GSEA F1 across cell types
    gsea_f1_path = os.path.join(analysis_dir, 'all_gsea_f1.json')
    try:
        fig_gsea_f1_across_celltypes(gsea_f1_path, output_dir)
    except Exception as e:
        print(f"  F13 failed: {e}")

    # F18: ROC/PR curves
    try:
        fig_roc_pr_curves(ablation_dir, os.path.join(output_dir, 'roc_pr'))
    except Exception as e:
        print(f"  F18 (ROC/PR) failed: {e}")

    # F19: Stats p-value heatmap
    try:
        fig_stats_pvalues(ablation_dir, os.path.join(output_dir, 'stats'))
    except Exception as e:
        print(f"  F19 (stats) failed: {e}")

    # Confusion matrices for each model
    for model_id in ABLATION_MODELS:
        eval_dir = os.path.join(ablation_dir, model_id, 'eval')
        for cat, classes, name in [
            ('species', SPECIES_CLASSES, 'species'),
            ('celltype', CELLTYPE_CLASSES, 'celltype'),
            ('disease', DISEASE_CLASSES, 'disease'),
        ]:
            cm_path = os.path.join(eval_dir, f'cm_{cat}.npy')
            if os.path.exists(cm_path):
                try:
                    cm = np.load(cm_path)
                    fig_confusion_matrix(
                        cm, classes,
                        f'{model_id}: {cat.capitalize()} Confusion Matrix',
                        output_dir, f'confusion_{model_id}_{name}')
                except Exception as e:
                    print(f"  CM ({model_id}/{cat}) failed: {e}")

    # F6: Beeswarm plots for each model (using SHAP values)
    for model_id in ABLATION_MODELS:
        shap_path = os.path.join(ablation_dir, model_id, 'attribution',
                                 f'{model_id}_shap_values.npy')
        if os.path.exists(shap_path) and test_features is not None:
            try:
                shap_values = np.load(shap_path)
                # Load attribution labels
                labels_path = os.path.join(ablation_dir, model_id, 'attribution',
                                           f'{model_id}_shap_test_labels.txt')
                with open(labels_path, 'r') as f:
                    attrib_labels = f.read().strip().split('\n')

                # Get expression for the balanced subset
                from data import sample_balanced_test_subset
                subset_features, _, _ = sample_balanced_test_subset(
                    test_features, test_labels_str, n_per_celltype=200, seed=1111)

                # Plot beeswarm for a few key classes
                for cls_idx, cls_name in enumerate(ALL_CLASSES):
                    fig_beeswarm(
                        shap_values, np.asarray(subset_features, dtype=np.float32),
                        gene_names, cls_idx, cls_name,
                        os.path.join(output_dir, 'beeswarm'),
                        top_n=20)
            except Exception as e:
                print(f"  F6 ({model_id}) failed: {e}")

    # F10: Venn diagrams (DXG vs DEG)
    # DXG results live in analysis_dir/<model>_<method>/dxg/ with
    # filenames prefixed by <model>_<method>_ (see run_analysis.py / dxg.py).
    deg_dir = os.path.join(analysis_dir, 'deg')
    if os.path.isdir(deg_dir):
        try:
            from dxg import get_all_comparisons
            comps = get_all_comparisons()
            # Use the first model/method combo that produced DXG results
            dxg_candidates = [
                d for d in sorted(os.listdir(analysis_dir))
                if os.path.isdir(os.path.join(analysis_dir, d, 'dxg'))
            ]
            if not dxg_candidates:
                raise FileNotFoundError("no DXG result directories found")
            combo = dxg_candidates[0]
            dxg_dir = os.path.join(analysis_dir, combo, 'dxg')
            for comp in comps[:3]:  # First 3 comparisons as examples
                name = comp['name']
                dxg_file = os.path.join(dxg_dir, f"{combo}_dxg_{name}.csv")
                deg_file = os.path.join(deg_dir, f"deg_{name}.csv")
                if os.path.exists(dxg_file) and os.path.exists(deg_file):
                    dxg_df = pd.read_csv(dxg_file)
                    deg_df = pd.read_csv(deg_file)
                    fig_dxg_deg_venn(
                        set(dxg_df['Feature']), set(deg_df['Feature']),
                        name, os.path.join(output_dir, 'venn'))
        except Exception as e:
            print(f"  F10 (Venn) failed: {e}")

    # F15: SHAP vs IG comparison
    for model_id in ABLATION_MODELS:
        shap_path = os.path.join(ablation_dir, model_id, 'attribution',
                                 f'{model_id}_shap_values.npy')
        ig_path = os.path.join(ablation_dir, model_id, 'attribution',
                               f'{model_id}_ig_values.npy')
        if os.path.exists(shap_path) and os.path.exists(ig_path) and gene_names:
            try:
                shap_values = np.load(shap_path)
                ig_values = np.load(ig_path)
                for cls_idx, cls_name in enumerate(ALL_CLASSES[:4]):  # First 4 classes
                    fig_shap_vs_ig(
                        shap_values, ig_values, gene_names,
                        cls_idx, cls_name,
                        os.path.join(output_dir, 'shap_vs_ig'),
                        top_n=20)
            except Exception as e:
                print(f"  F15 ({model_id}) failed: {e}")

    # UMAP projections
    if test_labels_path:
        try:
            from figures.umap_figures import run_all_umap
            from config import UMAPConfig
            run_all_umap(
                ablation_dir,
                os.path.join(output_dir, 'umap'),
                test_labels_path,
                UMAPConfig())
        except Exception as e:
            print(f"  UMAP failed: {e}")

    # Per-label explainability
    if test_features is not None and test_labels_str is not None and gene_names:
        try:
            from figures.explainability_figures import run_all_explainability
            from config import AttributionConfig
            run_all_explainability(
                ablation_dir,
                os.path.join(output_dir, 'explainability'),
                np.asarray(test_features, dtype=np.float32),
                test_labels_str, gene_names,
                AttributionConfig())
        except Exception as e:
            print(f"  Explainability failed: {e}")

    print(f"Figures saved to {output_dir}/")
