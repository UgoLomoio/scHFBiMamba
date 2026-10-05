#!/usr/bin/env python3
"""
Orchestration script for DXG + DEG + GSEA + Stats + UMAP + Explainability analysis.

Runs the full downstream analysis pipeline for all models and attribution methods:
  1. Load attribution scores (SHAP/IG) for each model
  2. Run DXG analysis (differentially explained genes)
  3. Run DEG analysis (traditional differentially expressed genes)
  4. Run GSEA for both DXG and DEG
  5. Evaluate F1 scores against true-positive GO terms
  6. ROC/PR curves
  7. DeLong test (ROC AUC comparison)
  8. McNemar test (accuracy comparison)
  9. Stats summary table
  10. UMAP projections (latent + classifier features)
  11. Per-label explainability figures
  12. All other publication figures

Usage:
  python run_analysis.py \
    --ablation_dir results/ablation \
    --output_dir results/analysis \
    --go_terms_path data/Suppl_Table_8_TP_GO_ID_per_cell_type.xlsx \
    --test_path data/240519_test_set_only_n2_samples.csv \
    --test_labels data/240519_test_set_only_n2_samples_labels.csv
"""
import argparse
import os
import json
import numpy as np
import torch
from typing import Dict, List

from config import ABLATION_MODELS, ABLATION_ATTRIBUTION, N_GENES, UMAPConfig, AttributionConfig
from data import csv_to_memmap, load_labels, load_label_strings, sample_balanced_test_subset
from go_terms import load_go_keywords
from dxg import run_all_dxg
from deg import run_all_deg
from gsea import run_gsea_for_all_comparisons


def parse_args():
    parser = argparse.ArgumentParser(description='Run full analysis pipeline')
    parser.add_argument('--ablation_dir', type=str, required=True,
                        help='Directory with trained models (from run_ablation.py)')
    parser.add_argument('--output_dir', type=str, default='results/analysis')
    parser.add_argument('--go_terms_path', type=str,
                        default='data/Suppl_Table_8_TP_GO_ID_per_cell_type.xlsx')
    parser.add_argument('--test_path', type=str, required=True)
    parser.add_argument('--test_labels', type=str, required=True)
    parser.add_argument('--cache_dir', type=str, default='data/cache')
    parser.add_argument('--gene_set', type=str, default='GO_Biological_Process_2023')
    parser.add_argument('--models', type=str, nargs='+', default=None)
    parser.add_argument('--attributions', type=str, nargs='+', default=None)
    parser.add_argument('--skip_gsea', action='store_true',
                        help='Skip GSEA (useful if gseapy not installed)')
    parser.add_argument('--skip_stats', action='store_true',
                        help='Skip statistical tests (ROC/PR, DeLong, McNemar)')
    parser.add_argument('--skip_umap', action='store_true',
                        help='Skip UMAP projections')
    parser.add_argument('--skip_explainability', action='store_true',
                        help='Skip per-label explainability figures')
    parser.add_argument('--skip_cg')
    return parser.parse_args()


def load_attribution(ablation_dir: str, model_id: str, method: str) -> dict:
    """Load pre-computed attribution scores for a model and method."""
    attrib_dir = os.path.join(ablation_dir, model_id, 'attribution')
    prefix = f"{model_id}_"

    if method == 'shap':
        values_path = os.path.join(attrib_dir, f'{prefix}shap_values.npy')
        labels_path = os.path.join(attrib_dir, f'{prefix}shap_test_labels.txt')
        genes_path = os.path.join(attrib_dir, f'{prefix}shap_gene_names.txt')
    else:  # ig
        values_path = os.path.join(attrib_dir, f'{prefix}ig_values.npy')
        labels_path = os.path.join(attrib_dir, f'{prefix}ig_test_labels.txt')
        genes_path = os.path.join(attrib_dir, f'{prefix}ig_gene_names.txt')

    if not os.path.exists(values_path):
        print(f"  Warning: {values_path} not found, skipping")
        return None

    values = np.load(values_path)
    with open(labels_path, 'r') as f:
        labels_str = f.read().strip().split('\n')
    with open(genes_path, 'r') as f:
        gene_names = f.read().strip().split('\n')

    return {
        'values': values,
        'labels_str': labels_str,
        'gene_names': gene_names,
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load test data
    print("Loading test data...")
    test_features, gene_names = csv_to_memmap(
        args.test_path, args.cache_dir, "test")
    test_labels_str = load_label_strings(args.test_labels)
    test_labels = load_labels(args.test_labels)

    # Save test labels as .npy for stats modules
    np.save(os.path.join(args.ablation_dir, 'test_labels.npy'), test_labels)
    # Also copy to each model dir for stats
    for model_id in ABLATION_MODELS:
        model_dir = os.path.join(args.ablation_dir, model_id)
        if os.path.isdir(model_dir):
            np.save(os.path.join(model_dir, 'test_labels.npy'), test_labels)

    # Load GO keywords
    print("Loading GO keywords...")
    go_keywords = load_go_keywords(args.go_terms_path)
    for ct, kws in go_keywords.items():
        print(f"  {ct}: {len(kws)} keywords")

    # Determine models and attribution methods
    model_ids = args.models or list(ABLATION_MODELS.keys())
    attrib_methods = args.attributions or ABLATION_ATTRIBUTION

    # ─── 1. DEG analysis (shared baseline) ───────────────────────────────────
    print("\n" + "=" * 60)
    print("Running DEG analysis (shared baseline)")
    print("=" * 60)
    deg_dir = os.path.join(args.output_dir, 'deg')
    deg_results = run_all_deg(
        np.asarray(test_features, dtype=np.float32),
        test_labels_str, gene_names, deg_dir)

    # ─── 2. DXG + GSEA for each model × attribution method ───────────────────
    
    if not args.skip_gsea:
       
        all_gsea_f1 = {}
        n_attrib_found = 0
        missing_attributions = []

        subset_features, subset_labels, subset_labels_str = (
            sample_balanced_test_subset(
                test_features,
                test_labels_str,
                n_per_celltype=200,
                seed=1111,
            )
        )

        subset_features = np.asarray(subset_features, dtype=np.float32)
        subset_labels = np.asarray(subset_labels, dtype=np.float32)
        subset_labels_str = np.asarray(subset_labels_str)

        for model_id in model_ids:
            for method in attrib_methods:
                print(f"\n{'=' * 60}")
                print(f"DXG + GSEA: {model_id} / {method.upper()}")
                print(f"{'=' * 60}")

                attrib = load_attribution(
                    args.ablation_dir,
                    model_id,
                    method,
                )

                if attrib is None:
                    missing_attributions.append((model_id, method))
                    print(f"Skipping missing attribution: {model_id}/{method}")
                    continue

                n_attrib_found += 1

                attrib_values = np.asarray(attrib["values"])
                attrib_labels_str = np.asarray(attrib["labels_str"])
                attrib_gene_names = np.asarray(attrib["gene_names"])

                print(
                    f"{model_id}/{method}: raw attribution shape = "
                    f"{attrib_values.shape}"
                )

                if attrib_values.ndim != 3:
                    raise ValueError(
                        f"{model_id}/{method}: run_all_dxg requires a 3D array "
                        f"(classes, samples, genes), got {attrib_values.shape}"
                    )

                n_classes, n_samples, n_genes = attrib_values.shape

                if n_samples != len(attrib_labels_str):
                    raise ValueError(
                        f"{model_id}/{method}: attribution samples={n_samples}, "
                        f"labels={len(attrib_labels_str)}"
                    )

                if n_genes != len(attrib_gene_names):
                    raise ValueError(
                        f"{model_id}/{method}: attribution genes={n_genes}, "
                        f"gene names={len(attrib_gene_names)}"
                    )

                print(
                    f"{model_id}/{method}: using class-specific attribution shape "
                    f"{attrib_values.shape}"
                )

                dxg_dir = os.path.join(
                    args.output_dir,
                    f"{model_id}_{method}",
                    "dxg",
                )

                dxg_results = run_all_dxg(
                    attrib_values,
                    attrib_labels_str,
                    attrib_gene_names,
                    subset_features,
                    dxg_dir,
                    model_name=f"{model_id}_{method}",
                )

                gsea_dir = os.path.join(
                    args.output_dir,
                    f"{model_id}_{method}",
                    "gsea",
                )

                gsea_f1 = run_gsea_for_all_comparisons(
                    dxg_results,
                    deg_results,
                    go_keywords,
                    gsea_dir,
                    gene_set=args.gene_set,
                )

                all_gsea_f1[f"{model_id}_{method}"] = gsea_f1

        print(f"Found {n_attrib_found} attribution files")

        if missing_attributions:
            print("Missing attribution files:")
            for model_id, method in missing_attributions:
                print(f"  - {model_id}/{method}")

        if n_attrib_found == 0:
            print("\n" + "!" * 60)
            print("WARNING: no attribution files found for any model under")
            print(f"  {args.ablation_dir}/<MODEL>/attribution/")
            print("DXG/GSEA and all attribution-based figures will be skipped.")
            print("To generate attributions from existing trained models, run:")
            print("  python run_ablation.py --skip_training \\")
            print("    --data_dir data --output_dir results/ablation")
            print("(or run the full run_ablation.py if models were never trained).")
            print("!" * 60)

        # Save GSEA F1 summary
        if all_gsea_f1 and not args.skip_gsea:
            with open(os.path.join(args.output_dir, 'all_gsea_f1.json'), 'w') as f:
                def serialize(obj):
                    if isinstance(obj, set):
                        return list(obj)
                    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
                json.dump(all_gsea_f1, f, indent=2, default=serialize)

    # ─── 3. Statistical tests ────────────────────────────────────────────────
    if not args.skip_stats:
        print("\n" + "=" * 60)
        print("Statistical Tests (ROC/PR, DeLong, McNemar)")
        print("=" * 60)

        stats_dir = os.path.join(args.output_dir, 'stats')
        os.makedirs(stats_dir, exist_ok=True)

        # ROC/PR curves
        try:
            from stats.roc_pr import run_roc_pr_analysis
            run_roc_pr_analysis(args.ablation_dir, stats_dir, model_ids)
            print(f"ROC/PR results saved to {stats_dir}/")
        except Exception as e:
            print(f"  ROC/PR failed: {e}")

        # DeLong test
        try:
            from stats.delong_test import run_delong_for_all_pairs
            delong_df = run_delong_for_all_pairs(args.ablation_dir, stats_dir, model_ids)
            print(f"DeLong results saved to {stats_dir}/delong_results.csv")
        except Exception as e:
            print(f"  DeLong failed: {e}")
            delong_df = None

        # McNemar test
        try:
            from stats.mcnemar_test import run_mcnemar_for_all_pairs
            mcnemar_df = run_mcnemar_for_all_pairs(args.ablation_dir, stats_dir, model_ids)
            print(f"McNemar results saved to {stats_dir}/mcnemar_results.csv")
        except Exception as e:
            print(f"  McNemar failed: {e}")
            mcnemar_df = None

        # Summary table
        try:
            from stats.summary_table import run_stats_summary
            if delong_df is not None and mcnemar_df is not None:
                run_stats_summary(delong_df, mcnemar_df, stats_dir)
                print(f"Stats summary saved to {stats_dir}/stats_summary.csv")
        except Exception as e:
            print(f"  Stats summary failed: {e}")

    # ─── 4. UMAP projections ─────────────────────────────────────────────────
    if not args.skip_umap:
        print("\n" + "=" * 60)
        print("UMAP Projections")
        print("=" * 60)

        try:
            from figures.umap_figures import run_all_umap
            umap_dir = os.path.join(args.output_dir, 'umap')
            run_all_umap(
                args.ablation_dir, umap_dir,
                args.test_labels,
                UMAPConfig(), model_ids)
        except Exception as e:
            print(f"  UMAP failed: {e}")
            import traceback
            traceback.print_exc()

    # ─── 5. Per-label explainability ─────────────────────────────────────────
    if not args.skip_explainability:
        print("\n" + "=" * 60)
        print("Per-Label Explainability Figures")
        print("=" * 60)

        try:
            from figures.explainability_figures import run_all_explainability
            expl_dir = os.path.join(args.output_dir, 'explainability')
            run_all_explainability(
                args.ablation_dir, expl_dir,
                np.asarray(test_features, dtype=np.float32),
                test_labels_str, gene_names,
                AttributionConfig(), model_ids)
        except Exception as e:
            print(f"  Explainability failed: {e}")
            import traceback
            traceback.print_exc()

    # ─── 6. All other figures ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Generating publication figures")
    print("=" * 60)
    try:
        from publication_figures import generate_all_figures
        generate_all_figures(
            ablation_dir=args.ablation_dir,
            analysis_dir=args.output_dir,
            output_dir=os.path.join(args.output_dir, 'figures'),
            go_keywords=go_keywords,
            test_features=test_features,
            test_labels_str=test_labels_str,
            gene_names=gene_names,
            test_labels_path=args.test_labels,
        )
    except Exception as e:
        print(f"  Figure generation failed: {e}")
        import traceback
        traceback.print_exc()

    print(f"\nAnalysis complete. Results saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
