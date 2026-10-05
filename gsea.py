"""
GSEA (Gene Set Enrichment Analysis) for DXG and DEG results.

Computes gene set enrichment using gseapy with GO Biological Process gene sets.
Evaluates results against true-positive GO terms from Supplementary Table 8.

Pipeline (matching paper):
  1. Compute GSEA score per gene: sign(FC) * log10(|FC|/p_bin + 1)
  2. Run preranked GSEA with GO Biological Process gene sets
  3. Match significant GO terms against true-positive keywords
  4. Compute F1-score for pathway recovery
"""
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from go_terms import load_go_keywords, match_go_terms_to_keywords, compute_gsea_f1


def compute_dxg_gsea_score(dxg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute GSEA score per gene for DXG results.
    Score = sign(FC) * log10(|FC|/p_bin + 1)
    where p_bin = 0.1 if p < 0.05, else 1.

    For DXG, FC is the cube-root fold change of SHAP values.
    """
    df = dxg_df.copy()
    df['p_bin'] = np.where(df['p_adj'] < 0.05, 0.1, 1.0)
    df['gsea_score'] = np.sign(df['avg_cube_root_FC']) * np.log10(
        np.abs(df['avg_cube_root_FC']) / df['p_bin'] + 1
    )
    return df


def run_gsea(
    scored_df: pd.DataFrame,
    gene_name_col: str = 'Feature',
    score_col: str = 'gsea_score',
    gene_set: str = 'GO_Biological_Process_2023',
    output_dir: str = 'gsea_results',
    name: str = 'gsea',
    min_gene_set_size: int = 10,
    max_gene_set_size: int = 500,
) -> pd.DataFrame:
    """
    Run preranked GSEA using gseapy.

    Args:
        scored_df: DataFrame with gene names and scores
        gene_name_col: column name for gene symbols
        score_col: column name for enrichment scores
        gene_set: gene set library name
        output_dir: directory for GSEA output
        name: name prefix for this run

    Returns:
        DataFrame with GSEA results (Term, ES, NES, pval, fdr)
    """
    import gseapy

    os.makedirs(output_dir, exist_ok=True)

    # Prepare ranked gene list
    rnk = scored_df[[gene_name_col, score_col]].copy()
    rnk.columns = ['Gene', 'Score']
    rnk = rnk.sort_values('Score', ascending=False)
    rnk = rnk.drop_duplicates(subset='Gene', keep='first')

    # Run preranked GSEA
    try:
        gsea_results = gseapy.prerank(
            rnk=rnk,
            gene_sets=gene_set,
            outdir=os.path.join(output_dir, name),
            min_size=min_gene_set_size,
            max_size=max_gene_set_size,
            permutation_num=1000,
            seed=1234,
            threads=4,
            no_plot=True,
            verbose=False,
        )

        # Extract results
        result_df = gsea_results.res2d.copy()
        return result_df

    except Exception as e:
        print(f"  GSEA failed for {name}: {e}")
        return pd.DataFrame()


def evaluate_gsea_f1(
    gsea_results: pd.DataFrame,
    cell_type: str,
    go_keywords: Dict[str, List[str]],
    p_thresh: float = 0.05,
) -> Dict:
    """
    Evaluate GSEA results against true-positive GO terms.

    Args:
        gsea_results: DataFrame from run_gsea (Term, NES, pval, fdr columns)
        cell_type: cell type name (for looking up keywords)
        go_keywords: dict from load_go_keywords
        p_thresh: p-value threshold for significant terms

    Returns:
        dict with F1, precision, recall, and term lists
    """
    if gsea_results is None or len(gsea_results) == 0:
        return {'f1': 0.0, 'precision': 0.0, 'recall': 0.0,
                'tp': 0, 'fp': 0, 'fn': 0,
                'found_terms': [], 'tp_terms': []}

    # Get significant terms
    p_col = 'FDR q-val' if 'FDR q-val' in gsea_results.columns else 'pval'
    sig = gsea_results[gsea_results[p_col] <= p_thresh]
    found_terms = set(sig['Term'].tolist()) if 'Term' in sig.columns else set()

    # Get true positive keywords for this cell type
    keywords = go_keywords.get(cell_type, [])

    # Match found terms against keywords
    all_terms = set(gsea_results['Term'].tolist()) if 'Term' in gsea_results.columns else set()
    tp_terms = match_go_terms_to_keywords(list(all_terms), keywords)

    # Compute F1
    f1_result = compute_gsea_f1(found_terms, tp_terms, all_terms)

    return {
        'f1': f1_result['f1'],
        'precision': f1_result['precision'],
        'recall': f1_result['recall'],
        'tp': f1_result['tp'],
        'fp': f1_result['fp'],
        'fn': f1_result['fn'],
        'found_terms': list(found_terms),
        'tp_terms': list(tp_terms),
    }


def run_gsea_for_all_comparisons(
    dxg_results: Dict[str, pd.DataFrame],
    deg_results: Dict[str, pd.DataFrame],
    go_keywords: Dict[str, List[str]],
    output_dir: str,
    gene_set: str = 'GO_Biological_Process_2023',
) -> Dict:
    """
    Run GSEA for all DXG and DEG comparisons and evaluate F1 scores.

    Args:
        dxg_results: dict from dxg.run_all_dxg
        deg_results: dict from deg.run_all_deg
        go_keywords: dict from load_go_keywords
        output_dir: output directory
        gene_set: gene set library

    Returns:
        dict with GSEA F1 results for DXG and DEG, per comparison
    """
    os.makedirs(output_dir, exist_ok=True)
    all_f1 = {}

    # Get all comparison names
    all_comps = set(list(dxg_results.keys()) + list(deg_results.keys()))

    for comp_name in sorted(all_comps):
        print(f"\n  GSEA for {comp_name}...")

        # Determine cell type from comparison name
        # Format: "Species_CellType_Disease_vs_Species_CellType_CTRL"
        parts = comp_name.split('_')
        cell_type = parts[1] if len(parts) > 1 else ''

        comp_f1 = {}

        # DXG GSEA (one failing comparison must not kill the whole stage)
        if comp_name in dxg_results and len(dxg_results[comp_name]) > 0:
            try:
                scored_dxg = compute_dxg_gsea_score(dxg_results[comp_name])
                gsea_dxg = run_gsea(
                    scored_dxg, output_dir=os.path.join(output_dir, 'dxg'),
                    name=f"dxg_{comp_name}", gene_set=gene_set)
                f1_dxg = evaluate_gsea_f1(gsea_dxg, cell_type, go_keywords)
                comp_f1['dxg'] = f1_dxg
                print(f"    DXG F1: {f1_dxg['f1']:.4f} (TP={f1_dxg.get('tp', 0)}, "
                      f"FP={f1_dxg.get('fp', 0)}, FN={f1_dxg.get('fn', 0)})")
            except Exception as e:
                print(f"    DXG GSEA failed for {comp_name}: {e}")

        # DEG GSEA
        if comp_name in deg_results and len(deg_results[comp_name]) > 0:
            try:
                from deg import compute_deg_score
                scored_deg = compute_deg_score(deg_results[comp_name])
                gsea_deg = run_gsea(
                    scored_deg, output_dir=os.path.join(output_dir, 'deg'),
                    name=f"deg_{comp_name}", gene_set=gene_set)
                f1_deg = evaluate_gsea_f1(gsea_deg, cell_type, go_keywords)
                comp_f1['deg'] = f1_deg
                print(f"    DEG F1: {f1_deg['f1']:.4f} (TP={f1_deg.get('tp', 0)}, "
                      f"FP={f1_deg.get('fp', 0)}, FN={f1_deg.get('fn', 0)})")
            except Exception as e:
                print(f"    DEG GSEA failed for {comp_name}: {e}")

        all_f1[comp_name] = comp_f1

    # Save summary
    summary_rows = []
    for comp_name, f1s in all_f1.items():
        for method, f1_data in f1s.items():
            summary_rows.append({
                'comparison': comp_name,
                'method': method,
                'f1': f1_data['f1'],
                'precision': f1_data['precision'],
                'recall': f1_data['recall'],
                'tp': f1_data.get('tp', 0),
                'fp': f1_data.get('fp', 0),
                'fn': f1_data.get('fn', 0),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(output_dir, 'gsea_f1_summary.csv'), index=False)

    print(f"\nGSEA F1 summary saved to {output_dir}/gsea_f1_summary.csv")
    return all_f1
