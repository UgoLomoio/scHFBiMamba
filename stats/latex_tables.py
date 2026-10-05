"""
LaTeX table generation for the classifier ablation (booktabs).

Tables:
  1. table_classifier_ablation.tex — macro-F1 (cell type / species / disease)
     and their arithmetic mean per arm, mean +/- std across folds x seeds
     (plain value when n = 1). Missing results are written as `--`.
  2. table_model_efficiency.tex — parameter counts, FLOPs, and inference
     time per arm; `--` where a quantity is not meaningful (e.g. XGBoost
     FLOPs), with footnotes documenting conventions.

Values are never fabricated: an arm without results renders as `--`.
"""
import os
import numpy as np
from typing import Dict, List, Optional

from config import (
    CLASSIFIER_ABLATION_ORDER, CLASSIFIER_ABLATION_PRETTY,
    CLASSIFIER_ABLATION_REFERENCE,
)

NA = r"--"


def _fmt_f1(mean: Optional[float], std: Optional[float], show_std: bool) -> str:
    """Format an F1 value (4 decimals), optionally with std."""
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return NA
    if show_std and std is not None and not np.isnan(std):
        return f"{mean:.4f} $\\pm$ {std:.4f}"
    return f"{mean:.4f}"


def _fmt_count(x: Optional[float]) -> str:
    """Format parameter/FLOP counts compactly (e.g. 1.2M, 4.5G)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return NA
    for suffix, scale in [("G", 1e9), ("M", 1e6), ("K", 1e3)]:
        if abs(x) >= scale:
            return f"{x / scale:.2f}{suffix}"
    return f"{int(x)}"


def _fmt_time(mean: Optional[float], std: Optional[float]) -> str:
    """Format inference time in ms (2 decimals), optionally with std."""
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return NA
    if std is not None and not np.isnan(std):
        return f"{mean:.2f} $\\pm$ {std:.2f}"
    return f"{mean:.2f}"


def write_classifier_ablation_table(agg: Dict[str, Dict], output_path: str,
                                    show_std: bool = True) -> str:
    """
    Write the classifier-ablation LaTeX table.

    Args:
        agg: {arm_id: {'f1_celltype': (mean, std), 'f1_species': (mean, std),
                       'f1_disease': (mean, std), 'f1_mean3': (mean, std),
                       'n_runs': int}}
        output_path: .tex output path
        show_std: append +/- std when n_runs > 1

    Returns:
        the LaTeX string
    """
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Classifier ablation: macro-F1 per category and their "
        r"arithmetic mean (mean $\pm$ std across folds and seeds).}",
        r"  \label{tab:classifier_ablation}",
        r"  \begin{tabular}{lcccc}",
        r"    \toprule",
        r"    Model & Cell type F1 & Species F1 & Disease F1 & Mean F1 \\",
        r"    \midrule",
    ]
    for arm in CLASSIFIER_ABLATION_ORDER:
        pretty = CLASSIFIER_ABLATION_PRETTY.get(arm, arm)
        entry = agg.get(arm)
        if entry is None or entry.get('n_runs', 0) == 0:
            cells = [NA, NA, NA, NA]
        else:
            use_std = show_std and entry.get('n_runs', 1) > 1
            cells = [
                _fmt_f1(*entry.get('f1_celltype', (None, None)), use_std),
                _fmt_f1(*entry.get('f1_species', (None, None)), use_std),
                _fmt_f1(*entry.get('f1_disease', (None, None)), use_std),
                _fmt_f1(*entry.get('f1_mean3', (None, None)), use_std),
            ]
        lines.append(f"    {pretty} & {' & '.join(cells)} \\\\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    tex = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(tex)
    print(f"  Saved {output_path}")
    return tex


def write_efficiency_table(eff: Dict[str, Dict], output_path: str) -> str:
    """
    Write the model-efficiency LaTeX table.

    Args:
        eff: {arm_id: {'params': int|None, 'flops': float|None,
                       'time_mean_ms': float|None, 'time_std_ms': float|None,
                       'note': str|None}}
        output_path: .tex output path

    Returns:
        the LaTeX string
    """
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Model efficiency: trainable parameters, analytical FLOP "
        r"estimate, and inference time per batch (warm-up + CUDA "
        r"synchronization; identical device, batch size, and precision for "
        r"all comparable models).}",
        r"  \label{tab:model_efficiency}",
        r"  \begin{tabular}{lccc}",
        r"    \toprule",
        r"    Model & Params & FLOPs & Inference time (ms/batch) \\",
        r"    \midrule",
    ]
    footnotes = []
    note_markers = {}  # deduplicate identical notes onto the same marker
    for arm in CLASSIFIER_ABLATION_ORDER:
        pretty = CLASSIFIER_ABLATION_PRETTY.get(arm, arm)
        entry = eff.get(arm)
        if entry is None:
            cells = [NA, NA, NA]
        else:
            cells = [
                _fmt_count(entry.get('params')),
                _fmt_count(entry.get('flops')),
                _fmt_time(entry.get('time_mean_ms'), entry.get('time_std_ms')),
            ]
            note = entry.get('note')
            if note:
                if note not in note_markers:
                    marker = chr(ord('a') + len(footnotes))
                    note_markers[note] = marker
                    footnotes.append(f"$^{{{marker}}}$ {note}")
                pretty = f"{pretty}$^{{{note_markers[note]}}}$"
        lines.append(f"    {pretty} & {' & '.join(cells)} \\\\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
    ]
    if footnotes:
        lines.append(r"  \begin{minipage}{\linewidth}")
        lines.append(r"  \footnotesize")
        for fn in footnotes:
            lines.append(f"  {fn}\\\\")
        lines.append(r"  \end{minipage}")
    lines.append(r"\end{table}")
    tex = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(tex)
    print(f"  Saved {output_path}")
    return tex


def check_reference_values(agg: Dict[str, Dict], tol: float = 0.03) -> List[str]:
    """
    Compare aggregated results against the preliminary reference values
    (MLP / Mamba / BiMamba). Prints PASS/WARN per arm; never fails hard.

    Returns:
        list of status lines (also printed)
    """
    messages = []
    for arm, ref in CLASSIFIER_ABLATION_REFERENCE.items():
        entry = agg.get(arm)
        if entry is None or entry.get('n_runs', 0) == 0:
            msg = f"  REF {arm}: no results available (reference mean F1 = {ref['f1_mean3']:.4f})"
            print(msg)
            messages.append(msg)
            continue
        mean3 = entry['f1_mean3'][0]
        delta = mean3 - ref['f1_mean3']
        status = "PASS" if abs(delta) <= tol else "WARN"
        msg = (f"  REF {arm}: mean F1 = {mean3:.4f} vs reference "
               f"{ref['f1_mean3']:.4f} (delta {delta:+.4f}) [{status}]")
        print(msg)
        messages.append(msg)
    return messages
