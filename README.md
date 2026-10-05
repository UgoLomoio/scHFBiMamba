# HF Mamba: Heart Failure Subtyping with Bidirectional Mamba Networks

A GPU-ready codebase that replicates and extends the workflow of Ruz Jurado et al. (2025),
replacing the MLP classifier with a **Bidirectional Mamba** state-space model architecture.

## Ablation Matrix

| Model | Encoder | Classifier | Direction | Purpose |
|-------|---------|------------|-----------|---------|
| M1 | Dense DAE (16545→350) | BiMamba | Bidirectional | Full model |
| M2 | Dense DAE (16545→350) | BiMamba | Unidirectional | Direction ablation |
| M3 | BiMamba AE (16545→350) | MLP | — | Classifier ablation |
| M4 | BiMamba AE (16545→350) | BiMamba | Bidirectional | Encoder ablation |

Each model is evaluated with both SHAP and Integrated Gradients attribution.

## Requirements

- **CUDA-capable GPU** (required for `mamba_ssm`)
- Python 3.10+

```bash
pip install torch>=2.1.0  # match your CUDA version
pip install causal-conv1d>=1.4.0
pip install mamba-ssm>=2.0.0
pip install shap captum gseapy umap-learn openpyxl
pip install matplotlib seaborn matplotlib-venn scipy statsmodels scikit-learn tqdm pandas numpy
```

See `requirements.txt` for full list.

## Dataset

Download from HuggingFace: `mruzjurado/CVD_dataset`

```bash
mkdir -p data
cd data
# Download these 6 files:
# RANDOMIZED_train_set_p_80.csv          (~8.8 GB)
# RANDOMIZED_val_set_p_20.csv            (~2.2 GB)
# 240519_test_set_only_n2_samples.csv    (~1.9 GB)
# RANDOMIZED_train_set_labels_p_80.csv
# RANDOMIZED_val_set_labels_p_20.csv
# 240519_test_set_only_n2_samples_labels.csv
```

Also download Supplementary Table 8 (GO terms) from the paper's GitHub:
```bash
wget https://raw.githubusercontent.com/MarianoRuzJurado/RuzJurado_et_al_2025/main/paper_code/SUPPL_TABLES/Suppl_Table_8_TP_GO_ID_per_cell_type.xlsx
```

## Quick Start

### 1. Train a single model

```bash
# M1: Dense DAE + BiMamba (BiDir)
python train.py \
  --train_path data/RANDOMIZED_train_set_p_80.csv \
  --val_path data/RANDOMIZED_val_set_p_20.csv \
  --test_path data/240519_test_set_only_n2_samples.csv \
  --train_labels data/RANDOMIZED_train_set_labels_p_80.csv \
  --val_labels data/RANDOMIZED_val_set_labels_p_20.csv \
  --test_labels data/240519_test_set_only_n2_samples_labels.csv \
  --encoder_type dense_dae --classifier_type mamba \
  --direction bidirectional --use_amp \
  --output_dir results/M1

# M3: BiMamba AE + MLP
python train.py \
  --train_path data/RANDOMIZED_train_set_p_80.csv \
  --val_path data/RANDOMIZED_val_set_p_20.csv \
  --test_path data/240519_test_set_only_n2_samples.csv \
  --train_labels data/RANDOMIZED_train_set_labels_p_80.csv \
  --val_labels data/RANDOMIZED_val_set_labels_p_20.csv \
  --test_labels data/240519_test_set_only_n2_samples_labels.csv \
  --encoder_type bimamba_ae --classifier_type mlp \
  --direction bidirectional --use_amp \
  --output_dir results/M3
```

### 2. Run the full ablation matrix (all 4 models + attribution)

```bash
python run_ablation.py \
  --data_dir data \
  --output_dir results/ablation \
  --go_terms_path data/Suppl_Table_8_TP_GO_ID_per_cell_type.xlsx \
  --epochs 500 --batch_size 1024 --use_amp
```

### 3. Run full analysis (DXG, DEG, GSEA, stats, UMAP, explainability, figures)

```bash
python run_analysis.py \
  --ablation_dir results/ablation \
  --output_dir results/analysis \
  --go_terms_path data/Suppl_Table_8_TP_GO_ID_per_cell_type.xlsx \
  --test_path data/240519_test_set_only_n2_samples.csv \
  --test_labels data/240519_test_set_only_n2_samples_labels.csv
```

### 4. Evaluate a trained model

```bash
python test.py \
  --model_dir results/M1 \
  --test_path data/240519_test_set_only_n2_samples.csv \
  --test_labels data/240519_test_set_only_n2_samples_labels.csv \
  --save_composite --save_embeddings
```

## Code Structure

```
hf_mamba/
├── config.py              # Hyperparameters, ablation configs, class definitions
├── data.py                # Data loading, memmap, label encoding, streaming
├── losses.py              # Macro F1-loss, metrics, confusion matrices
├── models/
│   ├── dae.py             # Dense Denoising Autoencoder (M1/M2 encoder)
│   ├── bimamba_ae.py      # BiMamba Autoencoder (M3/M4 encoder)
│   ├── mamba_block.py     # Mamba block (mamba_ssm wrapper)
│   ├── bimamba.py         # BiMamba layer (conv + blocks + residual)
│   ├── classifier.py      # BiMamba classifier (M1/M2/M4)
│   ├── mlp_classifier.py  # MLP classifier (M3, paper's architecture)
│   └── composite_model.py # Encoder + Classifier wrapper for gene-level attribution
├── train.py               # GPU training (all 4 configs, AMP, checkpointing)
├── test.py                # Evaluation, confusion matrices, embeddings
├── attribution/
│   ├── shap_attrib.py     # SHAP GradientExplainer (gene-level via composite)
│   └── ig_attrib.py       # Captum Integrated Gradients (gene-level via composite)
├── stats/
│   ├── roc_pr.py          # ROC/PR curves, AUC, Average Precision
│   ├── delong_test.py     # DeLong's test for correlated ROC AUC comparison
│   ├── mcnemar_test.py    # McNemar's test for accuracy comparison
│   └── summary_table.py   # Combined stats table + p-value heatmap
├── dxg.py                 # DXG analysis (Z-transform, t-test, BH, FC)
├── deg.py                 # Traditional DEG (Wilcoxon on expression)
├── gsea.py                # GSEA scoring, enrichment, F1 evaluation
├── go_terms.py            # True-positive GO keywords (Suppl Table 8)
├── figures.py             # Publication figures (workflow, curves, heatmaps, etc.)
├── figures/
│   ├── umap_figures.py    # UMAP projections (24 figures: 4 models × 2 emb × 3 colors)
│   └── explainability_figures.py  # Per-label beeswarm (52 figures: 13 labels × 4 models)
├── run_ablation.py        # Orchestrate 4-model ablation + attribution
├── run_analysis.py        # Orchestrate DXG/DEG/GSEA/stats/UMAP/figures
├── report_hf_mamba.md     # Publication-ready report template
├── requirements.txt       # GPU dependencies
└── README.md              # This file
```

## Output Files

After `run_ablation.py`:
- `results/ablation/shared_dense_dae/` — Shared Dense DAE (M1/M2)
- `results/ablation/shared_bimamba_ae/` — Shared BiMamba AE (M3/M4)
- `results/ablation/M{1-4}/encoder_best.pt` — Encoder weights
- `results/ablation/M{1-4}/classifier_best.pt` — Classifier weights
- `results/ablation/M{1-4}/composite_model.pt` — Composite model for attribution
- `results/ablation/M{1-4}/test_latent_embeddings.npy` — Latent for UMAP
- `results/ablation/M{1-4}/test_clf_features.npy` — Classifier features for UMAP
- `results/ablation/M{1-4}/attribution/` — SHAP and IG gene-level attributions

After `run_analysis.py`:
- `results/analysis/stats/` — ROC/PR curves, DeLong/McNemar results, p-value heatmap
- `results/analysis/umap/` — 24 UMAP figures
- `results/analysis/explainability/` — 52 per-label beeswarm figures
- `results/analysis/dxg/` — DXG results per model × method
- `results/analysis/deg/` — DEG results
- `results/analysis/gsea/` — GSEA results and F1 scores
- `results/analysis/figures/` — All other publication figures

## Figures Generated

| Category | Figures | Description |
|----------|---------|-------------|
| Workflow | 1 | F01: Pipeline schematic |
| Training | 4 | F03: Training curves per model |
| Classification | 3+ | F04: Classification rates, confusion matrices |
| ROC/PR | 6 | ROC and PR curves per category (species, celltype, disease) |
| Stats | 1 | P-value heatmap (DeLong + McNemar, 6 pairs × 3 categories) |
| UMAP | 24 | 4 models × 2 embeddings × 3 colorings |
| Explainability | 52 | 13 labels × 4 models beeswarm plots |
| Ablation | 1 | F14: Ablation heatmap |
| DXG/DEG | 3+ | F10: Venn diagrams, F15: SHAP vs IG |
| GSEA | 1 | F13: GSEA F1 across cell types |
