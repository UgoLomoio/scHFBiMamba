"""
Configuration for HF Mamba subtyping pipeline.
All hyperparameters, paths, ablation configs, and class definitions.
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ─── Fixed class definitions (identical to paper) ───────────────────────────
# 13 output nodes: 2 species + 7 cell types + 4 disease states
SPECIES_CLASSES = ['Human', 'Mice']
CELLTYPE_CLASSES = ['Cardiomyocytes', 'Endothelial', 'Fibroblasts', 'Immune.cells',
                    'Neuro', 'Pericytes', 'Smooth.Muscle']
DISEASE_CLASSES = ['AS', 'HFpEF', 'HFrEF', 'CTRL']

ALL_CLASSES = SPECIES_CLASSES + CELLTYPE_CLASSES + DISEASE_CLASSES  # 13 total
N_CLASSES = len(ALL_CLASSES)  # 13

# Index ranges for metric computation
SPECIES_IDX = (0, 2)       # [0:2]
CELLTYPE_IDX = (2, 9)      # [2:9]
DISEASE_IDX = (9, 13)      # [9:13]

# Label string format: "Species-Disease_CellType" e.g. "Human-HFpEF_Cardiomyocytes"
# The paper uses "-" as first separator and "_" as second
# e.g. "Human-CTRL_Pericytes" -> species=Human, disease=CTRL, celltype=Pericytes


# ─── Gene info ──────────────────────────────────────────────────────────────
N_GENES = 16545
MT_GENE_PREFIX = 'MT.'  # Mitochondrial genes excluded from SHAP


# ─── Data paths (defaults — override via CLI) ───────────────────────────────
@dataclass
class DataConfig:
    train_path: str = "data/RANDOMIZED_train_set_p_80.csv"
    val_path: str = "data/RANDOMIZED_val_set_p_20.csv"
    test_path: str = "data/240519_test_set_only_n2_samples.csv"
    train_labels_path: str = "data/RANDOMIZED_train_set_labels_p_80.csv"
    val_labels_path: str = "data/RANDOMIZED_val_set_labels_p_20.csv"
    test_labels_path: str = "data/240519_test_set_only_n2_samples_labels.csv"
    cache_dir: str = "data/cache"  # for memmap .npy files
    # Koenig validation dataset (optional)
    koenig_path: str = ""
    koenig_labels_path: str = ""


# ─── Dense DAE hyperparameters (from paper) ─────────────────────────────────
@dataclass
class DAEConfig:
    input_dim: int = N_GENES  # 16545
    encoder_neurons: List[int] = field(default_factory=lambda: [5000, 2400, 350])
    decoder_neurons: List[int] = field(default_factory=lambda: [2200, 5150])
    latent_dim: int = 350
    loss: str = "mse"
    optimizer: str = "adam"
    lr: float = 0.001
    lr_factor: float = 0.1
    lr_patience: int = 25
    lr_min: float = 1e-7
    early_stop_patience: int = 50
    max_epochs: int = 500
    batch_size: int = 1024
    seed: int = 1234


# ─── BiMamba Autoencoder hyperparameters ────────────────────────────────────
@dataclass
class BiMambaAEConfig:
    input_dim: int = N_GENES
    latent_dim: int = 350

    # Compression before Mamba
    stem_dim: int = 350
    n_latent_tokens: int = 32

    # Mamba processes only learned latent tokens
    d_model: int = 64
    d_state: int = 16
    d_conv: int = 4
    expand_factor: int = 2
    n_layers: int = 1
    dropout: float = 0.10

    batch_size: int = 350
    use_amp: bool = True

    # BiMamba stack
    n_layers: int = 1
    num_blocks: int = 1
    bidirectional: bool = True
    use_start_conv: bool = True
    use_end_conv: bool = True
    use_intra_residual: bool = True
    use_global_residual: bool = True
    conv_kernel: int = 3

    # Pooling
    pool: str = "mean_max"  # encoder pooling

    # Training
    loss: str = "mse"
    optimizer: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 1e-6
    lr_factor: float = 0.1
    lr_patience: int = 25
    lr_min: float = 1e-7
    early_stop_patience: int = 50
    max_epochs: int = 100
    batch_size: int = 32
    seed: int = 1234
    use_amp: bool = True


# ─── BiMamba classifier hyperparameters ─────────────────────────────────────
@dataclass
class MambaConfig:
    # Input
    input_mode: str = "latent"  # "latent" (from any encoder) — always 350
    seq_len: int = 350  # always 350 (latent dim from encoder)

    # Mamba block
    d_model: int = 64
    d_state: int = 16
    d_conv: int = 4
    expand_factor: int = 2

    # BiMamba stack
    n_layers: int = 3
    num_blocks: int = 2  # blocks per layer
    bidirectional: bool = True
    use_start_conv: bool = True
    use_end_conv: bool = True
    use_intra_residual: bool = True      # start→end residual within layer
    use_global_residual: bool = True     # first→last layer residual
    conv_kernel: int = 3

    # Classification head
    pool: str = "mean_max"  # "mean", "max", "mean_max", or "attention"
    classifier_hidden: int = 128
    dropout: float = 0.1
    n_outputs: int = N_CLASSES  # 13

    # Training
    loss: str = "bce"  # nominal; trainer uses BCE, or weighted BCE with "bce_weighted"
    optimizer: str = "adam"
    lr: float = 0.001
    lr_factor: float = 0.1
    lr_patience: int = 25
    lr_min: float = 1e-7
    early_stop_patience: int = 50
    max_epochs: int = 500
    batch_size: int = 1024
    threshold: float = 0.5
    seed: int = 1234
    use_amp: bool = True  # mixed precision

    # Checkpointing
    checkpoint_every: int = 10


# ─── MLP classifier hyperparameters (paper's architecture) ──────────────────
@dataclass
class MLPConfig:
    """MLP classifier matching the paper's architecture exactly."""
    input_dim: int = 350  # latent dim from encoder
    hidden_neurons: List[int] = field(default_factory=lambda: [795, 230, 105])
    n_outputs: int = N_CLASSES  # 13
    dropout: float = 0.1
    activation: str = "relu"

    # Training
    loss: str = "bce"
    optimizer: str = "adam"
    lr: float = 0.001
    lr_factor: float = 0.1
    lr_patience: int = 25
    lr_min: float = 1e-7
    early_stop_patience: int = 50
    max_epochs: int = 500
    batch_size: int = 1024
    threshold: float = 0.5
    seed: int = 1234
    use_amp: bool = True
    checkpoint_every: int = 10


# ─── Attribution hyperparameters ────────────────────────────────────────────
@dataclass
class AttributionConfig:
    method: str = "shap"  # "shap" or "ig"
    # SHAP
    shap_background_size: int = 1000
    shap_seed: int = 1111
    # IG
    ig_steps: int = 200
    ig_baseline: str = "zero"  # "zero" or "mean"
    # Test set: balanced sampling
    n_cells_per_celltype: int = 200
    # Per-label explainability
    n_cells_per_label: int = 100
    explainability_seed: int = 1111
    top_n_genes: int = 20
    # Exclude MT genes
    exclude_mt: bool = True


# ─── UMAP hyperparameters ───────────────────────────────────────────────────
@dataclass
class UMAPConfig:
    n_neighbors: int = 30
    min_dist: float = 0.3
    metric: str = "euclidean"
    n_components: int = 2
    random_state: int = 42


# ─── Statistical test hyperparameters ───────────────────────────────────────
@dataclass
class StatsConfig:
    # DeLong test
    delong_alpha: float = 0.05
    # McNemar test
    mcnemar_correction: bool = True  # continuity correction
    mcnemar_alpha: float = 0.05
    # ROC/PR
    roc_n_thresholds: int = 100


# ─── DXG / DEG / GSEA hyperparameters ───────────────────────────────────────
@dataclass
class AnalysisConfig:
    # DXG / DEG
    testing_method: str = "t"  # "t" or "wilcoxon"
    shap_thresh: float = 0.01
    logfc_thresh: float = 0.1
    expression_threshold: float = 0.1
    p_adj_thresh: float = 0.05
    # GSEA
    gsea_gene_set: str = "GO_Biological_Process_2023"
    gsea_p_thresh: float = 0.05
    # True positive GO terms file (Suppl Table 8)
    go_terms_path: str = "data/Suppl_Table_8_TP_GO_ID_per_cell_type.xlsx"


# ─── Ablation matrix definitions ────────────────────────────────────────────
# Updated: M3/M4 now use BiMamba Autoencoder instead of gene_token
# Encoder types: "dense_dae" (M1/M2), "bimamba_ae" (M3/M4)
# Classifier types: "mamba" (M1/M2/M4), "mlp" (M3)
# Direction: "bidirectional" (M1/M3/M4), "unidirectional" (M2)
ABLATION_MODELS = {
    "M0": {
        "encoder_type": "dense_dae",
        "classifier_type": "mlp",
        "direction": "None",
        "seq_len": 350,
        "direction_classifier": "None",
    },
    "M1": {
        "encoder_type": "dense_dae",
        "classifier_type": "mamba",
        "direction_classifier": "unidirectional",
        "direction": "None",
        "seq_len": 350,
    },
    "M2": {
        "encoder_type": "dense_dae",
        "classifier_type": "mamba",
        "direction_classifier": "bidirectional",
        "direction": "None",
        "seq_len": 350,
    },
    "M3": {
        "encoder_type": "bimamba_ae",
        "classifier_type": "mlp",
        "direction": "bidirectional",
        "direction_classifier": "None",
        "seq_len": 350,
    },
    "M4": {
        "encoder_type": "bimamba_ae",
        "classifier_type": "mamba",
        "direction": "bidirectional",
        "direction_classifier": "unidirectional",
        "seq_len": 350,
    },
    "M5": {
        "encoder_type": "bimamba_ae",
        "classifier_type": "mamba",
        "direction": "bidirectional",
        "direction_classifier": "bidirectional",
        "seq_len": 350,
    },
}

ABLATION_ATTRIBUTION = ["shap"]#, "ig"]

# All 6 model pairs for statistical comparison
model_names = list(ABLATION_MODELS.keys())
ABLATION_MODEL_PAIRS = [
    (model_names[i], model_names[j])
    for i in range(len(model_names))
    for j in range(i + 1, len(model_names))
]


def get_config_for_ablation(model_id: str) -> dict:
    """
    Return config objects for a given ablation model ID (M1-M4).
    Returns dict with 'encoder_cfg', 'classifier_cfg', and 'spec'.
    """
    spec = ABLATION_MODELS[model_id]

    # Encoder config
    if spec["encoder_type"] == "dense_dae":
        encoder_cfg = DAEConfig()
    else:  # bimamba_ae
        encoder_cfg = BiMambaAEConfig(
            bidirectional=(spec["direction"] == "bidirectional"),
        )

    # Classifier config
    if spec["classifier_type"] == "mamba":
        classifier_cfg = MambaConfig(
            seq_len=spec["seq_len"],
            bidirectional=(spec["direction_classifier"] == "bidirectional"),
        )
    else:  # mlp
        classifier_cfg = MLPConfig()

    return {
        "encoder_cfg": encoder_cfg,
        "classifier_cfg": classifier_cfg,
        "spec": spec,
    }


# ─── Classifier ablation matrix ─────────────────────────────────────────────
# All arms share the Dense DAE 350-dim latent; only the classifier changes.
# kind: "torch" (trained by train_classifier), "graph" (transductive GNN),
#       "xgboost" (sklearn-style baseline)
# loss: "bce" (default) or "bce_weighted" (per-class pos_weight from train fold)
# direction / pool map onto MambaConfig fields for mamba-kind arms.
CLASSIFIER_ABLATION_ARMS = {
    "mlp": {
        "kind": "torch",
        "classifier_type": "mlp",
        "description": "Paper's MLP (795/230/105 -> 13)",
    },
    "xgboost": {
        "kind": "xgboost",
        "description": "Three multi-class XGBoost models (species/celltype/disease)",
    },
    "gcn": {
        "kind": "graph",
        "graph_model": "gcn",
        "description": "GCN on kNN cell graph (latent space)",
    },
    "gat": {
        "kind": "graph",
        "graph_model": "gat",
        "description": "GAT on kNN cell graph (latent space)",
    },
    "mamba": {
        "kind": "torch",
        "classifier_type": "mamba",
        "direction": "unidirectional",
        "pool": "mean_max",
        "loss": "bce",
        "description": "Unidirectional Mamba (direction ablation)",
    },
    "bimamba": {
        "kind": "torch",
        "classifier_type": "mamba",
        "direction": "bidirectional",
        "pool": "mean_max",
        "loss": "bce",
        "description": "Bidirectional Mamba (full model)",
    },
    "bimamba_weighted": {
        "kind": "torch",
        "classifier_type": "mamba",
        "direction": "bidirectional",
        "pool": "mean_max",
        "loss": "bce_weighted",
        "description": "BiMamba + class-weighted BCE",
    },
    "bimamba_attention": {
        "kind": "torch",
        "classifier_type": "mamba",
        "direction": "bidirectional",
        "pool": "attention",
        "loss": "bce",
        "description": "BiMamba + attention pooling",
    },
    "bimamba_weighted_attention": {
        "kind": "torch",
        "classifier_type": "mamba",
        "direction": "bidirectional",
        "pool": "attention",
        "loss": "bce_weighted",
        "description": "BiMamba + class-weighted BCE + attention pooling",
    },
}

# Display order and pretty names for the LaTeX tables
CLASSIFIER_ABLATION_ORDER = [
    "mlp", "xgboost", "gcn", "gat", "mamba",
    "bimamba", "bimamba_weighted", "bimamba_attention",
    "bimamba_weighted_attention",
]
CLASSIFIER_ABLATION_PRETTY = {
    "mlp": "MLP",
    "xgboost": "XGBoost",
    "gcn": "GCN",
    "gat": "GAT",
    "mamba": "Mamba",
    "bimamba": "BiMamba",
    "bimamba_weighted": "BiMamba + weighted loss",
    "bimamba_attention": "BiMamba + attention",
    "bimamba_weighted_attention": "BiMamba + weighted loss + attention",
}

# Reference values from preliminary runs (validation sanity check, not a gate)
CLASSIFIER_ABLATION_REFERENCE = {
    "mlp": {"f1_celltype": 0.9329, "f1_species": 0.9996,
            "f1_disease": 0.9780, "f1_mean3": 0.9570},
    "mamba": {"f1_celltype": 0.9012, "f1_species": 0.9994,
              "f1_disease": 0.9494, "f1_mean3": 0.9311},
    "bimamba": {"f1_celltype": 0.9054, "f1_species": 0.9998,
                "f1_disease": 0.9598, "f1_mean3": 0.9391},
}


# ─── Classifier ablation / cross-validation settings ────────────────────────
@dataclass
class ClassifierAblationConfig:
    k_folds: int = 1
    seeds: List[int] = field(default_factory=lambda: [1234])

    knn_k: int = 10
    knn_metric: str = "euclidean"

    gnn_hidden: int = 64
    gnn_layers: int = 2
    gnn_dropout: float = 0.1
    gat_heads: int = 2

    xgb_n_estimators: int = 300
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1

    bench_warmup: int = 10
    bench_iters: int = 50
    bench_batch_size: int = 1024


# ─── Reproducibility ────────────────────────────────────────────────────────
def set_seed(seed: int = 1234):
    """Set all random seeds for reproducibility."""
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
