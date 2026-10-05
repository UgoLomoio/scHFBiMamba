"""
Loss functions and metrics for HF Mamba pipeline.
PyTorch re-implementation of the paper's macro F1-loss and metric functions.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple

from config import N_CLASSES, SPECIES_IDX, CELLTYPE_IDX, DISEASE_IDX

EPS = 1e-7


# ─── Macro F1 Loss (identical to paper) ──────────────────────────────────────

def macro_f1_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Macro F1-loss as defined in the paper.
    Computes F1 per class, then averages, returns 1 - mean_F1.

    Args:
        y_true: (B, 13) ground-truth multi-hot labels
        y_pred: (B, 13) predicted probabilities (sigmoid output)

    Returns:
        loss: scalar, 1 - mean(macro F1 across 13 classes)
    """
    y_true = y_true.float()
    y_pred = y_pred.float()

    # True positives, false positives, false negatives per class
    tp = (y_true * y_pred).sum(dim=0)
    fp = y_pred.sum(dim=0) - tp
    fn = y_true.sum(dim=0) - tp

    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)

    f1_scores = 2 * (precision * recall) / (precision + recall + EPS)
    cost = 1 - f1_scores
    macro_cost = cost.mean()

    return macro_cost


# ─── Class-weighted BCE (weighted-loss ablation) ────────────────────────────

def compute_pos_weight(y_true: np.ndarray) -> torch.Tensor:
    """
    Per-class pos_weight for BCEWithLogitsLoss: n_neg / n_pos for each of the
    13 tasks, computed from the training-fold multi-hot labels only.

    Args:
        y_true: (N, 13) multi-hot training labels

    Returns:
        (13,) float32 tensor of positive weights
    """
    y = np.asarray(y_true, dtype=np.float64)
    n_pos = y.sum(axis=0)
    n_neg = y.shape[0] - n_pos
    # Classes absent from the fold get weight 1.0 (no up-weighting possible)
    with np.errstate(divide='ignore', invalid='ignore'):
        w = np.where(n_pos > 0, n_neg / np.maximum(n_pos, 1), 1.0)
    return torch.tensor(w, dtype=torch.float32)


def make_criterion(loss_name: str, y_train: np.ndarray = None,
                   device: torch.device = None) -> nn.Module:
    """
    Build the training criterion for a classifier arm.

    Args:
        loss_name: "bce" (default) or "bce_weighted"
        y_train: (N, 13) training-fold labels (required for "bce_weighted")
        device: target device

    Returns:
        criterion module
    """
    if loss_name == "bce_weighted":
        if y_train is None:
            raise ValueError("bce_weighted requires y_train to compute pos_weight")
        pos_weight = compute_pos_weight(y_train)
        if device is not None:
            pos_weight = pos_weight.to(device)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return nn.BCEWithLogitsLoss()


# ─── Metric computation ──────────────────────────────────────────────────────

def compute_f1_per_class(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Compute F1 score for each of the 13 classes. Returns (13,) tensor."""
    y_true = y_true.float()
    y_pred = y_pred.float()

    tp = (y_true * y_pred).sum(dim=0)
    fp = y_pred.sum(dim=0) - tp
    fn = y_true.sum(dim=0) - tp

    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)

    f1 = 2 * (precision * recall) / (precision + recall + EPS)
    return f1


def compute_precision_per_class(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Compute precision for each of the 13 classes. Returns (13,) tensor."""
    y_true = y_true.float()
    y_pred = y_pred.float()

    tp = (y_true * y_pred).sum(dim=0)
    fp = y_pred.sum(dim=0) - tp

    return tp / (tp + fp + EPS)


def compute_recall_per_class(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Compute recall for each of the 13 classes. Returns (13,) tensor."""
    y_true = y_true.float()
    y_pred = y_pred.float()

    tp = (y_true * y_pred).sum(dim=0)
    fn = y_true.sum(dim=0) - tp

    return tp / (tp + fn + EPS)


def compute_metrics(y_true: np.ndarray, y_pred_binary: np.ndarray) -> Dict[str, float]:
    """
    Compute all metrics from binary predictions.
    Matches paper's metric groups: mean, species, celltype, disease.

    Args:
        y_true: (N, 13) ground-truth binary
        y_pred_binary: (N, 13) predicted binary (after thresholding)

    Returns:
        dict with f1, precision, recall for each group
    """
    y_true_t = torch.from_numpy(y_true).float()
    y_pred_t = torch.from_numpy(y_pred_binary).float()

    f1 = compute_f1_per_class(y_true_t, y_pred_t).numpy()
    precision = compute_precision_per_class(y_true_t, y_pred_t).numpy()
    recall = compute_recall_per_class(y_true_t, y_pred_t).numpy()

    return {
        'f1_mean': float(f1.mean()),
        'f1_species': float(f1[SPECIES_IDX[0]:SPECIES_IDX[1]].mean()),
        'f1_celltype': float(f1[CELLTYPE_IDX[0]:CELLTYPE_IDX[1]].mean()),
        'f1_disease': float(f1[DISEASE_IDX[0]:DISEASE_IDX[1]].mean()),
        'precision_mean': float(precision.mean()),
        'precision_species': float(precision[SPECIES_IDX[0]:SPECIES_IDX[1]].mean()),
        'precision_celltype': float(precision[CELLTYPE_IDX[0]:CELLTYPE_IDX[1]].mean()),
        'precision_disease': float(precision[DISEASE_IDX[0]:DISEASE_IDX[1]].mean()),
        'recall_mean': float(recall.mean()),
        'recall_species': float(recall[SPECIES_IDX[0]:SPECIES_IDX[1]].mean()),
        'recall_celltype': float(recall[CELLTYPE_IDX[0]:CELLTYPE_IDX[1]].mean()),
        'recall_disease': float(recall[DISEASE_IDX[0]:DISEASE_IDX[1]].mean()),
        'f1_per_class': f1.tolist(),
        'precision_per_class': precision.tolist(),
        'recall_per_class': recall.tolist(),
    }


# ─── Prediction utilities ────────────────────────────────────────────────────

def threshold_predictions(probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Threshold sigmoid probabilities to binary predictions.
    Matches paper: values above threshold -> 1, else 0.
    """
    return (probs >= threshold).astype(np.float32)


def onehot_predict(probs: np.ndarray) -> np.ndarray:
    """
    Get top-3 predictions per cell (one per category: species, celltype, disease).
    Matches paper's onehot_predict function.

    For each category group (species[0:2], celltype[2:9], disease[9:13]),
    select the class with highest probability.

    Args:
        probs: (N, 13) sigmoid probabilities

    Returns:
        (N, 13) binary array with exactly 3 ones per row
    """
    y_bin = np.zeros_like(probs)
    # Species: argmax of [0:2]
    species_idx = np.argmax(probs[:, 0:2], axis=1)
    y_bin[np.arange(len(probs)), species_idx] = 1.0
    # Cell type: argmax of [2:9]
    ct_idx = np.argmax(probs[:, 2:9], axis=1) + 2
    y_bin[np.arange(len(probs)), ct_idx] = 1.0
    # Disease: argmax of [9:13]
    dis_idx = np.argmax(probs[:, 9:13], axis=1) + 9
    y_bin[np.arange(len(probs)), dis_idx] = 1.0
    return y_bin


def correct_classification_rate(y_true: np.ndarray, y_pred_binary: np.ndarray) -> np.ndarray:
    """
    Per-class correct classification rate.
    For each class, fraction of cells with that label that are correctly predicted.
    """
    rates = np.zeros(N_CLASSES)
    for c in range(N_CLASSES):
        mask = y_true[:, c] == 1
        if mask.sum() > 0:
            rates[c] = (y_pred_binary[mask, c] == 1).mean()
    return rates


# ─── Confusion matrix ────────────────────────────────────────────────────────

def confusion_matrix_category(y_true: np.ndarray, y_pred_binary: np.ndarray,
                              cat_start: int, cat_end: int) -> np.ndarray:
    """
    Compute confusion matrix for a category (species, celltype, or disease).
    Each cell belongs to exactly one class within the category.

    Args:
        y_true: (N, 13) multi-hot
        y_pred_binary: (N, 13) binary predictions
        cat_start: start index of category
        cat_end: end index (exclusive)

    Returns:
        (n_classes, n_classes) confusion matrix
    """
    n_classes = cat_end - cat_start
    cm = np.zeros((n_classes, n_classes), dtype=int)

    # Get true and predicted class indices within category
    true_idx = np.argmax(y_true[:, cat_start:cat_end], axis=1)
    pred_idx = np.argmax(y_pred_binary[:, cat_start:cat_end], axis=1)

    for t, p in zip(true_idx, pred_idx):
        cm[t, p] += 1

    return cm
