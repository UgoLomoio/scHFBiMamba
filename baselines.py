"""
XGBoost baseline for the classifier ablation.

Three multi-class XGBClassifier models (species 2-way, cell type 7-way,
disease 4-way) trained on the shared autoencoder latent. Predicted class
probabilities are assembled into the standard (N, 13) matrix in ALL_CLASSES
column order, so the existing onehot_predict / compute_metrics evaluation
path applies unchanged.
"""
import os
import pickle
import numpy as np
from typing import Dict, List, Tuple

from config import (
    N_CLASSES, SPECIES_CLASSES, CELLTYPE_CLASSES, DISEASE_CLASSES,
    SPECIES_IDX, CELLTYPE_IDX, DISEASE_IDX,
)
from data import parse_label

# Category name -> (class list, column slice)
_CATEGORIES = [
    ('species', SPECIES_CLASSES, SPECIES_IDX),
    ('celltype', CELLTYPE_CLASSES, CELLTYPE_IDX),
    ('disease', DISEASE_CLASSES, DISEASE_IDX),
]


def _labels_to_class_indices(labels_str: List[str]) -> Dict[str, np.ndarray]:
    """Convert label strings to per-category class-index arrays."""
    out = {cat: np.zeros(len(labels_str), dtype=np.int64)
           for cat, _, _ in _CATEGORIES}
    for i, s in enumerate(labels_str):
        species, disease, cell_type = parse_label(s)
        out['species'][i] = SPECIES_CLASSES.index(species)
        out['celltype'][i] = CELLTYPE_CLASSES.index(cell_type)
        out['disease'][i] = DISEASE_CLASSES.index(disease)
    return out


def train_xgboost(train_latent: np.ndarray, train_labels_str: List[str],
                  n_estimators: int = 300, max_depth: int = 6,
                  learning_rate: float = 0.1, seed: int = 1234,
                  output_dir: str = None) -> Dict:
    """
    Train one XGBClassifier per category on the latent features.

    Returns:
        dict mapping category name -> fitted XGBClassifier
    """
    try:
        from xgboost import XGBClassifier
    except ImportError as e:
        raise ImportError(
            "xgboost is required for the XGBoost baseline. "
            "Install with: pip install xgboost"
        ) from e

    y = _labels_to_class_indices(train_labels_str)
    models = {}
    for cat, classes, _ in _CATEGORIES:
        print(f"  XGBoost [{cat}]: {len(classes)} classes, "
              f"{len(train_labels_str)} cells")
        clf = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective='multi:softprob',
            num_class=len(classes),
            tree_method='hist',
            random_state=seed,
            n_jobs=-1,
        )
        clf.fit(train_latent, y[cat])
        models[cat] = clf

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'xgboost_models.pkl'), 'wb') as f:
            pickle.dump(models, f)
        print(f"  Saved XGBoost models to {output_dir}/xgboost_models.pkl")

    return models


def predict_xgboost(models: Dict, latent: np.ndarray) -> np.ndarray:
    """
    Predict (N, 13) probabilities in ALL_CLASSES column order.

    Columns outside each category's slice are filled from that category's
    model; columns for classes the model never saw remain 0.
    """
    n = latent.shape[0]
    probs = np.zeros((n, N_CLASSES), dtype=np.float32)
    for cat, classes, (start, end) in _CATEGORIES:
        clf = models[cat]
        cat_probs = clf.predict_proba(latent)  # (N, n_seen_classes)
        for j, cls_label in enumerate(clf.classes_):
            probs[:, start + int(cls_label)] = cat_probs[:, j]
    return probs


def load_xgboost(output_dir: str) -> Dict:
    """Load previously saved XGBoost models."""
    path = os.path.join(output_dir, 'xgboost_models.pkl')
    with open(path, 'rb') as f:
        return pickle.load(f)


def count_xgboost_params(models: Dict) -> int:
    """Total number of tree nodes across all boosted trees (proxy for params)."""
    total = 0
    for clf in models.values():
        booster = clf.get_booster()
        df = booster.trees_to_dataframe()
        total += len(df)
    return int(total)
