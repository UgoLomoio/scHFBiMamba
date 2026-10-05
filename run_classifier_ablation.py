#!/usr/bin/env python3
"""
Classifier ablation orchestrator.

Holds the shared Dense DAE latent (350-dim) fixed and varies only the
classifier across the arms defined in config.CLASSIFIER_ABLATION_ARMS:
MLP, XGBoost, GCN, GAT, Mamba, BiMamba, BiMamba+weighted loss,
BiMamba+attention, BiMamba+weighted+attention.

For each arm x fold x seed:
  1. Reuse a valid checkpoint when available (no retraining)
  2. Otherwise train on the fold's training split
  3. Evaluate on the same held-out test set
  4. Save per-sample predictions and per-run metrics

Then: aggregate across folds/seeds, benchmark efficiency, and generate the
classifier-ablation and model-efficiency LaTeX tables.

Usage:
  python run_classifier_ablation.py \
    --data_dir data \
    --ablation_dir results/ablation \
    --output_dir results/classifier_ablation \
    --k_folds 1 --seeds 1234

  # 5-fold CV with 3 seeds, only the new arms:
  python run_classifier_ablation.py --k_folds 5 --seeds 1234 1235 1236 \
    --arms bimamba_weighted bimamba_attention bimamba_weighted_attention

  # Only missing experiments (reuse all valid checkpoints):
  python run_classifier_ablation.py --skip_training
"""
import argparse
import os
import json
import time
import numpy as np
import torch

from config import (
    DataConfig, DAEConfig, MambaConfig, MLPConfig, ClassifierAblationConfig,
    set_seed, N_CLASSES, ALL_CLASSES,
    SPECIES_CLASSES, CELLTYPE_CLASSES, DISEASE_CLASSES,
    SPECIES_IDX, CELLTYPE_IDX, DISEASE_IDX,
    CLASSIFIER_ABLATION_ARMS, CLASSIFIER_ABLATION_ORDER,
)
from data import (
    prepare_data, parse_label, stratified_kfold_indices,
)
from losses import compute_metrics, onehot_predict, make_criterion
from models.dae import DenoisingAutoencoder, compute_dae_latent
from models.classifier import BiMambaClassifier
from models.mlp_classifier import MLPClassifier

# Existing M1/M2 checkpoints that are directly reusable for k=1, seed=1234
# (same encoder, same train split, same seed): mlp -> M3, mamba -> M4, bimamba -> M5
LEGACY_CHECKPOINT_MAP = {
    'mlp': 'M3',
    'mamba': 'M4',
    'bimamba': 'M5',
}


def parse_args():
    parser = argparse.ArgumentParser(description='Run classifier ablation')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--ablation_dir', type=str, default='results/ablation',
                        help='Existing ablation dir (shared encoder + M3/M4/M5 checkpoints)')
    parser.add_argument('--output_dir', type=str,
                        default='results/classifier_ablation')
    parser.add_argument('--arms', type=str, nargs='+', default=None,
                        help='Subset of arms (default: all)')
    parser.add_argument('--k_folds', type=int, default=1,
                        help='k-fold CV over the train+val pool (1 = existing split)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1234])
    parser.add_argument('--knn_k', type=int, default=15)
    parser.add_argument('--knn_metric', type=str, default='euclidean')
    # Architecture overrides (same names as run_ablation.py)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--d_state', type=int, default=16)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--num_blocks', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--quick_test', action='store_true')
    # Flow control
    parser.add_argument('--skip_training', action='store_true',
                        help='Only evaluate existing valid checkpoints; '
                             'arms without one are reported as missing')
    parser.add_argument('--force_retrain', action='store_true',
                        help='Ignore existing checkpoints and retrain')
    parser.add_argument('--skip_benchmark', action='store_true')
    parser.add_argument('--skip_tables', action='store_true')
    return parser.parse_args()


# ─── Integrity checks ────────────────────────────────────────────────────────

class IntegrityReport:
    """Collects integrity warnings; written to JSON at the end."""

    def __init__(self):
        self.issues = []

    def check(self, condition: bool, message: str):
        if not condition:
            print(f"  INTEGRITY WARNING: {message}")
            self.issues.append(message)

    def save(self, output_dir: str):
        path = os.path.join(output_dir, 'integrity_report.json')
        with open(path, 'w') as f:
            json.dump({'n_issues': len(self.issues), 'issues': self.issues},
                      f, indent=2)
        print(f"  Integrity report: {len(self.issues)} issue(s) -> {path}")


def check_array(report: IntegrityReport, arr: np.ndarray, name: str):
    """NaN/Inf and shape sanity for an array."""
    report.check(not np.isnan(arr).any(), f"{name} contains NaN")
    report.check(not np.isinf(arr).any(), f"{name} contains Inf")


def try_load_classifier_checkpoint(model: torch.nn.Module, path: str,
                                   device: torch.device,
                                   report: IntegrityReport, tag: str) -> bool:
    """Load a classifier checkpoint; verify keys and NaN-free weights."""
    if not os.path.exists(path):
        return False
    try:
        state = torch.load(path, map_location=device)
        model.load_state_dict(state)
    except Exception as e:
        report.check(False, f"{tag}: invalid checkpoint {path} ({e})")
        return False
    for name, p in model.named_parameters():
        if torch.isnan(p).any():
            report.check(False, f"{tag}: NaN in parameter {name} of {path}")
            return False
    print(f"  Reused checkpoint: {path}")
    return True


# ─── Arm construction ────────────────────────────────────────────────────────

def build_arm_classifier(arm_id: str, args):
    """Build a torch classifier (mamba/mlp) for an ablation arm."""
    spec = CLASSIFIER_ABLATION_ARMS[arm_id]
    if spec['classifier_type'] == 'mamba':
        cfg = MambaConfig(
            seq_len=350,
            d_model=args.d_model,
            d_state=args.d_state,
            n_layers=args.n_layers,
            num_blocks=args.num_blocks,
            bidirectional=(spec['direction'] == 'bidirectional'),
            pool=spec.get('pool', 'mean_max'),
            loss=spec.get('loss', 'bce'),
            lr=args.lr,
            max_epochs=args.epochs,
            batch_size=args.batch_size,
            use_amp=args.use_amp,
        )
        if args.quick_test:
            cfg.max_epochs = 5
            cfg.batch_size = 256
        return BiMambaClassifier(cfg), cfg
    cfg = MLPConfig(lr=args.lr, max_epochs=args.epochs,
                    batch_size=args.batch_size, use_amp=args.use_amp)
    if args.quick_test:
        cfg.max_epochs = 5
        cfg.batch_size = 256
    return MLPClassifier(cfg), cfg


# ─── Evaluation helpers ──────────────────────────────────────────────────────

def probs_to_predictions_csv(probs: np.ndarray, labels_str: list,
                             path: str):
    """Save per-sample predictions: true label, 13 probabilities, prediction."""
    import pandas as pd
    binary = onehot_predict(probs)
    pred_str = []
    for row in binary:
        sp = SPECIES_CLASSES[int(np.argmax(row[SPECIES_IDX[0]:SPECIES_IDX[1]]))]
        ct = CELLTYPE_CLASSES[int(np.argmax(row[CELLTYPE_IDX[0]:CELLTYPE_IDX[1]]))]
        dis = DISEASE_CLASSES[int(np.argmax(row[DISEASE_IDX[0]:DISEASE_IDX[1]]))]
        pred_str.append(f"{sp}-{dis}_{ct}")
    df = pd.DataFrame({'true_label': labels_str, 'pred_label': pred_str})
    for i, name in enumerate(ALL_CLASSES):
        df[f'prob_{name}'] = probs[:, i]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def evaluate_probs(probs: np.ndarray, test_labels: np.ndarray,
                   test_labels_str: list, report: IntegrityReport,
                   tag: str) -> dict:
    """Integrity-check probabilities and compute the metric bundle."""
    check_array(report, probs, f"{tag} probs")
    report.check(probs.shape[1] == N_CLASSES,
                 f"{tag}: probs have {probs.shape[1]} columns, expected {N_CLASSES}")
    report.check(probs.shape[0] == test_labels.shape[0],
                 f"{tag}: probs/labels row mismatch "
                 f"({probs.shape[0]} vs {test_labels.shape[0]})")
    binary = onehot_predict(probs)
    metrics = compute_metrics(test_labels, binary)
    # Arithmetic mean of the three category macro-F1 scores
    metrics['f1_mean3'] = float(np.mean([
        metrics['f1_species'], metrics['f1_celltype'], metrics['f1_disease']]))
    return metrics


# ─── Graph arm training (transductive, features only) ────────────────────────

def train_graph_model(arm_id: str, features: np.ndarray, labels: np.ndarray,
                      train_idx: np.ndarray, val_idx: np.ndarray,
                      adj, device: torch.device, args,
                      output_dir: str, seed: int):
    """
    Train a GCN/GAT on the pooled graph. Loss on train nodes only;
    early stopping on val nodes. Test nodes contribute features/topology
    but never labels.
    """
    from models.gcn_classifier import GCNClassifier
    from models.gat_classifier import GATClassifier
    from graph import normalize_adjacency_gcn, adjacency_to_edge_index

    spec = CLASSIFIER_ABLATION_ARMS[arm_id]
    abl_cfg = ClassifierAblationConfig(knn_k=args.knn_k,
                                       knn_metric=args.knn_metric)
    if spec['graph_model'] == 'gcn':
        model = GCNClassifier(input_dim=features.shape[1],
                              hidden_dim=abl_cfg.gnn_hidden,
                              n_layers=abl_cfg.gnn_layers,
                              dropout=abl_cfg.gnn_dropout).to(device)
        graph_input = normalize_adjacency_gcn(adj).to(device)
    else:
        model = GATClassifier(input_dim=features.shape[1],
                              hidden_dim=abl_cfg.gnn_hidden,
                              n_layers=abl_cfg.gnn_layers,
                              n_heads=abl_cfg.gat_heads,
                              dropout=abl_cfg.gnn_dropout).to(device)
        graph_input = adjacency_to_edge_index(adj).to(device)

    x = torch.from_numpy(np.asarray(features, dtype=np.float32)).to(device)
    y = torch.from_numpy(np.asarray(labels, dtype=np.float32)).to(device)
    train_t = torch.from_numpy(train_idx).to(device)
    val_t = torch.from_numpy(val_idx).to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=25, min_lr=1e-7)

    max_epochs = 5 if args.quick_test else args.epochs
    best_val = float('inf')
    patience = 0
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, 'classifier_best.pt')

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(x, graph_input)
        loss = criterion(logits[train_t], y[train_t])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(x, graph_input)
            val_loss = criterion(val_logits[val_t], y[val_t]).item()
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            patience = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience += 1
            if patience >= 50:
                print(f"  Early stopping at epoch {epoch}")
                break
        if epoch % 25 == 0 or epoch == 1:
            print(f"  Epoch {epoch}/{max_epochs} | train {loss.item():.6f} | "
                  f"val {val_loss:.6f}")

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model, graph_input


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    report = IntegrityReport()

    arm_ids = args.arms or list(CLASSIFIER_ABLATION_ORDER)
    for arm in arm_ids:
        if arm not in CLASSIFIER_ABLATION_ARMS:
            raise ValueError(f"Unknown arm '{arm}'. "
                             f"Choose from {list(CLASSIFIER_ABLATION_ARMS)}")

    # ─── Data ────────────────────────────────────────────────────────────
    data_cfg = DataConfig(
        train_path=os.path.join(args.data_dir, 'RANDOMIZED_train_set_p_80.csv'),
        val_path=os.path.join(args.data_dir, 'RANDOMIZED_val_set_p_20.csv'),
        test_path=os.path.join(args.data_dir, '240519_test_set_only_n2_samples.csv'),
        train_labels_path=os.path.join(args.data_dir, 'RANDOMIZED_train_set_labels_p_80.csv'),
        val_labels_path=os.path.join(args.data_dir, 'RANDOMIZED_val_set_labels_p_20.csv'),
        test_labels_path=os.path.join(args.data_dir, '240519_test_set_only_n2_samples_labels.csv'),
        cache_dir=os.path.join(args.data_dir, 'cache'),
    )
    data = prepare_data(data_cfg, device)

    # Label integrity: all label strings must parse to known classes
    for split in ['train', 'val', 'test']:
        col = {'train': 'train_labels', 'val': 'val_labels',
               'test': 'test_labels'}[split]
        labels = data[col]
        check_array(report, labels, f"{split} labels")
        report.check(labels.shape[1] == N_CLASSES,
                     f"{split} labels have {labels.shape[1]} columns")
    from data import load_label_strings
    train_labels_str = load_label_strings(data_cfg.train_labels_path)
    val_labels_str = load_label_strings(data_cfg.val_labels_path)
    test_labels_str = load_label_strings(data_cfg.test_labels_path)
    train_classes = set(train_labels_str) | set(val_labels_str)
    unseen = set(test_labels_str) - train_classes
    report.check(len(unseen) == 0,
                 f"test label combinations unseen in train/val: {sorted(unseen)}")

    # ─── Shared Dense DAE encoder (reuse; train only if missing) ─────────
    shared_dir = os.path.join(args.ablation_dir, 'shared_dense_dae')
    enc_ckpt = os.path.join(shared_dir, 'encoder_best.pt')
    latent_paths = {s: os.path.join(shared_dir, f'{s}_latent.npy')
                    for s in ['train', 'val', 'test']}
    have_latents = all(os.path.exists(p) for p in latent_paths.values())

    if have_latents:
        print(f"Loading cached latents from {shared_dir}")
        train_latent = np.load(latent_paths['train'])
        val_latent = np.load(latent_paths['val'])
        test_latent = np.load(latent_paths['test'])
    else:
        encoder = DenoisingAutoencoder(
            input_dim=DAEConfig().input_dim,
            encoder_neurons=DAEConfig().encoder_neurons,
            decoder_neurons=DAEConfig().decoder_neurons).to(device)
        if os.path.exists(enc_ckpt):
            print(f"Loading shared encoder from {enc_ckpt}")
            encoder.load_state_dict(torch.load(enc_ckpt, map_location=device))
        else:
            if args.skip_training:
                raise FileNotFoundError(
                    f"--skip_training but no shared encoder/latents at {shared_dir}. "
                    f"Run run_ablation.py first (at least the encoder stage).")
            from train import train_dense_dae
            encoder = train_dense_dae(DAEConfig(), data, device, shared_dir)
        from train import compute_and_save_latents
        train_latent, val_latent, test_latent = compute_and_save_latents(
            encoder, 'dense_dae', data, device, shared_dir)

    for name, lat in [('train', train_latent), ('val', val_latent),
                      ('test', test_latent)]:
        check_array(report, lat, f"{name} latent")
        report.check(lat.shape[0] == data[f'{name}_labels'].shape[0],
                     f"{name} latent/label row mismatch")

    n_train, n_val = len(train_labels_str), len(val_labels_str)
    pooled_latent = np.concatenate([train_latent, val_latent], axis=0)
    pooled_labels = np.concatenate([data['train_labels'], data['val_labels']], axis=0)
    pooled_labels_str = train_labels_str + val_labels_str

    # ─── Graph for GCN/GAT arms (built once, transductive features only) ──
    graph_adj = None
    graph_n_edges = None
    if any(CLASSIFIER_ABLATION_ARMS[a]['kind'] == 'graph' for a in arm_ids):
        from graph import build_knn_graph
        graph_dir = os.path.join(args.output_dir, 'graph')
        os.makedirs(graph_dir, exist_ok=True)
        graph_path = os.path.join(
            graph_dir, f'knn_k{args.knn_k}_{args.knn_metric}.npz')
        import scipy.sparse as sp
        if os.path.exists(graph_path):
            print(f"Loading cached kNN graph from {graph_path}")
            graph_adj = sp.load_npz(graph_path)
        else:
            all_latent = np.concatenate([pooled_latent, test_latent], axis=0)
            graph_adj = build_knn_graph(all_latent, k=args.knn_k,
                                        metric=args.knn_metric)
            sp.save_npz(graph_path, graph_adj)
        graph_n_edges = graph_adj.nnz

    # ─── Run arms x seeds x folds ─────────────────────────────────────────
    all_run_metrics = []  # list of per-run dicts

    for arm_id in arm_ids:
        spec = CLASSIFIER_ABLATION_ARMS[arm_id]
        for seed in args.seeds:
            set_seed(seed)

            # Fold definitions
            if args.k_folds == 1:
                folds = [(np.arange(n_train), n_train + np.arange(n_val))]
            else:
                folds = stratified_kfold_indices(pooled_labels_str,
                                                 args.k_folds, seed=seed)

            for fold_i, (tr_idx, va_idx) in enumerate(folds):
                tag = f"{arm_id}/fold{fold_i}_seed{seed}"
                run_dir = os.path.join(args.output_dir, arm_id,
                                       f'fold{fold_i}_seed{seed}')
                metrics_path = os.path.join(run_dir, 'run_metrics.json')

                if os.path.exists(metrics_path) and not args.force_retrain:
                    print(f"\n=== {tag}: metrics exist, skipping (reuse) ===")
                    with open(metrics_path) as f:
                        all_run_metrics.append(json.load(f))
                    continue

                print(f"\n{'=' * 60}")
                print(f"ARM {tag}: {spec['description']}")
                print(f"{'=' * 60}")

                fold_train_labels = pooled_labels[tr_idx]
                fold_train_str = [pooled_labels_str[i] for i in tr_idx]

                probs = None

                # ── torch arms ──────────────────────────────────────────
                if spec['kind'] == 'torch':
                    model, clf_cfg = build_arm_classifier(arm_id, args)
                    model = model.to(device)
                    ckpt_path = os.path.join(run_dir, 'classifier_best.pt')

                    loaded = False
                    if not args.force_retrain:
                        # 1) arm-local checkpoint
                        loaded = try_load_classifier_checkpoint(
                            model, ckpt_path, device, report, tag)
                        # 2) legacy M1/M2 checkpoint (k=1, seed=1234 only)
                        if (not loaded and args.k_folds == 1 and seed == 1234
                                and arm_id in LEGACY_CHECKPOINT_MAP):
                            legacy = os.path.join(
                                args.ablation_dir,
                                LEGACY_CHECKPOINT_MAP[arm_id],
                                'classifier_best.pt')
                            loaded = try_load_classifier_checkpoint(
                                model, legacy, device, report,
                                f"{tag} (legacy {LEGACY_CHECKPOINT_MAP[arm_id]})")

                    if not loaded:
                        if args.skip_training:
                            print(f"  {tag}: no valid checkpoint, "
                                  f"--skip_training set -> missing")
                            continue
                        criterion = make_criterion(
                            spec.get('loss', 'bce'), fold_train_labels, device)
                        from train import train_classifier
                        model = train_classifier(
                            clf_cfg, data, device, run_dir,
                            pooled_latent[tr_idx], pooled_latent[va_idx],
                            test_latent,
                            classifier_type=spec['classifier_type'],
                            criterion=criterion,
                            train_labels=fold_train_labels,
                            val_labels=pooled_labels[va_idx])

                    # Inference (logits -> sigmoid probabilities)
                    model.eval()
                    from data import create_dataloader
                    loader = create_dataloader(
                        test_latent, data['test_labels'],
                        args.batch_size, shuffle=False, device=device,
                        num_workers=4, pin_memory=True)
                    logits_all = []
                    with torch.no_grad():
                        for xb, _ in loader:
                            xb = xb.to(device, non_blocking=True)
                            logits_all.append(model(xb).cpu().numpy())
                    probs = 1 / (1 + np.exp(-np.concatenate(logits_all, axis=0)))

                # ── graph arms ──────────────────────────────────────────
                elif spec['kind'] == 'graph':
                    all_latent = np.concatenate([pooled_latent, test_latent],
                                                axis=0)
                    n_all = all_latent.shape[0]
                    all_labels = np.concatenate(
                        [pooled_labels, np.zeros_like(data['test_labels'])],
                        axis=0)
                    test_idx = np.arange(n_train + n_val, n_all)

                    model = None
                    ckpt_path = os.path.join(run_dir, 'classifier_best.pt')
                    if not args.force_retrain and os.path.exists(ckpt_path):
                        from models.gcn_classifier import GCNClassifier
                        from models.gat_classifier import GATClassifier
                        from graph import (normalize_adjacency_gcn,
                                           adjacency_to_edge_index)
                        cls = GCNClassifier if spec['graph_model'] == 'gcn' \
                            else GATClassifier
                        model = cls(input_dim=all_latent.shape[1]).to(device)
                        graph_input = (normalize_adjacency_gcn(graph_adj)
                                       if spec['graph_model'] == 'gcn'
                                       else adjacency_to_edge_index(graph_adj)
                                       ).to(device)
                        loaded = try_load_classifier_checkpoint(
                            model, ckpt_path, device, report, tag)
                        if not loaded:
                            model = None
                    if model is None:
                        if args.skip_training:
                            print(f"  {tag}: no valid checkpoint, "
                                  f"--skip_training set -> missing")
                            continue
                        model, graph_input = train_graph_model(
                            arm_id, all_latent, all_labels, tr_idx, va_idx,
                            graph_adj, device, args, run_dir, seed)

                    model.eval()
                    x_all = torch.from_numpy(all_latent.astype(np.float32)).to(device)
                    with torch.no_grad():
                        logits = model(x_all, graph_input)
                    probs = 1 / (1 + np.exp(-logits[test_idx].cpu().numpy()))

                # ── xgboost ─────────────────────────────────────────────
                elif spec['kind'] == 'xgboost':
                    import baselines as xgb_mod
                    model_path = os.path.join(run_dir, 'xgboost_models.pkl')
                    if os.path.exists(model_path) and not args.force_retrain:
                        print(f"  Reused checkpoint: {model_path}")
                        models = xgb_mod.load_xgboost(run_dir)
                    else:
                        if args.skip_training:
                            print(f"  {tag}: no valid checkpoint, "
                                  f"--skip_training set -> missing")
                            continue
                        models = xgb_mod.train_xgboost(
                            pooled_latent[tr_idx], fold_train_str,
                            seed=seed, output_dir=run_dir)
                    probs = xgb_mod.predict_xgboost(models, test_latent)

                # ── evaluate + save ─────────────────────────────────────
                metrics = evaluate_probs(probs, data['test_labels'],
                                         test_labels_str, report, tag)
                metrics.update({
                    'arm': arm_id, 'fold': fold_i, 'seed': seed,
                    'n_train': int(len(tr_idx)), 'n_val': int(len(va_idx)),
                })
                os.makedirs(run_dir, exist_ok=True)
                with open(metrics_path, 'w') as f:
                    json.dump(metrics, f, indent=2)
                np.save(os.path.join(run_dir, 'test_probs.npy'), probs)
                probs_to_predictions_csv(
                    probs, test_labels_str,
                    os.path.join(args.output_dir, 'predictions',
                                 f'predictions_{arm_id}_fold{fold_i}_seed{seed}.csv'))
                all_run_metrics.append(metrics)
                print(f"  {tag}: mean3 F1 = {metrics['f1_mean3']:.4f} "
                      f"(ct={metrics['f1_celltype']:.4f} "
                      f"sp={metrics['f1_species']:.4f} "
                      f"dis={metrics['f1_disease']:.4f})")

    # ─── Aggregate across folds/seeds ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("Aggregating results across folds and seeds")
    print("=" * 60)
    agg = {}
    for arm in arm_ids:
        runs = [m for m in all_run_metrics if m['arm'] == arm]
        entry = {'n_runs': len(runs)}
        for key in ['f1_celltype', 'f1_species', 'f1_disease', 'f1_mean3']:
            vals = [m[key] for m in runs if key in m]
            entry[key] = (float(np.mean(vals)) if vals else None,
                          float(np.std(vals)) if len(vals) > 1 else None)
        agg[arm] = entry
        if runs:
            print(f"  {arm}: n={len(runs)}, mean3 F1 = "
                  f"{entry['f1_mean3'][0]:.4f}")

    with open(os.path.join(args.output_dir, 'aggregated_metrics.json'), 'w') as f:
        json.dump(agg, f, indent=2)
    import pandas as pd
    rows = [{'arm': a, 'n_runs': e['n_runs'],
             **{k: v[0] for k, v in e.items() if k != 'n_runs'},
             **{f'{k}_std': v[1] for k, v in e.items() if k != 'n_runs'}}
            for a, e in agg.items()]
    pd.DataFrame(rows).to_csv(
        os.path.join(args.output_dir, 'aggregated_metrics.csv'), index=False)

    # ─── Efficiency benchmark ─────────────────────────────────────────────
    eff = {}
    if not args.skip_benchmark:
        from benchmark import (count_parameters, estimate_flops_torch,
                               benchmark_torch_model, measure_inference_time)
        abl_cfg = ClassifierAblationConfig()
        bench_bs = abl_cfg.bench_batch_size
        print("\n" + "=" * 60)
        print(f"Efficiency benchmark (batch={bench_bs}, warmup="
              f"{abl_cfg.bench_warmup}, iters={abl_cfg.bench_iters}, "
              f"device={device})")
        print("=" * 60)
        for arm in arm_ids:
            spec = CLASSIFIER_ABLATION_ARMS[arm]
            entry = {'params': None, 'flops': None,
                     'time_mean_ms': None, 'time_std_ms': None, 'note': None}
            try:
                if spec['kind'] == 'torch':
                    model, _ = build_arm_classifier(arm, args)
                    model = model.to(device).eval()
                    entry['params'] = count_parameters(model)
                    kind = 'mlp' if spec['classifier_type'] == 'mlp' else 'sequence'
                    entry['flops'] = estimate_flops_torch(
                        model, kind, seq_len=350)
                    if kind == 'sequence':
                        entry['note'] = ('FLOPs: analytical estimate '
                                         '(2 x params x sequence length)')
                    t = benchmark_torch_model(
                        model, (bench_bs, 350), device,
                        warmup=abl_cfg.bench_warmup, iters=abl_cfg.bench_iters,
                        use_amp=args.use_amp)
                    entry['time_mean_ms'] = t['mean_ms']
                    entry['time_std_ms'] = t['std_ms']
                elif spec['kind'] == 'graph':
                    from models.gcn_classifier import GCNClassifier
                    from models.gat_classifier import GATClassifier
                    cls = GCNClassifier if spec['graph_model'] == 'gcn' \
                        else GATClassifier
                    model = cls(input_dim=350).to(device).eval()
                    entry['params'] = count_parameters(model)
                    n_nodes = (pooled_latent.shape[0] + test_latent.shape[0])
                    entry['flops'] = estimate_flops_torch(
                        model, spec['graph_model'], n_nodes=n_nodes,
                        n_edges=graph_n_edges or 0,
                        hidden_dim=abl_cfg.gnn_hidden)
                    entry['note'] = ('FLOPs scale with graph edges; full-graph '
                                     'transductive pass normalized per '
                                     f'{bench_bs} cells')
                    if graph_adj is not None:
                        from graph import (normalize_adjacency_gcn,
                                           adjacency_to_edge_index)
                        gi = (normalize_adjacency_gcn(graph_adj)
                              if spec['graph_model'] == 'gcn'
                              else adjacency_to_edge_index(graph_adj)).to(device)
                        x_all = torch.from_numpy(
                            np.concatenate([pooled_latent, test_latent])
                            .astype(np.float32)).to(device)
                        t = measure_inference_time(
                            lambda: model(x_all, gi), device,
                            warmup=abl_cfg.bench_warmup,
                            iters=abl_cfg.bench_iters)
                        scale = bench_bs / x_all.shape[0]
                        entry['time_mean_ms'] = t['mean_ms'] * scale
                        entry['time_std_ms'] = t['std_ms'] * scale
                elif spec['kind'] == 'xgboost':
                    entry['note'] = ('CPU inference; FLOPs not meaningful '
                                     'for boosted trees; params = total '
                                     'tree nodes')
                    xgb_dir = os.path.join(args.output_dir, arm,
                                           f'fold0_seed{args.seeds[0]}')
                    model_path = os.path.join(xgb_dir, 'xgboost_models.pkl')
                    if os.path.exists(model_path):
                        import baselines as xgb_mod
                        models = xgb_mod.load_xgboost(xgb_dir)
                        entry['params'] = xgb_mod.count_xgboost_params(models)
                        xb = test_latent[:bench_bs]
                        t0 = time.perf_counter()
                        for _ in range(abl_cfg.bench_warmup):
                            xgb_mod.predict_xgboost(models, xb)
                        times = []
                        for _ in range(abl_cfg.bench_iters):
                            t0 = time.perf_counter()
                            xgb_mod.predict_xgboost(models, xb)
                            times.append((time.perf_counter() - t0) * 1000.0)
                        entry['time_mean_ms'] = float(np.mean(times))
                        entry['time_std_ms'] = float(np.std(times))
            except Exception as e:
                print(f"  Benchmark failed for {arm}: {e}")
            eff[arm] = entry
            print(f"  {arm}: params={entry['params']}, "
                  f"flops={entry['flops']}, "
                  f"time={entry['time_mean_ms']} ms")

        with open(os.path.join(args.output_dir, 'efficiency.json'), 'w') as f:
            json.dump(eff, f, indent=2)

    # ─── LaTeX tables ─────────────────────────────────────────────────────
    if not args.skip_tables:
        from stats.latex_tables import (write_classifier_ablation_table,
                                        write_efficiency_table,
                                        check_reference_values)
        tables_dir = os.path.join(args.output_dir, 'tables')
        write_classifier_ablation_table(
            agg, os.path.join(tables_dir, 'table_classifier_ablation.tex'),
            show_std=True)
        if eff:
            write_efficiency_table(
                eff, os.path.join(tables_dir, 'table_model_efficiency.tex'))
        print("\nReference-value check (preliminary runs):")
        check_reference_values(agg)

    report.save(args.output_dir)
    print(f"\nClassifier ablation complete. Results in {args.output_dir}/")


if __name__ == '__main__':
    main()
