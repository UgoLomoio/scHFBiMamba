#!/usr/bin/env python3
"""
Orchestration script for the full ablation matrix (M1-M4).

  M0: Dense DAE + MLP (baseline)
  M1: Dense DAE + Mamba (UniDir)
  M2: Dense DAE + BiMamba (BiDir)
  M3: BiMamba AE + MLP
  M4: BiMamba AE + Mamba (UniDir)
  M5: BiMamba AE + BiMamba (BiDir)

Encoders are shared: Dense DAE between M1/M2, BiMamba AE between M3/M4.
After training, runs SHAP and IG attribution via composite models,
and saves embeddings for UMAP.

Usage:
  python run_ablation.py \
    --data_dir data \
    --output_dir results/ablation \
    --go_terms_path data/Suppl_Table_8_TP_GO_ID_per_cell_type.xlsx \
    --epochs 500 --batch_size 1024 --use_amp

  # Quick test:
  python run_ablation.py --epochs 5 --batch_size 256 --quick_test

  # Only specific models:
  python run_ablation.py --models M1 M3
"""
import argparse
import os
import json
import time
import numpy as np
import torch

from config import (
    DataConfig, DAEConfig, BiMambaAEConfig, MambaConfig, MLPConfig,
    set_seed, N_GENES, N_CLASSES, ABLATION_MODELS, ABLATION_ATTRIBUTION,
    get_config_for_ablation
)
from data import prepare_data, create_dataloader, sample_balanced_test_subset
from losses import compute_metrics, onehot_predict
from models.dae import DenoisingAutoencoder, compute_dae_latent
from models.bimamba_ae import BiMambaAutoencoder, compute_bimamba_ae_latent
from models.classifier import BiMambaClassifier
from models.mlp_classifier import MLPClassifier
from models.composite_model import CompositeModel, compute_latent_and_features


def parse_args():
    parser = argparse.ArgumentParser(description='Run full ablation matrix')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--output_dir', type=str, default='results/ablation')
    parser.add_argument('--go_terms_path', type=str,
                        default='data/Suppl_Table_8_TP_GO_ID_per_cell_type.xlsx')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--encoder_epochs', type=int, default=500)
    parser.add_argument('--encoder_batch_size', type=int, default=1024)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--d_state', type=int, default=16)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--num_blocks', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--quick_test', action='store_true',
                        help='Quick test: 5 epochs, small batch')
    parser.add_argument('--skip_training', action='store_true',
                        help='Skip training, only run attribution on existing models')
    parser.add_argument('--skip_attribution', action='store_true',
                        help='Skip attribution, only train models')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                        help='Subset of models to run (e.g. M1 M3). Default: all.')
    parser.add_argument('--attributions', type=str, nargs='+', default="shap",
                        help='Subset of attribution methods (e.g. shap). Default: all.')
    return parser.parse_args()


def _apply_cli_overrides(encoder_cfg, classifier_cfg, args):
    """Apply CLI argument overrides to encoder and classifier configs."""
    if hasattr(encoder_cfg, 'd_model'):
        encoder_cfg.d_model = args.d_model
        encoder_cfg.d_state = args.d_state
        encoder_cfg.n_layers = args.n_layers
        encoder_cfg.num_blocks = args.num_blocks
    encoder_cfg.lr = args.lr
    encoder_cfg.seed = args.seed
    encoder_cfg.use_amp = args.use_amp
    encoder_cfg.batch_size = args.encoder_batch_size
    if args.quick_test:
        encoder_cfg.max_epochs = 5
        encoder_cfg.batch_size = 256
    else:
        encoder_cfg.max_epochs = args.encoder_epochs

    if hasattr(classifier_cfg, 'd_model'):
        classifier_cfg.d_model = args.d_model
        classifier_cfg.d_state = args.d_state
        classifier_cfg.n_layers = args.n_layers
        classifier_cfg.num_blocks = args.num_blocks
    classifier_cfg.lr = args.lr
    classifier_cfg.seed = args.seed
    classifier_cfg.use_amp = args.use_amp
    classifier_cfg.batch_size = args.batch_size
    if args.quick_test:
        classifier_cfg.max_epochs = 5
        classifier_cfg.batch_size = 256
    else:
        classifier_cfg.max_epochs = args.epochs


def load_trained_model_for_attribution(model_id, data, device, args, output_dir):
    """
    Rebuild a previously trained model from its composite checkpoint
    (used with --skip_training to run attribution without retraining).

    Loads encoder + classifier state dicts from <model_dir>/composite_model.pt
    and the cached test latents from the shared encoder dir. Also regenerates
    UMAP embeddings and test-set metrics/probabilities if they are missing.
    """
    spec = ABLATION_MODELS[model_id]
    model_dir = os.path.join(output_dir, model_id)
    ckpt_path = os.path.join(model_dir, 'composite_model.pt')

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"--skip_training: composite checkpoint not found at {ckpt_path}. "
            f"Train {model_id} first (run run_ablation.py without --skip_training)."
        )

    print(f"\n{'#' * 60}")
    print(f"# Loading trained {model_id}: {spec['encoder_type']} + {spec['classifier_type']}")
    print(f"{'#' * 60}")

    # Rebuild configs exactly as during training
    configs = get_config_for_ablation(model_id)
    encoder_cfg = configs['encoder_cfg']
    classifier_cfg = configs['classifier_cfg']
    _apply_cli_overrides(encoder_cfg, classifier_cfg, args)

    encoder_type = spec['encoder_type']
    classifier_type = spec['classifier_type']

    ckpt = torch.load(ckpt_path, map_location=device)

    # Rebuild encoder
    if encoder_type == "dense_dae":
        encoder = DenoisingAutoencoder(
            input_dim=encoder_cfg.input_dim,
            encoder_neurons=encoder_cfg.encoder_neurons,
            decoder_neurons=encoder_cfg.decoder_neurons,
        ).to(device)
    else:
        encoder = BiMambaAutoencoder(encoder_cfg).to(device)
    encoder.load_state_dict(ckpt['encoder_state'])
    encoder.eval()

    # Rebuild classifier
    if classifier_type == "mamba":
        model = BiMambaClassifier(classifier_cfg).to(device)
    else:
        model = MLPClassifier(classifier_cfg).to(device)
    model.load_state_dict(ckpt['classifier_state'])
    model.eval()
    print(f"  Loaded composite checkpoint from {ckpt_path}")

    composite = CompositeModel(encoder, model, encoder_type, classifier_type)

    # Load cached test latents (needed for optional re-evaluation)
    shared_dir_name = "shared_dense_dae" if encoder_type == "dense_dae" else "shared_bimamba_ae"
    shared_dir = os.path.join(output_dir, shared_dir_name)
    latent_path = os.path.join(shared_dir, 'test_latent.npy')
    test_latent = None
    if os.path.exists(latent_path):
        test_latent = np.load(latent_path)
        print(f"  Loaded cached test latents: {test_latent.shape}")

    # Regenerate UMAP embeddings if missing
    lat_emb_path = os.path.join(model_dir, 'test_latent_embeddings.npy')
    clf_emb_path = os.path.join(model_dir, 'test_clf_features.npy')
    if not (os.path.exists(lat_emb_path) and os.path.exists(clf_emb_path)):
        print("  Regenerating embeddings for UMAP...")
        latents_emb, clf_features = compute_latent_and_features(
            composite, data['test_features'], device, batch_size=512)
        np.save(lat_emb_path, latents_emb)
        np.save(clf_emb_path, clf_features)
        print(f"  Saved embeddings: latent={latents_emb.shape}, features={clf_features.shape}")

    # Re-evaluate on test set if metrics/probabilities are missing
    metrics = None
    metrics_path = os.path.join(model_dir, 'test_metrics.json')
    probs_path = os.path.join(model_dir, 'test_probs.npy')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
    if metrics is None or not os.path.exists(probs_path):
        if test_latent is None:
            print("  Cached test latents missing; computing from encoder...")
            if encoder_type == "dense_dae":
                test_latent = compute_dae_latent(encoder, data['test_features'], device)
            else:
                test_latent = compute_bimamba_ae_latent(encoder, data['test_features'], device)
        from train import evaluate_on_test
        print("  Re-evaluating on test set (metrics/probs missing)...")
        metrics, probs, binary = evaluate_on_test(
            model, data, device, model_dir, test_latent)

    return {
        'model_id': model_id,
        'model_dir': model_dir,
        'metrics': metrics,
        'encoder_type': encoder_type,
        'classifier_type': classifier_type,
        'encoder': encoder,
        'classifier': model,
        'composite': composite,
        'test_latent': test_latent,
    }


def train_single_model(model_id, data, device, args, output_dir):
    """
    Train a single ablation model (M1-M4).
    Uses shared encoders where possible.
    """
    spec = ABLATION_MODELS[model_id]
    model_dir = os.path.join(output_dir, model_id)
    os.makedirs(model_dir, exist_ok=True)

    print(f"\n{'#' * 60}")
    print(f"# Training {model_id}: {spec['encoder_type']} + {spec['classifier_type']}")
    print(f"# Direction AE: {spec['direction']}")
    print(f"# Direction Classifier: {spec['direction_classifier']}")
    print(f"{'#' * 60}")

    # Get configs
    configs = get_config_for_ablation(model_id)
    encoder_cfg = configs['encoder_cfg']
    classifier_cfg = configs['classifier_cfg']

    # Apply CLI overrides
    _apply_cli_overrides(encoder_cfg, classifier_cfg, args)

    encoder_type = spec['encoder_type']
    classifier_type = spec['classifier_type']

    # ─── Encoder (shared) ────────────────────────────────────────────────────
    # Dense DAE shared between M1/M2; BiMamba AE shared between M3/M4
    shared_dir_name = "shared_dense_dae" if encoder_type == "dense_dae" else "shared_bimamba_ae"
    shared_dir = os.path.join(output_dir, shared_dir_name)
    encoder_ckpt = os.path.join(shared_dir, 'encoder_best.pt')

    train_latent = val_latent = test_latent = None
    encoder = None

    #print("Encoder config:", encoder_cfg)
    if os.path.exists(encoder_ckpt):
        print(f"  Loading shared encoder from {encoder_ckpt}")
        if encoder_type == "dense_dae":
            encoder = DenoisingAutoencoder(
                input_dim=encoder_cfg.input_dim,
                encoder_neurons=encoder_cfg.encoder_neurons,
                decoder_neurons=encoder_cfg.decoder_neurons,
            ).to(device)
        else:
            encoder = BiMambaAutoencoder(encoder_cfg).to(device)
        print(f"  Encoder {encoder_type} parameters: {sum(p.numel() for p in encoder.parameters()):,}")
        encoder.load_state_dict(torch.load(encoder_ckpt, map_location=device))
    else:
        os.makedirs(shared_dir, exist_ok=True)
        from train import train_dense_dae, train_bimamba_ae
        if encoder_type == "dense_dae":
            encoder = train_dense_dae(encoder_cfg, data, device, shared_dir)
        else:
            encoder = train_bimamba_ae(encoder_cfg, data, device, shared_dir)

    # Compute latents (shared)
    latent_path = os.path.join(shared_dir, 'test_latent.npy')
    if os.path.exists(latent_path):
        train_latent = np.load(os.path.join(shared_dir, 'train_latent.npy'))
        val_latent = np.load(os.path.join(shared_dir, 'val_latent.npy'))
        test_latent = np.load(latent_path)
        print(f"  Loaded cached latents: {train_latent.shape}")
    else:
        from train import compute_and_save_latents
        train_latent, val_latent, test_latent = compute_and_save_latents(
            encoder, encoder_type, data, device, shared_dir)

    # ─── Classifier ──────────────────────────────────────────────────────────
    from train import train_classifier, evaluate_on_test
    #print("Classifier config:", classifier_cfg)
    model = train_classifier(
        classifier_cfg, data, device, model_dir,
        train_latent, val_latent, test_latent,
        classifier_type=classifier_type)

    # Evaluate
    metrics, probs, binary = evaluate_on_test(
        model, data, device, model_dir, test_latent)

    # Save composite model and embeddings
    composite = CompositeModel(encoder, model, encoder_type, classifier_type)
    torch.save({
        'encoder_state': encoder.state_dict(),
        'classifier_state': model.state_dict(),
        'encoder_type': encoder_type,
        'classifier_type': classifier_type,
    }, os.path.join(model_dir, 'composite_model.pt'))
    print(f"  Saved composite model to {model_dir}/composite_model.pt")

    # Save embeddings for UMAP
    latents_emb, clf_features = compute_latent_and_features(
        composite, data['test_features'], device, batch_size=512)
    np.save(os.path.join(model_dir, 'test_latent_embeddings.npy'), latents_emb)
    np.save(os.path.join(model_dir, 'test_clf_features.npy'), clf_features)
    print(f"  Saved embeddings: latent={latents_emb.shape}, features={clf_features.shape}")

    return {
        'model_id': model_id,
        'model_dir': model_dir,
        'metrics': metrics,
        'encoder_type': encoder_type,
        'classifier_type': classifier_type,
        'encoder': encoder,
        'classifier': model,
        'composite': composite,
        'test_latent': test_latent,
    }


def run_attribution_for_model(model_result, data, device, args, output_dir):
    """Run both SHAP and IG attribution for a trained model (gene-level via composite)."""
    # Lazy imports for attribution
    from attribution.shap_attrib import compute_shap_attributions_composite, save_shap_results
    from attribution.ig_attrib import compute_ig_attributions_composite, save_ig_results

    model_id = model_result['model_id']
    model_dir = model_result['model_id'] if 'model_dir' not in model_result else model_result['model_dir']
    composite = model_result['composite']

    # Prepare balanced test subset
    subset_features, subset_labels, subset_labels_str = sample_balanced_test_subset(
        data['test_features'], data['test_labels_str'],
        n_per_celltype=200, seed=1111)

    print(f"\n  Balanced test subset: {subset_features.shape}")

    attrib_dir = os.path.join(model_dir, 'attribution')
    os.makedirs(attrib_dir, exist_ok=True)

    attrib_methods = args.attributions or ABLATION_ATTRIBUTION
    if "all" in attrib_methods:
        attrib_methods = ["shap", "ig"]
    else: 
        attrib_methods = [m.lower() for m in attrib_methods.split('+')]
    for method in attrib_methods:
        print(f"\n  === {model_id} / {method.upper()} ===")

        if method == 'shap':
            shap_values = compute_shap_attributions_composite(
                composite, subset_features, data['train_features'], device,
                background_size=1000, seed=1111,
                mt_indices=data['mt_indices'])
            save_shap_results(shap_values, subset_labels_str,
                              data['gene_names'], attrib_dir, model_name=model_id)

        elif method == 'ig':
            ig_values = compute_ig_attributions_composite(
                composite, subset_features, device,
                n_steps=200, baseline="zero",
                mt_indices=data['mt_indices'])
            save_ig_results(ig_values, subset_labels_str,
                            data['gene_names'], attrib_dir, model_name=model_id)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Data config
    data_cfg = DataConfig(
        train_path=os.path.join(args.data_dir, 'RANDOMIZED_train_set_p_80.csv'),
        val_path=os.path.join(args.data_dir, 'RANDOMIZED_val_set_p_20.csv'),
        test_path=os.path.join(args.data_dir, '240519_test_set_only_n2_samples.csv'),
        train_labels_path=os.path.join(args.data_dir, 'RANDOMIZED_train_set_labels_p_80.csv'),
        val_labels_path=os.path.join(args.data_dir, 'RANDOMIZED_val_set_labels_p_20.csv'),
        test_labels_path=os.path.join(args.data_dir, '240519_test_set_only_n2_samples_labels.csv'),
        cache_dir=os.path.join(args.data_dir, 'cache'),
    )

    # Prepare data
    data = prepare_data(data_cfg, device)

    # Determine which models to run
    model_ids = args.models or list(ABLATION_MODELS.keys())

    # Train all models (or load existing checkpoints with --skip_training)
    all_results = {}
    for model_id in model_ids:
        if args.skip_training:
            result = load_trained_model_for_attribution(
                model_id, data, device, args, args.output_dir)
        else:
            result = train_single_model(model_id, data, device, args, args.output_dir)
        all_results[model_id] = result

    # Run attribution for all models
    if not args.skip_attribution:
        for model_id, result in all_results.items():
            run_attribution_for_model(result, data, device, args, args.output_dir)
    else:
        print("\nSkipping attribution (--skip_attribution).")

    # Save ablation summary
    summary = {}
    for model_id, result in all_results.items():
        spec = ABLATION_MODELS[model_id]
        summary[model_id] = {
            'encoder_type': spec['encoder_type'],
            'classifier_type': spec['classifier_type'],
            'direction': spec['direction'],
            "direction_classifier": spec['direction_classifier'],
            'metrics': result['metrics'],
        }

    with open(os.path.join(args.output_dir, 'ablation_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 60}")
    print("Ablation matrix complete!")
    print(f"{'=' * 60}")
    for model_id, s in summary.items():
        m = s['metrics']
        if not m:
            print(f"  {model_id} ({s['encoder_type']}/{s['classifier_type']}/{s['direction']}): "
                  f"metrics unavailable")
            continue
        print(f"  {model_id} ({s['encoder_type']}/{s['classifier_type']}/{s['direction_classifier']}): "
              f"F1={m['f1_mean']:.4f} "
              f"(sp={m['f1_species']:.4f} ct={m['f1_celltype']:.4f} "
              f"dis={m['f1_disease']:.4f})")

    print(f"\nAll results saved to {args.output_dir}/")
    print(f"\nNext step: python run_analysis.py --ablation_dir {args.output_dir}")


if __name__ == '__main__':
    main()
