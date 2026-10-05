#!/usr/bin/env python3
"""
Evaluation script for HF Mamba subtyping pipeline.

Loads a trained model (any of M1-M4), evaluates on the test set, computes
metrics, confusion matrices, and saves all results. Also supports building
a composite model for gene-level attribution.

Usage:
  python test.py \
    --model_dir results/M1 \
    --test_path data/240519_test_set_only_n2_samples.csv \
    --test_labels data/240519_test_set_only_n2_samples_labels.csv \
    --cache_dir data/cache

  # With pre-computed test latent:
  python test.py \
    --model_dir results/M3 \
    --test_path data/240519_test_set_only_n2_samples.csv \
    --test_labels data/240519_test_set_only_n2_samples_labels.csv \
    --test_latent_path results/M3/test_latent.npy
"""
import argparse
import os
import json
import numpy as np
import torch

from config import (
    DataConfig, DAEConfig, BiMambaAEConfig, MambaConfig, MLPConfig,
    set_seed, N_GENES, N_CLASSES, ALL_CLASSES,
    SPECIES_CLASSES, CELLTYPE_CLASSES, DISEASE_CLASSES,
    SPECIES_IDX, CELLTYPE_IDX, DISEASE_IDX
)
from data import csv_to_memmap, load_labels, load_label_strings, create_dataloader
from losses import (
    compute_metrics, threshold_predictions, onehot_predict,
    correct_classification_rate, confusion_matrix_category
)
from models.dae import DenoisingAutoencoder, compute_dae_latent
from models.bimamba_ae import BiMambaAutoencoder, compute_bimamba_ae_latent
from models.classifier import BiMambaClassifier
from models.mlp_classifier import MLPClassifier
from models.composite_model import CompositeModel, build_composite_model, compute_latent_and_features


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate trained HF Mamba classifier')

    # Model
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory containing classifier_best.pt and config.json')
    parser.add_argument('--classifier_weights', type=str, default='',
                        help='Override path to classifier weights')

    # Data
    parser.add_argument('--test_path', type=str, required=True)
    parser.add_argument('--test_labels', type=str, required=True)
    parser.add_argument('--cache_dir', type=str, default='data/cache')
    parser.add_argument('--output_dir', type=str, default='',
                        help='Output directory (default: model_dir/eval)')

    # Encoder
    parser.add_argument('--encoder_type', type=str, default='',
                        choices=['', 'dense_dae', 'bimamba_ae'],
                        help='Override encoder type (auto-detected from config)')
    parser.add_argument('--encoder_checkpoint', type=str, default='',
                        help='Path to encoder weights (default: model_dir/encoder_best.pt)')
    parser.add_argument('--test_latent_path', type=str, default='',
                        help='Pre-computed test latent .npy (skip encoder inference)')

    # Evaluation
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--num_workers', type=int, default=4)

    # Composite model for attribution / UMAP
    parser.add_argument('--save_composite', action='store_true',
                        help='Save composite model and latent/features for UMAP')
    parser.add_argument('--save_embeddings', action='store_true',
                        help='Save latent and classifier features for UMAP')

    return parser.parse_args()


def load_model_config(model_dir: str) -> dict:
    """Load the saved config.json from training."""
    config_path = os.path.join(model_dir, 'config.json')
    with open(config_path, 'r') as f:
        return json.load(f)


def build_classifier_from_config(saved_cfg: dict):
    """Reconstruct classifier from saved config."""
    classifier_type = saved_cfg.get('classifier_type', 'mamba')

    if classifier_type == 'mamba':
        cfg = MambaConfig(
            seq_len=saved_cfg.get('seq_len', 350),
            d_model=saved_cfg.get('d_model', 64),
            d_state=saved_cfg.get('d_state', 16),
            n_layers=saved_cfg.get('n_layers', 3),
            num_blocks=saved_cfg.get('num_blocks', 2),
            bidirectional=saved_cfg.get('bidirectional', True),
            use_start_conv=saved_cfg.get('use_start_conv', True),
            use_end_conv=saved_cfg.get('use_end_conv', True),
            use_intra_residual=saved_cfg.get('use_intra_residual', True),
            use_global_residual=saved_cfg.get('use_global_residual', True),
            threshold=saved_cfg.get('threshold', 0.5),
        )
        return BiMambaClassifier(cfg), cfg, 'mamba'
    else:  # mlp
        cfg = MLPConfig(
            input_dim=saved_cfg.get('input_dim', 350),
            hidden_neurons=saved_cfg.get('hidden_neurons', [795, 230, 105]),
            threshold=saved_cfg.get('threshold', 0.5),
        )
        return MLPClassifier(cfg), cfg, 'mlp'


def build_encoder_from_config(saved_cfg: dict, args):
    """Reconstruct encoder from saved config."""
    encoder_type = args.encoder_type or saved_cfg.get('encoder_type', 'dense_dae')

    if encoder_type == 'dense_dae':
        cfg = DAEConfig()
        encoder = DenoisingAutoencoder(
            input_dim=cfg.input_dim,
            encoder_neurons=cfg.encoder_neurons,
            decoder_neurons=cfg.decoder_neurons,
        )
        return encoder, cfg, 'dense_dae'
    else:  # bimamba_ae
        cfg = BiMambaAEConfig(
            d_model=saved_cfg.get('d_model', 64),
            d_state=saved_cfg.get('d_state', 16),
            n_layers=saved_cfg.get('n_layers', 3),
            num_blocks=saved_cfg.get('num_blocks', 2),
            bidirectional=saved_cfg.get('bidirectional', True),
            use_start_conv=saved_cfg.get('use_start_conv', True),
            use_end_conv=saved_cfg.get('use_end_conv', True),
            use_intra_residual=saved_cfg.get('use_intra_residual', True),
            use_global_residual=saved_cfg.get('use_global_residual', True),
        )
        encoder = BiMambaAutoencoder(cfg)
        return encoder, cfg, 'bimamba_ae'


def evaluate_model(model, test_features, test_labels, device, batch_size=512):
    """Run model inference on test set and return probabilities."""
    model.eval()
    test_loader = create_dataloader(
        test_features, test_labels,
        batch_size=batch_size, shuffle=False, device=device,
        num_workers=4, pin_memory=True)

    all_probs = []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device, non_blocking=True)
            pred = model(x)
            all_probs.append(pred.cpu().numpy())

    return np.concatenate(all_probs, axis=0)


def compute_confusion_matrices(y_true, y_pred_binary):
    """Compute confusion matrices for species, celltype, and disease."""
    cm_species = confusion_matrix_category(
        y_true, y_pred_binary, SPECIES_IDX[0], SPECIES_IDX[1])
    cm_celltype = confusion_matrix_category(
        y_true, y_pred_binary, CELLTYPE_IDX[0], CELLTYPE_IDX[1])
    cm_disease = confusion_matrix_category(
        y_true, y_pred_binary, DISEASE_IDX[0], DISEASE_IDX[1])
    return {
        'species': cm_species,
        'celltype': cm_celltype,
        'disease': cm_disease,
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    output_dir = args.output_dir or os.path.join(args.model_dir, 'eval')
    os.makedirs(output_dir, exist_ok=True)

    # Load saved config
    saved_cfg = load_model_config(args.model_dir)
    print(f"Loaded config: {saved_cfg}")

    # Build and load classifier
    classifier, clf_cfg, clf_type = build_classifier_from_config(saved_cfg)
    classifier = classifier.to(device)
    weights_path = args.classifier_weights or os.path.join(
        args.model_dir, 'classifier_best.pt')
    classifier.load_state_dict(torch.load(weights_path, map_location=device))
    print(f"Loaded classifier weights from {weights_path}")

    # Load test data
    print("Loading test data...")
    test_features, gene_names = csv_to_memmap(
        args.test_path, args.cache_dir, "test")
    test_labels = load_labels(args.test_labels)
    test_labels_str = load_label_strings(args.test_labels)
    print(f"Test set: {test_features.shape[0]} cells x {test_features.shape[1]} genes")

    # Get test latent (through encoder)
    if args.test_latent_path and os.path.exists(args.test_latent_path):
        print(f"Loading pre-computed test latent from {args.test_latent_path}")
        test_input = np.load(args.test_latent_path)
    else:
        # Build and load encoder
        encoder, enc_cfg, enc_type = build_encoder_from_config(saved_cfg, args)
        encoder = encoder.to(device)
        enc_path = args.encoder_checkpoint or os.path.join(
            args.model_dir, 'encoder_best.pt')
        print(f"Loading encoder from {enc_path}")
        encoder.load_state_dict(torch.load(enc_path, map_location=device))

        # Compute latent
        if enc_type == 'dense_dae':
            test_input = compute_dae_latent(encoder, test_features, device)
        else:
            test_input = compute_bimamba_ae_latent(encoder, test_features, device)

    # Evaluate classifier on latent
    print("Running inference...")
    probs = evaluate_model(classifier, test_input, test_labels, device,
                           batch_size=args.batch_size)

    # Binary predictions
    binary = onehot_predict(probs)
    binary_thresh = threshold_predictions(probs, args.threshold)

    # Metrics
    metrics = compute_metrics(test_labels, binary)
    metrics_thresh = compute_metrics(test_labels, binary_thresh)
    ccr = correct_classification_rate(test_labels, binary)
    cms = compute_confusion_matrices(test_labels, binary)

    # Save all results
    np.save(os.path.join(output_dir, 'test_probs.npy'), probs)
    np.save(os.path.join(output_dir, 'test_binary.npy'), binary)
    np.save(os.path.join(output_dir, 'test_binary_thresh.npy'), binary_thresh)
    np.save(os.path.join(output_dir, 'cm_species.npy'), cms['species'])
    np.save(os.path.join(output_dir, 'cm_celltype.npy'), cms['celltype'])
    np.save(os.path.join(output_dir, 'cm_disease.npy'), cms['disease'])

    results = {
        'metrics_onehot': metrics,
        'metrics_threshold': metrics_thresh,
        'correct_classification_rate': ccr.tolist(),
        'class_names': ALL_CLASSES,
        'n_test_cells': int(len(test_labels)),
        'classifier_type': clf_type,
        'encoder_type': saved_cfg.get('encoder_type', 'dense_dae'),
    }
    with open(os.path.join(output_dir, 'eval_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("Evaluation Results (onehot prediction)")
    print("=" * 60)
    print(f"  F1 (mean):     {metrics['f1_mean']:.4f}")
    print(f"  F1 (species):  {metrics['f1_species']:.4f}")
    print(f"  F1 (celltype): {metrics['f1_celltype']:.4f}")
    print(f"  F1 (disease):  {metrics['f1_disease']:.4f}")
    print(f"  Precision:     {metrics['precision_mean']:.4f}")
    print(f"  Recall:        {metrics['recall_mean']:.4f}")

    print("\n  Correct classification rates per class:")
    for i, name in enumerate(ALL_CLASSES):
        print(f"    {name:20s}: {ccr[i]:.4f}")

    # Save composite model and embeddings if requested
    if args.save_composite or args.save_embeddings:
        print("\nSaving composite model and embeddings for UMAP/attribution...")

        # Rebuild encoder if not already loaded
        if 'encoder' not in dir():
            encoder, enc_cfg, enc_type = build_encoder_from_config(saved_cfg, args)
            encoder = encoder.to(device)
            enc_path = args.encoder_checkpoint or os.path.join(
                args.model_dir, 'encoder_best.pt')
            encoder.load_state_dict(torch.load(enc_path, map_location=device))

        composite = CompositeModel(encoder, classifier, enc_type, clf_type)

        # Save composite model state dict
        if args.save_composite:
            torch.save({
                'encoder_state': encoder.state_dict(),
                'classifier_state': classifier.state_dict(),
                'encoder_type': enc_type,
                'classifier_type': clf_type,
            }, os.path.join(args.model_dir, 'composite_model.pt'))
            print(f"  Saved composite model to {args.model_dir}/composite_model.pt")

        # Compute and save embeddings
        if args.save_embeddings:
            latents, clf_features = compute_latent_and_features(
                composite, test_features, device, batch_size=512)
            np.save(os.path.join(args.model_dir, 'test_latent_embeddings.npy'), latents)
            np.save(os.path.join(args.model_dir, 'test_clf_features.npy'), clf_features)
            print(f"  Saved latent embeddings: {latents.shape}")
            print(f"  Saved classifier features: {clf_features.shape}")

    print(f"\nAll results saved to {output_dir}/")


if __name__ == '__main__':
    main()
