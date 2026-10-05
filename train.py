#!/usr/bin/env python3
"""
Training script for HF Mamba subtyping pipeline.

Supports all 4 ablation configurations:
  M1: Dense DAE + BiMamba (BiDir)
  M2: Dense DAE + BiMamba (UniDir)
  M3: BiMamba AE + MLP
  M4: BiMamba AE + BiMamba (BiDir)

Requires GPU with mamba_ssm and causal_conv1d installed.

Usage:
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
    --encoder_type bimamba_ae --classifier_type mlp \
    --direction bidirectional --use_amp \
    --output_dir results/M3
"""
import argparse
import os
import json
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import (
    DataConfig, DAEConfig, BiMambaAEConfig, MambaConfig, MLPConfig,
    set_seed, N_GENES, N_CLASSES, ALL_CLASSES
)
from data import prepare_data, create_dataloader, HFDataset
from losses import macro_f1_loss, compute_metrics, threshold_predictions, onehot_predict
from models.dae import DenoisingAutoencoder, compute_dae_latent
from models.bimamba_ae import BiMambaAutoencoder, compute_bimamba_ae_latent
from models.classifier import BiMambaClassifier
from models.mlp_classifier import MLPClassifier


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train HF Mamba classifier (any ablation config)')

    # Data paths
    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument('--val_path', type=str, required=True)
    parser.add_argument('--test_path', type=str, required=True)
    parser.add_argument('--train_labels', type=str, required=True)
    parser.add_argument('--val_labels', type=str, required=True)
    parser.add_argument('--test_labels', type=str, required=True)
    parser.add_argument('--cache_dir', type=str, default='data/cache')
    parser.add_argument('--output_dir', type=str, required=True)

    # Model type selection
    parser.add_argument('--encoder_type', type=str, default='dense_dae',
                        choices=['dense_dae', 'bimamba_ae'],
                        help='Encoder type: dense_dae (M1/M2) or bimamba_ae (M3/M4)')
    parser.add_argument('--classifier_type', type=str, default='mamba',
                        choices=['mamba', 'mlp'],
                        help='Classifier type: mamba (M1/M2/M4) or mlp (M3)')
    parser.add_argument('--direction', type=str, default='bidirectional',
                        choices=['bidirectional', 'unidirectional', 'None'],
                        help='Direction for BiMamba AE (M3/M4). Default: None (use DAE).')
    parser.add_argument('--direction_classifier', type=str, default='bidirectional',
                        choices=['bidirectional', 'unidirectional', 'None'],
                        help='Direction for BiMamba classifier (M1/M2/M4). Default: bidirectional.')
    # Dense DAE
    parser.add_argument('--dae_encoder_neurons', type=int, nargs='+',
                        default=[5000, 2400, 350])
    parser.add_argument('--dae_decoder_neurons', type=int, nargs='+',
                        default=[2200, 5150])
    parser.add_argument('--dae_checkpoint', type=str, default='',
                        help='Path to pre-trained DAE weights (skip DAE training)')

    # BiMamba AE
    parser.add_argument('--ae_d_model', type=int, default=64)
    parser.add_argument('--ae_d_state', type=int, default=16)
    parser.add_argument('--ae_n_layers', type=int, default=3)
    parser.add_argument('--ae_num_blocks', type=int, default=2)
    parser.add_argument('--ae_checkpoint', type=str, default='',
                        help='Path to pre-trained BiMamba AE weights (skip AE training)')

    # Mamba classifier
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--d_state', type=int, default=16)
    parser.add_argument('--d_conv', type=int, default=4)
    parser.add_argument('--expand_factor', type=int, default=2)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--num_blocks', type=int, default=2)
    parser.add_argument('--conv_kernel', type=int, default=3)
    parser.add_argument('--classifier_hidden', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--pool', type=str, default='mean_max',
                        choices=['mean', 'max', 'mean_max'])

    # MLP classifier
    parser.add_argument('--mlp_hidden', type=int, nargs='+',
                        default=[795, 230, 105])

    # Architecture micro-ablation flags
    parser.add_argument('--no_start_conv', action='store_true')
    parser.add_argument('--no_end_conv', action='store_true')
    parser.add_argument('--no_intra_residual', action='store_true')
    parser.add_argument('--no_global_residual', action='store_true')

    # Encoder training
    parser.add_argument('--encoder_batch_size', type=int, default=1024)
    parser.add_argument('--encoder_epochs', type=int, default=500)
    parser.add_argument('--encoder_lr', type=float, default=0.0001)

    # Classifier training
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_factor', type=float, default=0.1)
    parser.add_argument('--lr_patience', type=int, default=25)
    parser.add_argument('--lr_min', type=float, default=1e-7)
    parser.add_argument('--early_stop_patience', type=int, default=50)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--use_amp', action='store_true',
                        help='Use mixed precision training')
    parser.add_argument('--checkpoint_every', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                        help='Weight decay for optimizer')
    return parser.parse_args()


# ─── Config builders ─────────────────────────────────────────────────────────

def build_encoder_config(args):
    """Build encoder config from CLI args."""
    if args.encoder_type == 'dense_dae':
        return DAEConfig(
            input_dim=N_GENES,
            encoder_neurons=args.dae_encoder_neurons,
            decoder_neurons=args.dae_decoder_neurons,
            batch_size=args.encoder_batch_size,
            max_epochs=args.encoder_epochs,
            lr=args.encoder_lr,
            seed=args.seed,
        )
    else:  # bimamba_ae
        return BiMambaAEConfig(
            input_dim=N_GENES,
            d_model=args.ae_d_model,
            d_state=args.ae_d_state,
            n_layers=args.ae_n_layers,
            num_blocks=args.ae_num_blocks,
            use_start_conv=not args.no_start_conv,
            use_end_conv=not args.no_end_conv,
            use_intra_residual=not args.no_intra_residual,
            use_global_residual=not args.no_global_residual,
            batch_size=args.encoder_batch_size,
            max_epochs=args.encoder_epochs,
            lr=args.encoder_lr,
            weight_decay=args.weight_decay,
            seed=args.seed,
            use_amp=args.use_amp,
        )


def build_classifier_config(args):
    """Build classifier config from CLI args."""
    if args.classifier_type == 'mamba':
        return MambaConfig(
            seq_len=350,
            d_model=args.d_model,
            d_state=args.d_state,
            d_conv=args.d_conv,
            expand_factor=args.expand_factor,
            n_layers=args.n_layers,
            num_blocks=args.num_blocks,
            bidirectional=(args.direction_classifier == 'bidirectional'),
            use_start_conv=not args.no_start_conv,
            use_end_conv=not args.no_end_conv,
            use_intra_residual=not args.no_intra_residual,
            use_global_residual=not args.no_global_residual,
            conv_kernel=args.conv_kernel,
            pool=args.pool,
            classifier_hidden=args.classifier_hidden,
            dropout=args.dropout,
            lr=args.lr,
            lr_factor=args.lr_factor,
            lr_patience=args.lr_patience,
            lr_min=args.lr_min,
            early_stop_patience=args.early_stop_patience,
            max_epochs=args.epochs,
            batch_size=args.batch_size,
            threshold=args.threshold,
            seed=args.seed,
            use_amp=args.use_amp,
            checkpoint_every=args.checkpoint_every,
        )
    else:  # mlp
        return MLPConfig(
            input_dim=350,
            hidden_neurons=args.mlp_hidden,
            dropout=args.dropout,
            lr=args.lr,
            lr_factor=args.lr_factor,
            lr_patience=args.lr_patience,
            lr_min=args.lr_min,
            early_stop_patience=args.early_stop_patience,
            max_epochs=args.epochs,
            batch_size=args.batch_size,
            threshold=args.threshold,
            seed=args.seed,
            use_amp=args.use_amp,
            checkpoint_every=args.checkpoint_every,
        )


# ─── Encoder training ────────────────────────────────────────────────────────

def train_dense_dae(dae_cfg, data, device, output_dir):
    """Train the Dense Denoising Autoencoder."""
    print("\n" + "=" * 60)
    print("Training Dense DAE")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    dae = DenoisingAutoencoder(
        input_dim=dae_cfg.input_dim,
        encoder_neurons=dae_cfg.encoder_neurons,
        decoder_neurons=dae_cfg.decoder_neurons,
    ).to(device)

    optimizer = Adam(dae.parameters(), lr=dae_cfg.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=dae_cfg.lr_factor,
                                  patience=dae_cfg.lr_patience, min_lr=dae_cfg.lr_min)
    criterion = nn.MSELoss()

    train_loader = create_dataloader(
        data['train_features'], data['train_labels'],
        dae_cfg.batch_size, shuffle=True, device=device,
        num_workers=4, pin_memory=True)
    val_loader = create_dataloader(
        data['val_features'], data['val_labels'],
        dae_cfg.batch_size, shuffle=False, device=device,
        num_workers=4, pin_memory=True)

    best_val_loss = float('inf')
    patience_counter = 0
    history = {'loss': [], 'val_loss': [], 'epoch': []}

    for epoch in range(1, dae_cfg.max_epochs + 1):
        dae.train()
        train_loss = 0.0
        n_batches = 0
        for x, _ in train_loader:
            x = x.to(device, non_blocking=True)
            optimizer.zero_grad()
            recon = dae(x)
            loss = criterion(recon, x)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        train_loss /= n_batches

        dae.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device, non_blocking=True)
                recon = dae(x)
                loss = criterion(recon, x)
                val_loss += loss.item()
                n_val += 1
        val_loss /= n_val

        scheduler.step(val_loss)
        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['epoch'].append(epoch)

        print(f"  DAE Epoch {epoch}/{dae_cfg.max_epochs} | "
              f"Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(dae.state_dict(), os.path.join(output_dir, 'encoder_best.pt'))
        else:
            patience_counter += 1
            if patience_counter >= dae_cfg.early_stop_patience:
                print(f"  DAE early stopping at epoch {epoch}")
                break

    torch.save(dae.state_dict(), os.path.join(output_dir, 'encoder_final.pt'))
    with open(os.path.join(output_dir, 'encoder_history.pkl'), 'wb') as f:
        pickle.dump(history, f)

    print(f"DAE training complete. Best val loss: {best_val_loss:.6f}")
    dae.load_state_dict(torch.load(os.path.join(output_dir, 'encoder_best.pt')))
    return dae


def train_bimamba_ae(ae_cfg, data, device, output_dir):
    """Train the BiMamba Autoencoder."""
    print("\n" + "=" * 60)
    print("Training BiMamba Autoencoder")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    ae = BiMambaAutoencoder(ae_cfg).to(device)
    n_params = sum(p.numel() for p in ae.parameters())
    print(f"  Parameters: {n_params:,}")

    optimizer = AdamW(ae.parameters(), lr=ae_cfg.lr, weight_decay=ae_cfg.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=ae_cfg.lr_factor,
        patience=ae_cfg.lr_patience,
        min_lr=ae_cfg.lr_min,
    )
    criterion = nn.MSELoss()

    train_loader = create_dataloader(
        data["train_features"],
        data["train_labels"],
        ae_cfg.batch_size,
        shuffle=True,
        device=device,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = create_dataloader(
        data["val_features"],
        data["val_labels"],
        ae_cfg.batch_size,
        shuffle=False,
        device=device,
        num_workers=4,
        pin_memory=True,
    )

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"loss": [], "val_loss": [], "epoch": []}

    for epoch in range(1, ae_cfg.max_epochs + 1):
        ae.train()
        train_loss = 0.0
        n_batches = 0

        for x, _ in train_loader:
            x = x.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            amp_enabled = (
                ae_cfg.use_amp
                and device.type == "cuda"
            )

            amp_dtype = torch.bfloat16

            with torch.autocast(
                device_type="cuda",
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                recon = ae(x)
                loss = criterion(recon, x)

            loss.backward()
            optimizer.step()

            train_loss += loss.detach().item()
            n_batches += 1

        train_loss /= max(n_batches, 1)

        ae.eval()
        val_loss = 0.0
        n_val = 0

        with torch.inference_mode():
            for x, _ in val_loader:
                x = x.to(device, non_blocking=True)

                amp_enabled = (
                    ae_cfg.use_amp
                    and device.type == "cuda"
                )

                amp_dtype = torch.bfloat16

                with torch.autocast(
                    device_type="cuda",
                    dtype=amp_dtype,
                    enabled=amp_enabled,
                ):
                    recon = ae(x)
                    loss = criterion(recon, x)

                val_loss += loss.item()
                n_val += 1

        val_loss /= max(n_val, 1)

        scheduler.step(val_loss)

        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["epoch"].append(epoch)

        print(
            f"  BiMamba AE Epoch {epoch}/{ae_cfg.max_epochs} | "
            f"Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                ae.state_dict(),
                os.path.join(output_dir, "encoder_best.pt"),
            )
        else:
            patience_counter += 1
            if patience_counter >= ae_cfg.early_stop_patience:
                print(f"  BiMamba AE early stopping at epoch {epoch}")
                break

    torch.save(ae.state_dict(), os.path.join(output_dir, "encoder_final.pt"))

    with open(os.path.join(output_dir, "encoder_history.pkl"), "wb") as f:
        pickle.dump(history, f)

    print(f"BiMamba AE training complete. Best val loss: {best_val_loss:.6f}")

    ae.load_state_dict(
        torch.load(
            os.path.join(output_dir, "encoder_best.pt"),
            map_location=device,
            weights_only=True,
        )
    )
    return ae


def compute_and_save_latents(encoder, encoder_type, data, device, output_dir):
    """Compute latent features for train/val/test and save."""
    print("\nComputing latent features...")

    if encoder_type == "dense_dae":
        train_latent = compute_dae_latent(encoder, data['train_features'], device)
        val_latent = compute_dae_latent(encoder, data['val_features'], device)
        test_latent = compute_dae_latent(encoder, data['test_features'], device)
    else:  # bimamba_ae
        train_latent = compute_bimamba_ae_latent(encoder, data['train_features'], device)
        val_latent = compute_bimamba_ae_latent(encoder, data['val_features'], device)
        test_latent = compute_bimamba_ae_latent(encoder, data['test_features'], device)

    np.save(os.path.join(output_dir, 'train_latent.npy'), train_latent)
    np.save(os.path.join(output_dir, 'val_latent.npy'), val_latent)
    np.save(os.path.join(output_dir, 'test_latent.npy'), test_latent)

    print(f"  Train latent: {train_latent.shape}")
    print(f"  Val latent: {val_latent.shape}")
    print(f"  Test latent: {test_latent.shape}")

    return train_latent, val_latent, test_latent


# ─── Classifier training ─────────────────────────────────────────────────────

def train_classifier(clf_cfg, data, device, output_dir,
                     train_latent, val_latent, test_latent,
                     classifier_type="mamba", criterion=None,
                     train_labels=None, val_labels=None):
    """
    Train the classifier (BiMamba or MLP) on latent features.

    Args:
        criterion: optional loss module (e.g. weighted BCE from
            losses.make_criterion). Defaults to BCEWithLogitsLoss.
        train_labels/val_labels: optional label arrays aligned with
            train_latent/val_latent (used by k-fold CV, where the latent
            arrays are fold subsets of the pooled train+val data).
            Default: data['train_labels'] / data['val_labels'].
    """
    if train_labels is None:
        train_labels = data['train_labels']
    if val_labels is None:
        val_labels = data['val_labels']
    clf_name = "BiMamba" if classifier_type == "mamba" else "MLP"
    print("\n" + "=" * 60)
    print(f"Training {clf_name} Classifier")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # Build model
    if classifier_type == "mamba":
        model = BiMambaClassifier(clf_cfg).to(device)
    else:  # mlp
        model = MLPClassifier(clf_cfg).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")

    optimizer = Adam(model.parameters(), lr=clf_cfg.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=clf_cfg.lr_factor,
                                  patience=clf_cfg.lr_patience, min_lr=clf_cfg.lr_min)

    use_amp = getattr(clf_cfg, 'use_amp', False)
    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()
    scaler = None
    if use_amp:
        from torch.amp import GradScaler, autocast
        device_type = 'cuda' if device.type == 'cuda' else 'cpu'
        scaler = GradScaler(device_type, enabled=True)

    train_loader = create_dataloader(
        train_latent, train_labels,
        clf_cfg.batch_size, shuffle=True, device=device,
        num_workers=getattr(clf_cfg, 'num_workers', 4) if hasattr(clf_cfg, 'num_workers') else 4,
        pin_memory=True)
    val_loader = create_dataloader(
        val_latent, val_labels,
        clf_cfg.batch_size, shuffle=False, device=device,
        num_workers=4, pin_memory=True)

    best_val_loss = float('inf')
    patience_counter = 0
    history = {
        'loss': [], 'val_loss': [],
        'f1_mean': [], 'val_f1_mean': [],
        'f1_species': [], 'val_f1_species': [],
        'f1_celltype': [], 'val_f1_celltype': [],
        'f1_disease': [], 'val_f1_disease': [],
        'precision_mean': [], 'val_precision_mean': [],
        'recall_mean': [], 'val_recall_mean': [],
        'epoch': [],
    }

    max_epochs = getattr(clf_cfg, 'max_epochs', 500)
    early_stop = getattr(clf_cfg, 'early_stop_patience', 50)
    checkpoint_every = getattr(clf_cfg, 'checkpoint_every', 10)

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()

        # Train
        model.train()
        train_loss = 0.0
        n_batches = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad()
            if use_amp:
                with autocast(device_type=device_type, enabled=use_amp):
                    logits = model(x)
                    loss = criterion(logits, y.float())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x)
                loss = criterion(logits, y.float())
                loss.backward()
                optimizer.step()

            train_loss += loss.item()
            n_batches += 1
        train_loss /= n_batches

        # Validate
        model.eval()
        val_loss = 0.0
        n_val = 0
        all_val_preds = []
        all_val_labels = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                if use_amp:
                    with autocast(device_type=device_type, enabled=use_amp):
                        logits = model(x)
                        loss = criterion(logits, y.float())
                else:
                    logits = model(x)
                    loss = criterion(logits, y.float())
                val_loss += loss.item()
                n_val += 1
                all_val_preds.append(logits.cpu().numpy())
                all_val_labels.append(y.cpu().numpy())

        val_loss /= n_val
        scheduler.step(val_loss)

        val_probs = np.concatenate(all_val_preds, axis=0)
        val_labels = np.concatenate(all_val_labels, axis=0)
        val_binary = onehot_predict(val_probs)
        val_metrics = compute_metrics(val_labels, val_binary)

        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_f1_mean'].append(val_metrics['f1_mean'])
        history['val_f1_species'].append(val_metrics['f1_species'])
        history['val_f1_celltype'].append(val_metrics['f1_celltype'])
        history['val_f1_disease'].append(val_metrics['f1_disease'])
        history['val_precision_mean'].append(val_metrics['precision_mean'])
        history['val_recall_mean'].append(val_metrics['recall_mean'])
        history['epoch'].append(epoch)

        epoch_time = time.time() - epoch_start
        print(f"  Epoch {epoch}/{max_epochs} [{epoch_time:.1f}s] | "
              f"Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
              f"Val F1: {val_metrics['f1_mean']:.4f} "
              f"(sp={val_metrics['f1_species']:.4f} "
              f"ct={val_metrics['f1_celltype']:.4f} "
              f"dis={val_metrics['f1_disease']:.4f}) | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(),
                       os.path.join(output_dir, 'classifier_best.pt'))
        else:
            patience_counter += 1
            if patience_counter >= early_stop:
                print(f"  Early stopping at epoch {epoch}")
                break

        if epoch % checkpoint_every == 0:
            torch.save(model.state_dict(),
                       os.path.join(output_dir, f'classifier_epoch_{epoch}.pt'))

    torch.save(model.state_dict(),
               os.path.join(output_dir, 'classifier_final.pt'))
    with open(os.path.join(output_dir, 'classifier_history.pkl'), 'wb') as f:
        pickle.dump(history, f)

    # Save config
    config_dict = {
        'encoder_type': 'dense_dae' if classifier_type == 'mamba' else 'bimamba_ae',
        'classifier_type': classifier_type,
        'seq_len': 350,
        'n_params': n_params,
    }
    if classifier_type == "mamba":
        config_dict.update({
            'd_model': clf_cfg.d_model,
            'd_state': clf_cfg.d_state,
            'n_layers': clf_cfg.n_layers,
            'num_blocks': clf_cfg.num_blocks,
            'bidirectional': clf_cfg.bidirectional,
            'use_start_conv': clf_cfg.use_start_conv,
            'use_end_conv': clf_cfg.use_end_conv,
            'use_intra_residual': clf_cfg.use_intra_residual,
            'use_global_residual': clf_cfg.use_global_residual,
        })
    else:
        config_dict.update({
            'hidden_neurons': clf_cfg.hidden_neurons,
            'input_dim': clf_cfg.input_dim,
        })
    config_dict.update({
        'batch_size': clf_cfg.batch_size,
        'threshold': clf_cfg.threshold,
        'seed': clf_cfg.seed,
    })

    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config_dict, f, indent=2)

    print(f"\n{clf_name} training complete. Best val loss: {best_val_loss:.6f}")
    model.load_state_dict(torch.load(os.path.join(output_dir, 'classifier_best.pt')))
    return model


def evaluate_on_test(model, data, device, output_dir, test_latent):
    """Evaluate the trained model on the test set."""
    print("\n" + "=" * 60)
    print("Test Set Evaluation")
    print("=" * 60)

    model.eval()

    test_loader = create_dataloader(
        test_latent, data['test_labels'],
        batch_size=512, shuffle=False, device=device,
        num_workers=4, pin_memory=True)

    all_probs = []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device, non_blocking=True)
            pred = model(x)
            all_probs.append(pred.cpu().numpy())

    probs = np.concatenate(all_probs, axis=0)
    binary = onehot_predict(probs)
    metrics = compute_metrics(data['test_labels'], binary)

    np.save(os.path.join(output_dir, 'test_probs.npy'), probs)
    np.save(os.path.join(output_dir, 'test_binary.npy'), binary)

    with open(os.path.join(output_dir, 'test_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"  Test F1 (mean):     {metrics['f1_mean']:.4f}")
    print(f"  Test F1 (species):  {metrics['f1_species']:.4f}")
    print(f"  Test F1 (celltype): {metrics['f1_celltype']:.4f}")
    print(f"  Test F1 (disease):  {metrics['f1_disease']:.4f}")
    print(f"  Test Precision:     {metrics['precision_mean']:.4f}")
    print(f"  Test Recall:        {metrics['recall_mean']:.4f}")

    return metrics, probs, binary


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA: {torch.version.cuda}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Build configs
    data_cfg = DataConfig(
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
        train_labels_path=args.train_labels,
        val_labels_path=args.val_labels,
        test_labels_path=args.test_labels,
        cache_dir=args.cache_dir,
    )
    encoder_cfg = build_encoder_config(args)
    clf_cfg = build_classifier_config(args)

    # Prepare data
    data = prepare_data(data_cfg, device)

    # ─── Train encoder ───────────────────────────────────────────────────────
    encoder = None
    train_latent = val_latent = test_latent = None

    # Check for pre-trained encoder
    encoder_ckpt = args.dae_checkpoint if args.encoder_type == 'dense_dae' else args.ae_checkpoint
    if not encoder_ckpt:
        encoder_ckpt = os.path.join(args.output_dir, 'encoder_best.pt')

    if os.path.exists(encoder_ckpt):
        print(f"\nLoading pre-trained encoder from {encoder_ckpt}")
        if args.encoder_type == 'dense_dae':
            encoder = DenoisingAutoencoder(
                input_dim=encoder_cfg.input_dim,
                encoder_neurons=encoder_cfg.encoder_neurons,
                decoder_neurons=encoder_cfg.decoder_neurons,
            ).to(device)
        else:
            encoder = BiMambaAutoencoder(encoder_cfg).to(device)
        encoder.load_state_dict(torch.load(encoder_ckpt, map_location=device))
    else:
        if args.encoder_type == 'dense_dae':
            encoder = train_dense_dae(encoder_cfg, data, device, args.output_dir)
        else:
            print(
                f"BiMamba AE config: batch_size={encoder_cfg.batch_size}, "
                f"epochs={encoder_cfg.max_epochs}"
            )
            encoder = train_bimamba_ae(encoder_cfg, data, device, args.output_dir)

    # Compute latents
    latent_path = os.path.join(args.output_dir, 'test_latent.npy')
    if os.path.exists(latent_path):
        train_latent = np.load(os.path.join(args.output_dir, 'train_latent.npy'))
        val_latent = np.load(os.path.join(args.output_dir, 'val_latent.npy'))
        test_latent = np.load(latent_path)
        print(f"Loaded cached latents: train={train_latent.shape}, test={test_latent.shape}")
    else:
        train_latent, val_latent, test_latent = compute_and_save_latents(
            encoder, args.encoder_type, data, device, args.output_dir)

    # ─── Train classifier ────────────────────────────────────────────────────
    model = train_classifier(
        clf_cfg, data, device, args.output_dir,
        train_latent, val_latent, test_latent,
        classifier_type=args.classifier_type)

    # Evaluate on test
    evaluate_on_test(model, data, device, args.output_dir, test_latent)

    print(f"\nAll outputs saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
