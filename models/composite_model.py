"""
Composite Model: wraps encoder + classifier as a single nn.Module.

This enables gene-level attribution (SHAP/IG) for ALL models, including
those that use an autoencoder bottleneck. The composite model takes raw
gene expression (B, 16545) as input and produces (B, 13) classification
output, with gradients flowing through both the encoder and classifier.

Also provides methods to extract intermediate representations for UMAP:
  - get_latent(): autoencoder latent (350-dim)
  - get_features(): pre-classification pooled features (128 or 105-dim)
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union

from models.dae import DenoisingAutoencoder
from models.bimamba_ae import BiMambaAutoencoder
from models.classifier import BiMambaClassifier
from models.mlp_classifier import MLPClassifier
from config import MambaConfig, MLPConfig


class CompositeModel(nn.Module):
    """
    Wraps an encoder (Dense DAE or BiMamba AE) and a classifier
    (BiMamba or MLP) into a single module.

    The forward pass: raw genes → encoder → latent → classifier → 13 outputs
    This allows SHAP/IG attribution at the gene level (16545) for all models.
    """

    def __init__(self,
                 encoder: Union[DenoisingAutoencoder, BiMambaAutoencoder],
                 classifier: Union[BiMambaClassifier, MLPClassifier],
                 encoder_type: str = "dense_dae",
                 classifier_type: str = "mamba"):
        """
        Args:
            encoder: trained encoder (Dense DAE or BiMamba AE)
            classifier: trained classifier (BiMamba or MLP)
            encoder_type: "dense_dae" or "bimamba_ae"
            classifier_type: "mamba" or "mlp"
        """
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier
        self.encoder_type = encoder_type
        self.classifier_type = classifier_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass: genes → latent → classification.

        Args:
            x: (B, 16545) raw gene expression values
        Returns:
            (B, 13) multi-label sigmoid probabilities
        """
        latent = self.encoder.encode(x)  # (B, 350)
        output = self.classifier(latent)  # (B, 13)
        return output

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract autoencoder latent representation.

        Args:
            x: (B, 16545) raw gene expression
        Returns:
            (B, 350) latent representation
        """
        return self.encoder.encode(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract pre-classification features (for UMAP).

        For Mamba classifier: pooled features before classification head (128-dim)
        For MLP classifier: last hidden layer output (105-dim)

        Args:
            x: (B, 16545) raw gene expression
        Returns:
            (B, feature_dim) pre-classification features
        """
        latent = self.encoder.encode(x)
        if hasattr(self.classifier, 'get_features'):
            return self.classifier.get_features(latent)
        else:
            # Fallback: use latent as features
            return latent


def build_composite_model(
    encoder_type: str,
    classifier_type: str,
    encoder_config,
    classifier_config,
    device: torch.device,
    encoder_weights_path: str = "",
    classifier_weights_path: str = "",
) -> CompositeModel:
    """
    Build a composite model from configs and optionally load weights.

    Args:
        encoder_type: "dense_dae" or "bimamba_ae"
        classifier_type: "mamba" or "mlp"
        encoder_config: DAEConfig or BiMambaAEConfig
        classifier_config: MambaConfig or MLPConfig
        device: torch device
        encoder_weights_path: path to encoder weights (optional)
        classifier_weights_path: path to classifier weights (optional)

    Returns:
        CompositeModel instance
    """
    # Build encoder
    if encoder_type == "dense_dae":
        encoder = DenoisingAutoencoder(
            input_dim=encoder_config.input_dim,
            encoder_neurons=encoder_config.encoder_neurons,
            decoder_neurons=encoder_config.decoder_neurons,
        )
    else:  # bimamba_ae
        encoder = BiMambaAutoencoder(encoder_config)

    # Build classifier
    if classifier_type == "mamba":
        classifier = BiMambaClassifier(classifier_config)
    else:  # mlp
        classifier = MLPClassifier(classifier_config)

    # Load weights if provided
    if encoder_weights_path:
        encoder.load_state_dict(torch.load(encoder_weights_path, map_location=device))
    if classifier_weights_path:
        classifier.load_state_dict(torch.load(classifier_weights_path, map_location=device))

    model = CompositeModel(encoder, classifier, encoder_type, classifier_type)
    return model.to(device)


def compute_latent_and_features(
    composite: CompositeModel,
    features: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> tuple:
    """
    Compute both latent embeddings and classifier features for a dataset.
    Used for UMAP visualization.

    Args:
        composite: CompositeModel (encoder + classifier)
        features: (N, 16545) raw gene expression
        device: torch device
        batch_size: batch size for inference

    Returns:
        latents: (N, 350) autoencoder latent
        clf_features: (N, feature_dim) pre-classification features
    """
    composite.eval()
    n = len(features)

    # Determine output dims
    with torch.no_grad():
        sample = torch.from_numpy(
            np.asarray(features[:1], dtype=np.float32)
        ).to(device)
        lat_dim = composite.get_latent(sample).shape[1]
        feat_dim = composite.get_features(sample).shape[1]

    latents = np.zeros((n, lat_dim), dtype=np.float32)
    clf_features = np.zeros((n, feat_dim), dtype=np.float32)

    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = torch.from_numpy(
                np.asarray(features[i:i + batch_size], dtype=np.float32)
            ).to(device)
            latents[i:i + len(batch)] = composite.get_latent(batch).cpu().numpy()
            clf_features[i:i + len(batch)] = composite.get_features(batch).cpu().numpy()

    return latents, clf_features
