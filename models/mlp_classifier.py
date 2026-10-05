"""
MLP Classifier — matches the paper's architecture exactly.

Paper architecture (from UP_models.py):
  Input: 350 (DAE latent)
  Hidden: 795 → 230 → 105 (ReLU + Dropout)
  Output: 13 (Logits)
  Loss: Macro F1-loss

This is used for M3 (BiMamba AE + MLP) to compare Mamba vs MLP classifiers
on the same autoencoder latent representation.
"""
import torch
import torch.nn as nn
from typing import Optional

from config import MLPConfig, N_CLASSES


class MLPClassifier(nn.Module):
    """
    Multi-layer perceptron classifier matching the paper's architecture.

    Architecture: input_dim → 795 → 230 → 105 → 13 (logits)
    With ReLU activations and dropout between hidden layers.
    """

    def __init__(self, config: MLPConfig = None):
        """
        Args:
            config: MLPConfig with hyperparameters
        """
        super().__init__()
        if config is None:
            config = MLPConfig()
        self.config = config

        # Build hidden layers
        layers = []
        prev = config.input_dim
        for n in config.hidden_neurons:
            layers.append(nn.Linear(prev, n))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout))
            prev = n

        # Output layer
        layers.append(nn.Linear(prev, config.n_outputs))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim) latent representation from encoder
        Returns:
            (B, 13) multi-label logits
        """
        return self.network(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract pre-classification features (last hidden layer output).
        Used for UMAP visualization.

        Args:
            x: (B, input_dim) latent representation
        Returns:
            (B, last_hidden_dim) features before output layer
        """
        # Run through all layers except the last layer (output linear)
        for layer in list(self.network)[:-1 ]:  # Exclude last layer
            x = layer(x)
        return x


def build_mlp_from_config(config: MLPConfig) -> MLPClassifier:
    """Convenience function to build MLP from config."""
    return MLPClassifier(config)
