"""
Denoising Autoencoder (DAE) — PyTorch re-implementation of the paper's TensorFlow DAE.

Paper architecture (from UP_models.py):
  Encoder: 16545 -> 5000 -> 2400 -> 350  (ReLU)
  Decoder: 350 -> 2200 -> 5150 -> 16545  (ReLU on hidden, linear on output)
  Loss: MSE (L2)
  Asymmetric encoder/decoder (not symmetric, per paper)
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple


class DenoisingAutoencoder(nn.Module):
    """
    Denoising Autoencoder for dimensionality reduction of snRNA-seq data.
    Re-implements the paper's DEAutoencoder in PyTorch.

    The encoder compresses 16,545 gene expression values into a low-dimensional
    latent representation (350 dims). The decoder reconstructs the original
    expression. The latent representation is used as input to the BiMamba
    classifier (in the DAE-latent ablation configuration).
    """

    def __init__(self, input_dim: int = 16545,
                 encoder_neurons: List[int] = None,
                 decoder_neurons: List[int] = None):
        """
        Args:
            input_dim: number of input genes (16545)
            encoder_neurons: list of encoder layer sizes, e.g. [5000, 2400, 350]
                             Last element is the latent dimension.
            decoder_neurons: list of decoder hidden layer sizes, e.g. [2200, 5150]
                             Output layer (input_dim) is added automatically.
        """
        super().__init__()
        if encoder_neurons is None:
            encoder_neurons = [5000, 2400, 350]
        if decoder_neurons is None:
            decoder_neurons = [2200, 5150]

        self.input_dim = input_dim
        self.encoder_neurons = encoder_neurons
        self.decoder_neurons = decoder_neurons
        self.latent_dim = encoder_neurons[-1]

        # Build encoder
        encoder_layers = []
        prev = input_dim
        for n in encoder_neurons:
            encoder_layers.append(nn.Linear(prev, n))
            encoder_layers.append(nn.ReLU())
            prev = n
        self.encoder = nn.Sequential(*encoder_layers)

        # Build decoder
        decoder_layers = []
        prev = self.latent_dim
        for n in decoder_neurons:
            decoder_layers.append(nn.Linear(prev, n))
            decoder_layers.append(nn.ReLU())
            prev = n
        # Final reconstruction layer (linear, no activation)
        decoder_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to reconstruction."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass: encode then decode."""
        z = self.encode(x)
        recon = self.decode(z)
        return recon

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Get latent representation only (for feeding to classifier)."""
        return self.encode(x)


def compute_dae_latent(dae: DenoisingAutoencoder,
                       features: np.ndarray,
                       device: torch.device,
                       batch_size: int = 1024) -> np.ndarray:
    """
    Compute DAE latent features for a dataset.
    Used after DAE training to generate inputs for the DAE-latent Mamba config.

    Args:
        dae: trained DenoisingAutoencoder
        features: (N, input_dim) numpy array or memmap
        device: torch device
        batch_size: batch size for inference

    Returns:
        (N, latent_dim) numpy array of latent features
    """
    dae.eval()
    n = len(features)
    latent_dim = dae.latent_dim
    latents = np.zeros((n, latent_dim), dtype=np.float32)

    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = torch.from_numpy(
                np.asarray(features[i:i + batch_size], dtype=np.float32)
            ).to(device)
            z = dae.get_latent(batch)
            latents[i:i + len(batch)] = z.cpu().numpy()

    return latents
