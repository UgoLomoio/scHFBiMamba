"""
Bidirectional Mamba Autoencoder (BiMamba AE).

Replaces the Dense DAE for M3/M4. Instead of fully-connected layers,
the encoder and decoder use BiMamba layers that process genes as a
sequence, capturing gene-gene interactions through selective state-space
recurrence in both directions.

Architecture:
  Encoder: (B, 16545) → input_proj → 3 BiMamba layers → mean+max pool → Linear(128→350)
  Decoder: (B, 350) → Linear(350→64) → position embedding → 3 BiMamba layers → Linear(64→1) per gene → (B, 16545)

The decoder uses learnable position embeddings to distinguish gene positions
when expanding from a single latent vector back to the full gene sequence.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional

from .bimamba import BiMambaLayer
from config import BiMambaAEConfig, N_GENES
from torch.utils.checkpoint import checkpoint


import torch
import torch.nn as nn
import torch.nn.functional as F

class BiMambaEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.stem = nn.Sequential(
            nn.Linear(config.input_dim, config.stem_dim),
            nn.LayerNorm(config.stem_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        self.to_tokens = nn.Linear(
            config.stem_dim,
            config.n_latent_tokens * config.d_model,
        )

        self.layers = nn.ModuleList([
            BiMambaLayer(
                d_model=config.d_model,
                d_state=config.d_state,
                d_conv=config.d_conv,
                expand=config.expand_factor,
                num_blocks=1,
                bidirectional=config.bidirectional,
                use_start_conv=True,
                use_end_conv=True,
                use_intra_residual=True,
                conv_kernel=3,
            )
            for _ in range(config.n_layers)
        ])

        self.to_latent = nn.Sequential(
            nn.LayerNorm(config.n_latent_tokens * config.d_model),
            nn.Linear(
                config.n_latent_tokens * config.d_model,
                config.latent_dim,
            ),
        )

    def forward(self, x):
        h = self.stem(x)  # (B, 1024)

        tokens = self.to_tokens(h)
        tokens = tokens.view(
            x.shape[0],
            self.config.n_latent_tokens,
            self.config.d_model,
        )  # (B, 32, 64)

        for layer in self.layers:
            tokens = layer(tokens)

        return self.to_latent(tokens.flatten(1))  # (B, 350)

class MLPDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.stem_dim),
            nn.LayerNorm(config.stem_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.stem_dim, config.input_dim),
        )

    def forward(self, z):
        return self.decoder(z)  # (B, 16545)

class BiMambaAutoencoder(nn.Module):
    """
    Full Bidirectional Mamba Autoencoder: encoder + decoder.
    Used as the encoder for M3 (BiMamba AE + MLP) and M4 (BiMamba AE + Mamba).
    """

    def __init__(self, config: BiMambaAEConfig = None):
        super().__init__()
        if config is None:
            config = BiMambaAEConfig()
        self.config = config
        self.encoder = BiMambaEncoder(config)
        self.decoder = MLPDecoder(config)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to reconstruction."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward: encode then decode."""
        z = self.encode(x)
        recon = self.decode(z)
        return recon

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Get latent representation only."""
        return self.encode(x)


def compute_bimamba_ae_latent(ae: BiMambaAutoencoder,
                               features: np.ndarray,
                               device: torch.device,
                               batch_size: int = 1024) -> np.ndarray:
    """
    Compute BiMamba AE latent features for a dataset.

    Args:
        ae: trained BiMambaAutoencoder
        features: (N, input_dim) numpy array or memmap
        device: torch device
        batch_size: batch size for inference

    Returns:
        (N, latent_dim) numpy array of latent features
    """
    ae.eval()
    n = len(features)
    latent_dim = ae.config.latent_dim
    latents = np.zeros((n, latent_dim), dtype=np.float32)

    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = torch.from_numpy(
                np.asarray(features[i:i + batch_size], dtype=np.float32)
            ).to(device)
            z = ae.get_latent(batch)
            latents[i:i + len(batch)] = z.cpu().numpy()

    return latents
