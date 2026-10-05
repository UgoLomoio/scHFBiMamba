"""
Full BiMamba classifier: input projection -> BiMamba stack -> pooling -> classification head.

This replaces the paper's MLP classifier (795/230/105 -> 13 logits) with a
Bidirectional Mamba architecture. The DAE (if used) feeds its latent representation
as the input sequence; otherwise raw gene expression values are used directly.

Architecture:
  Input: (B, L)  where L=16545 (gene_token) or L=350 (dae_latent)
    |
    +-- InputProjection: Linear(1 -> d_model) per token  -> (B, L, d_model)
    |
    +-- BiMamba Layer 1..N (each with conv + blocks + intra-residual)
    |
    +-- Global residual: layer1_input + layerN_output  (first-to-last)
    |
    +-- Pooling: mean + max over sequence  -> (B, 2*d_model)
    |
    +-- Head: Linear(2*d_model -> hidden) -> ReLU -> Dropout -> Linear(hidden -> 13)
    |
    Output: (B, 13) multi-label logits probabilities
"""
import torch
import torch.nn as nn
from typing import Optional, List

from .bimamba import BiMambaLayer
from config import MambaConfig, N_CLASSES


class BiMambaClassifier(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()
        assert config.pool in ("mean", "max", "mean_max", "attention"), config.pool
        self.config = config
        seq_len = config.seq_len

        self.input_projection = nn.Linear(1, config.d_model)
        # gene identity embedding (one vector per gene position)
        self.gene_embedding = nn.Parameter(torch.zeros(1, seq_len, config.d_model))
        nn.init.normal_(self.gene_embedding, std=0.02)

        self.layers = nn.ModuleList([
            BiMambaLayer(
                d_model=config.d_model, d_state=config.d_state, d_conv=config.d_conv,
                expand=config.expand_factor, num_blocks=config.num_blocks,
                bidirectional=config.bidirectional, use_start_conv=config.use_start_conv,
                use_end_conv=config.use_end_conv, use_intra_residual=config.use_intra_residual,
                conv_kernel=config.conv_kernel,
            ) for _ in range(config.n_layers)
        ])
        self.final_norm = nn.LayerNorm(config.d_model)

        pool_dim = config.d_model * (2 if config.pool in ("mean_max", "attention") else 1)
        if config.pool == "attention":
            self.attention_score = nn.Sequential(
                nn.Linear(config.d_model, config.d_model), nn.Tanh(),
                nn.Linear(config.d_model, 1),
            )
        self.classifier = nn.Sequential(
            nn.Linear(pool_dim, config.classifier_hidden), nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.classifier_hidden, config.n_outputs),
        )

    def _pool(self, x):
        p = self.config.pool
        if p == "mean":
            return x.mean(1)
        if p == "max":
            return x.max(1).values
        if p == "attention":
            a = torch.softmax(self.attention_score(x), dim=1)
            return torch.cat([(x * a).sum(1), x.max(1).values], -1)
        return torch.cat([x.mean(1), x.max(1).values], -1)

    def _encode(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        x = self.input_projection(x) + self.gene_embedding[:, :x.size(1)]
        first = x
        for layer in self.layers:
            x = layer(x)
        if self.config.use_global_residual:
            x = x + first
        return self._pool(self.final_norm(x))

    def forward(self, x):
        return self.classifier(self._encode(x))

    def get_features(self, x):
        return self._encode(x)

    def get_feature_extractor(self):
        class FeatureExtractor(nn.Module):
            def __init__(self, parent):
                super().__init__()
                self.parent = parent
            def forward(self, x):
                return self.parent._encode(x)
        return FeatureExtractor(self)

def build_classifier_from_config(config: MambaConfig) -> BiMambaClassifier:
    """Convenience function to build classifier from config."""
    return BiMambaClassifier(config)
