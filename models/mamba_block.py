"""
Mamba block wrapper using mamba_ssm (GPU CUDA kernels).

This module provides a unified interface to the Mamba selective state-space
model block. On GPU, it uses the optimized mamba_ssm.Mamba implementation
with CUDA selective scan kernels.

The Mamba block implements:
  - Input projection (d_model -> d_inner = expand * d_model)
  - Conv1d (d_conv kernel) for local feature extraction
  - Selective SSM scan (input-dependent A, B, C, delta)
  - Output projection (d_inner -> d_model)

Reference: Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective
State Spaces" (2023).
"""
import torch
import torch.nn as nn

# mamba_ssm requires a CUDA GPU. The import is deferred to __init__ so that
# --help and other CLI commands work even without mamba_ssm installed.
_Mamba = None


def _get_mamba_cls():
    """Lazily import mamba_ssm.Mamba. Called at model instantiation time."""
    global _Mamba
    if _Mamba is None:
        try:
            from mamba_ssm import Mamba2 as Mamba
            _Mamba = Mamba
        except ImportError:
            raise ImportError(
                "mamba_ssm is required to run the BiMamba model. "
                "Install it on a CUDA-capable machine:\n"
                "  pip install causal-conv1d>=1.4.0\n"
                "  pip install mamba-ssm>=2.0.0\n"
                "See: https://github.com/state-spaces/mamba"
            )
    return _Mamba


class MambaBlock(nn.Module):
    """
    Wrapper around mamba_ssm.Mamba providing a clean interface.

    The Mamba block processes a sequence (B, L, d_model) and returns
    (B, L, d_model) with the same dimensions, applying the selective
    state-space model with input-dependent gating.
    """

    def __init__(self, d_model: int = 64, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2):
        """
        Args:
            d_model: model dimension (input and output)
            d_state: SSM state dimension (N)
            d_conv: convolution kernel size in Mamba block
            expand: inner expansion factor (d_inner = expand * d_model)
        """
        super().__init__()
        MambaCls = _get_mamba_cls()
        self.mamba = MambaCls(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model)
        Returns:
            (B, L, d_model)
        """
        return self.mamba(x)
