"""
Bidirectional Mamba layer with convolutions and residual connections.

Architecture per layer (as specified by user):
  1. start_conv: Conv1d at the start of the layer
  2. Multiple Mamba blocks (bidirectional: forward + backward, summed)
  3. end_conv: Conv1d at the end of the layer
  4. Intra-layer residual: layer_out = end_conv_out + start_conv_in

The full BiMamba stack also has a global residual connection between
the first and last layers (implemented in classifier.py).
"""
import torch
import torch.nn as nn
from typing import Optional

from .mamba_block import MambaBlock


class BiMambaLayer(nn.Module):
    """
    A single Bidirectional Mamba layer with:
      - Start/end convolutions
      - Multiple bidirectional Mamba blocks
      - Intra-layer residual connection (start_conv input + end_conv output)

    Bidirectional mechanism:
      Forward:  y_fwd = MambaBlock(x)              — processes L→R
      Backward: y_bwd = flip(MambaBlock(flip(x)))  — processes R→L
      Fusion:   y = y_fwd + y_bwd                  — sum fusion
    """

    def __init__(self, d_model: int = 64, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2,
                 num_blocks: int = 2,
                 bidirectional: bool = True,
                 use_start_conv: bool = True,
                 use_end_conv: bool = True,
                 use_intra_residual: bool = True,
                 conv_kernel: int = 3):
        """
        Args:
            d_model: model dimension
            d_state: SSM state dimension
            d_conv: Mamba internal conv kernel
            expand: Mamba expansion factor
            num_blocks: number of bidirectional Mamba blocks in this layer
            bidirectional: if True, use forward+backward; if False, forward only
            use_start_conv: include Conv1d at layer start
            use_end_conv: include Conv1d at layer end
            use_intra_residual: add start_conv input to end_conv output
            conv_kernel: kernel size for start/end convolutions
        """
        super().__init__()
        self.d_model = d_model
        self.bidirectional = bidirectional
        self.use_start_conv = use_start_conv
        self.use_end_conv = use_end_conv
        self.use_intra_residual = use_intra_residual

        # Start convolution
        if use_start_conv:
            self.start_conv = nn.Conv1d(
                d_model, d_model, kernel_size=conv_kernel,
                padding=conv_kernel // 2, groups=1
            )
        else:
            self.start_conv = None

        # Bidirectional Mamba blocks
        # Forward blocks
        self.forward_blocks = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand)
            for _ in range(num_blocks)
        ])
        # Backward blocks (only if bidirectional)
        if self.bidirectional:
            self.backward_blocks = nn.ModuleList([
                MambaBlock(d_model, d_state, d_conv, expand)
                for _ in range(num_blocks)
            ])
        else:
            self.backward_blocks = None

        # End convolution
        if use_end_conv:
            self.end_conv = nn.Conv1d(
                d_model, d_model, kernel_size=conv_kernel,
                padding=conv_kernel // 2, groups=1
            )
        else:
            self.end_conv = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model)
        Returns:
            (B, L, d_model)
        """
        residual_input = x  # save for intra-layer residual

        # Start convolution
        if self.start_conv is not None:
            # Conv1d expects (B, C, L)
            x = x.transpose(1, 2)  # (B, d_model, L)
            x = self.start_conv(x)
            x = x.transpose(1, 2)  # (B, L, d_model)
            conv_start_out = x  # save for residual

        # Bidirectional Mamba blocks
        for i, fwd_block in enumerate(self.forward_blocks):
            # Forward pass
            x_fwd = fwd_block(x)

            if self.bidirectional and self.backward_blocks is not None:
                # Backward pass: flip sequence, process, flip back
                x_flip = torch.flip(x, dims=[1])
                x_bwd = self.backward_blocks[i](x_flip)
                x_bwd = torch.flip(x_bwd, dims=[1])
                # Sum fusion (bidirectional)
                x = x_fwd + x_bwd
            else:
                x = x_fwd

        # End convolution
        if self.end_conv is not None:
            x = x.transpose(1, 2)  # (B, d_model, L)
            x = self.end_conv(x)
            x = x.transpose(1, 2)  # (B, L, d_model)

        # Intra-layer residual: start_conv output + end_conv output
        if self.use_intra_residual and self.start_conv is not None:
            x = x + conv_start_out

        return x
