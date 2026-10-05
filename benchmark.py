"""
Efficiency benchmark for the classifier ablation: parameter counts, FLOPs,
and inference time.

Timing protocol (identical for all comparable models):
  - same device, batch size, and precision (AMP flag) for every model
  - `warmup` untimed iterations first
  - `torch.cuda.synchronize()` before/after each timed iteration
  - report mean +/- std over `iters` timed iterations

FLOPs are analytical estimates (documented per model type below); they are
reported as `--` where the notion is not meaningful (XGBoost).
"""
import time
import numpy as np
import torch
from typing import Callable, Dict, Optional


def count_parameters(model: torch.nn.Module) -> int:
    """Total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_flops_torch(model: torch.nn.Module, kind: str,
                         seq_len: int = 350,
                         n_nodes: Optional[int] = None,
                         n_edges: Optional[int] = None,
                         hidden_dim: int = 128) -> Optional[float]:
    """
    Analytical FLOP estimate per classification of `seq_len` tokens (sequence
    models), per sample (MLP), or per full-graph pass (GCN/GAT).

    Conventions (multiply-accumulate = 2 FLOPs):
      - "mlp":      2 * n_params                       (each weight used once)
      - "sequence": 2 * n_params * seq_len             (each weight used once
                    per token; standard transformer-style estimate for the
                    Mamba/BiMamba stack incl. projections and convolutions)
      - "gcn"/"gat": 2 * n_params * n_nodes (feature transforms)
                    + 2 * n_edges * hidden_dim per layer (neighborhood
                    aggregation / attention messages)

    Returns:
        FLOP count (float), or None if not meaningful for this model kind
    """
    n_params = count_parameters(model)
    if kind == "mlp":
        return 2.0 * n_params
    if kind == "sequence":
        return 2.0 * n_params * seq_len
    if kind in ("gcn", "gat"):
        if n_nodes is None or n_edges is None:
            return None
        n_layers = len(getattr(model, 'layers', [])) or 1
        transform = 2.0 * n_params * n_nodes
        aggregate = 2.0 * n_edges * hidden_dim * n_layers
        return transform + aggregate
    return None


def measure_inference_time(forward_fn: Callable[[], None], device: torch.device,
                           warmup: int = 10, iters: int = 50) -> Dict[str, float]:
    """
    Time a forward callable with warm-up iterations and CUDA synchronization.

    Args:
        forward_fn: zero-argument callable running one forward pass
        device: torch device (CUDA sync applied only on CUDA)
        warmup: number of untimed warm-up iterations
        iters: number of timed iterations

    Returns:
        dict with mean_ms and std_ms
    """
    is_cuda = device.type == 'cuda'

    with torch.no_grad():
        for _ in range(warmup):
            forward_fn()
        if is_cuda:
            torch.cuda.synchronize()

        times = []
        for _ in range(iters):
            if is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            forward_fn()
            if is_cuda:
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)

    return {'mean_ms': float(np.mean(times)), 'std_ms': float(np.std(times))}


def benchmark_torch_model(model: torch.nn.Module, input_shape: tuple,
                          device: torch.device, warmup: int = 10,
                          iters: int = 50, use_amp: bool = False,
                          extra_input: torch.Tensor = None
                          ) -> Dict[str, float]:
    """
    Benchmark a torch classifier on random inputs of the given shape.

    Args:
        model: classifier in eval mode
        input_shape: input tensor shape including batch, e.g. (1024, 350)
        device: torch device
        warmup/iters: timing protocol
        use_amp: mixed precision flag (same for all comparable models)
        extra_input: optional second forward argument (e.g. graph adjacency
            or edge index for GNN arms)

    Returns:
        dict with mean_ms and std_ms
    """
    model.eval()
    x = torch.randn(*input_shape, device=device)
    if extra_input is not None:
        extra_input = extra_input.to(device)

    is_cuda = device.type == 'cuda'
    device_type = 'cuda' if is_cuda else 'cpu'

    def forward():
        if use_amp:
            with torch.autocast(device_type=device_type, enabled=True):
                if extra_input is not None:
                    model(x, extra_input)
                else:
                    model(x)
        else:
            if extra_input is not None:
                model(x, extra_input)
            else:
                model(x)

    return measure_inference_time(forward, device, warmup=warmup, iters=iters)