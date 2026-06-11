"""Utility functions: seeding, device selection, Stiefel projection, logging."""
from __future__ import annotations

import os
import random
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def seed_everything(seed: int = 42) -> None:
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic mode (slightly slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Set all random seeds to {seed}")


def get_device(preference: str = "auto") -> torch.device:
    """Select the best available device.

    Args:
        preference: One of 'auto', 'cpu', 'mps', 'cuda'.

    Returns:
        The selected torch.device.
    """
    if preference == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(preference)

    logger.info(f"Using device: {device}")
    return device


def project_to_stiefel(R: torch.Tensor) -> torch.Tensor:
    """Project a matrix onto the Stiefel manifold (orthonormal rows).

    Given R ∈ ℝ^{r×d}, returns the closest matrix with orthonormal rows
    via polar decomposition (SVD). At high r, R can drift to states where
    on-device SVD (especially MPS) fails to converge for ill-conditioned
    inputs. We do the SVD on CPU in fp64 for stability, then cast back. If
    SVD still fails, we fall back to QR-based orthogonalisation, which is
    not the closest orthogonal but always succeeds.

    Args:
        R: Matrix of shape (r, d) where r <= d.

    Returns:
        R_orth: Matrix of shape (r, d) with orthonormal rows.
    """
    device, dtype = R.device, R.dtype
    # Chain device→dtype: MPS doesn't allow fp64, so we must reach CPU
    # before casting to fp64. Combined kwargs in a single .to() call can
    # confuse the MPS backend.
    R_cpu = R.detach().cpu().to(torch.float64)
    try:
        U, _, Vt = torch.linalg.svd(R_cpu, full_matrices=False)
        result = U @ Vt
    except torch._C._LinAlgError:
        # QR fallback: for an r×d matrix with r ≤ d, QR of R^T gives
        # Q ∈ ℝ^{d×r} with orthonormal columns, so Q^T has orthonormal rows.
        Q, _ = torch.linalg.qr(R_cpu.T, mode="reduced")
        result = Q.T
    return result.to(dtype).to(device)


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def ensure_dir(path: str | Path) -> Path:
    """Create directory and parents if they don't exist. Returns the Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def bits_per_dim(projected_dim: int, input_dim: int = 128) -> float:
    """Compute the bits per dimension for a given projected dimension.

    After projecting from input_dim to projected_dim, each dimension
    is stored as 1 bit, so total bits = projected_dim and
    bits/dim = projected_dim / input_dim.
    """
    return projected_dim / input_dim


def bytes_per_token(projected_dim: int) -> float:
    """Compute storage cost per token in bytes.

    Each token is stored as projected_dim bits = projected_dim/8 bytes.
    """
    return projected_dim / 8


def compression_ratio(projected_dim: int, input_dim: int = 128, precision: int = 32) -> float:
    """Compute compression ratio vs full-precision storage.

    Args:
        projected_dim: Number of dimensions after projection.
        input_dim: Original embedding dimension.
        precision: Original precision in bits (32 for FP32).

    Returns:
        Compression ratio (e.g., 64× for projected_dim=64, input_dim=128, precision=32).
    """
    original_bits = input_dim * precision
    compressed_bits = projected_dim  # 1 bit per projected dim
    return original_bits / compressed_bits


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Configure logging for the project."""
    handlers = [logging.StreamHandler()]
    if log_file:
        ensure_dir(Path(log_file).parent)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


def format_metrics(metrics: dict, prefix: str = "") -> str:
    """Format a metrics dictionary for logging. Keys prefixed with '_' are skipped."""
    parts = []
    for k, v in sorted(metrics.items()):
        if k.startswith("_"):
            continue
        name = f"{prefix}/{k}" if prefix else k
        if isinstance(v, float):
            parts.append(f"{name}: {v:.4f}")
        else:
            parts.append(f"{name}: {v}")
    return " | ".join(parts)
