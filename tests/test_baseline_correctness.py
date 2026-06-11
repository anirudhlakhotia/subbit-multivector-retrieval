"""Correctness tests for the ITQ + RaBitQ 100k baselines (paper tab:pareto).

These pin the math so the re-baselined-to-augmented numbers are trustworthy:
- ITQ: the learned rotation is orthogonal AND reduces the sign-quantization loss
  vs a random rotation (otherwise ITQ is doing nothing).
- RaBitQ: the rotation is orthogonal/deterministic AND the per-token correction
  scalar recovers the true inner product (the whole point of RaBitQ).
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------- RaBitQ
from evaluation import eval_rabitq_100k_aug as rabitq


def test_rabitq_R_orthogonal_and_frobenius():
    R = rabitq.random_orthogonal(128, 42, torch.device("cpu"))
    assert (R @ R.T - torch.eye(128)).abs().max().item() < 1e-4
    assert abs(R.norm().item() - 128 ** 0.5) < 1e-3   # ||R||_F = sqrt(128)


def test_rabitq_R_deterministic():
    a = rabitq.random_orthogonal(128, 42, torch.device("cpu"))
    b = rabitq.random_orthogonal(128, 42, torch.device("cpu"))
    assert torch.allclose(a, b)


def test_rabitq_binary_is_pm1():
    R = rabitq.random_orthogonal(128, 42, torch.device("cpu"))
    y = torch.randn(64, 128) @ R.T
    b, _ = rabitq.rabitq_encode(y)
    assert set(torch.unique(b).tolist()) <= {-1.0, 1.0}


def test_rabitq_self_inner_product_exact():
    """Definitive correctness: for q=d the estimator recovers <d,d>=||d||^2
    EXACTLY (correction*<Rd,sign(Rd)> = ||Rd||^2). This is the formula's fixed
    point and proves the norm/vdot/correction math is right."""
    torch.manual_seed(0)
    R = rabitq.random_orthogonal(128, 42, torch.device("cpu"))
    D = torch.randn(300, 128)
    y = D @ R.T
    binary, correction = rabitq.rabitq_encode(y)
    approx = correction * (y * binary).sum(-1)        # q == d
    true_self = (D * D).sum(-1)                        # ||d||^2
    assert torch.allclose(approx, true_self, rtol=1e-3, atol=1e-3), \
        f"max err {(approx - true_self).abs().max().item():.2e}"


def test_rabitq_correction_unbiased_and_correlated():
    """For random independent pairs the estimator is UNBIASED (mean error ~0)
    and positively correlated with the truth. (Per-pair variance is inherent to
    a 128-bit code on near-orthogonal random vectors; ranking is robust to it.)"""
    torch.manual_seed(0)
    R = rabitq.random_orthogonal(128, 42, torch.device("cpu"))
    D = torch.randn(2000, 128)
    Q = torch.randn(2000, 128)
    true_ip = (Q * D).sum(-1)
    y = D @ R.T
    binary, correction = rabitq.rabitq_encode(y)
    approx = correction * ((Q @ R.T) * binary).sum(-1)
    corr = float(np.corrcoef(approx.numpy(), true_ip.numpy())[0, 1])
    bias = float((approx - true_ip).mean() / true_ip.std())   # unbiasedness
    assert corr > 0.7, f"RaBitQ recovery corr={corr:.3f}"
    assert abs(bias) < 0.1, f"RaBitQ estimator biased: {bias:.3f}"


def test_rabitq_sqrt_dim_constant():
    assert abs(rabitq.SQRT_DIM - 128 ** 0.5) < 1e-9
    assert rabitq.DIM == 128


# ---------------------------------------------------------------- ITQ
from evaluation import eval_itq_baseline as itq


def _sign_quant_loss(V, Q):
    P = V @ Q
    B = np.sign(P)
    B[B == 0] = 1.0
    return float(np.linalg.norm(B - P) ** 2)


def test_itq_rotation_orthogonal():
    rng = np.random.default_rng(0)
    V = rng.standard_normal((2000, 64)).astype(np.float32)
    Q = itq.fit_itq(V, n_iters=30, seed=42)
    assert Q.shape == (64, 64)
    assert np.abs(Q @ Q.T - np.eye(64)).max() < 1e-4


def test_itq_reduces_quantization_loss_vs_random():
    rng = np.random.default_rng(0)
    V = rng.standard_normal((2000, 64)).astype(np.float32)
    Q_itq = itq.fit_itq(V, n_iters=40, seed=42)
    Q_rand, _ = np.linalg.qr(rng.standard_normal((64, 64)))
    assert _sign_quant_loss(V, Q_itq) < _sign_quant_loss(V, Q_rand), \
        "ITQ rotation did not reduce sign-quantization loss vs a random rotation"
