"""Unit tests for the PLAID-style baseline (`src/subbit/baselines_plaid.py`)."""
from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.subbit.baselines_plaid import PLAIDConfig, PLAIDQuantizer
from src.subbit.scoring import maxsim


def _make_synth_tokens(n: int, d: int, n_clusters: int, seed: int = 0) -> torch.Tensor:
    """Gaussian mixture with well-separated modes — easy for k-means to latch onto."""
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(n_clusters, d, generator=g) * 5.0
    assignments = torch.randint(0, n_clusters, (n,), generator=g)
    noise = torch.randn(n, d, generator=g) * 0.3
    return centers[assignments] + noise


# ---------------------------------------------------------------------------
# Config arithmetic
# ---------------------------------------------------------------------------


def test_bytes_per_token_default_config():
    cfg = PLAIDConfig(n_centroids=65536, residual_bits=2)
    assert cfg.centroid_bits == 16
    assert cfg.bytes_per_token(128) == (16 + 128 * 2) / 8.0  # 34.0


@pytest.mark.parametrize("b,expected", [(1, 18.0), (2, 34.0), (4, 66.0)])
def test_bytes_per_token_sweeps_b(b: int, expected: float):
    cfg = PLAIDConfig(n_centroids=65536, residual_bits=b)
    assert cfg.bytes_per_token(128) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Training + encode/decode
# ---------------------------------------------------------------------------


def test_train_then_encode_decode_shapes():
    d, n = 16, 512
    x = _make_synth_tokens(n, d, n_clusters=8)
    cfg = PLAIDConfig(n_centroids=8, residual_bits=2, kmeans_iters=5, kmeans_sample=n)
    pq = PLAIDQuantizer(d=d, config=cfg)
    pq.train(x)

    assert pq.centroids.shape == (8, d)
    assert pq.bucket_edges.shape == (d, 3)   # 2^2 - 1
    assert pq.bucket_centers.shape == (d, 4)  # 2^2

    ids, residual = pq.encode(x)
    assert ids.shape == (n,)
    assert residual.shape == (n, d)
    assert residual.dtype == torch.uint8
    assert ids.min() >= 0 and ids.max() < 8
    assert residual.min() >= 0 and residual.max() < 4

    reconstructed = pq.decode(ids, residual)
    assert reconstructed.shape == (n, d)


def test_reconstruction_beats_centroid_only():
    """Adding residuals must reduce MSE vs. using centroids alone."""
    d, n = 16, 2048
    x = _make_synth_tokens(n, d, n_clusters=16)
    cfg = PLAIDConfig(n_centroids=16, residual_bits=2, kmeans_iters=10, kmeans_sample=n)
    pq = PLAIDQuantizer(d=d, config=cfg)
    pq.train(x)

    with_residual = pq.encode_decode(x)
    ids, _ = pq.encode(x)
    centroid_only = pq.centroids[ids]

    mse_centroid = (x - centroid_only).pow(2).mean().item()
    mse_full = (x - with_residual).pow(2).mean().item()
    assert mse_full < mse_centroid, (
        f"residual decode should strictly improve reconstruction "
        f"(centroid-only={mse_centroid:.4f}, with-residual={mse_full:.4f})"
    )


def test_more_residual_bits_reduces_error():
    """Monotone: b=4 reconstructs at least as well as b=2 >= b=1."""
    d, n = 16, 2048
    x = _make_synth_tokens(n, d, n_clusters=16, seed=1)
    errs = {}
    for b in (1, 2, 4):
        cfg = PLAIDConfig(n_centroids=16, residual_bits=b, kmeans_iters=10, kmeans_sample=n)
        pq = PLAIDQuantizer(d=d, config=cfg)
        pq.train(x)
        errs[b] = (x - pq.encode_decode(x)).pow(2).mean().item()
    assert errs[4] <= errs[2] <= errs[1] * 1.05  # 5% slack — noisy with small n


# ---------------------------------------------------------------------------
# Behavior under scoring
# ---------------------------------------------------------------------------


def test_maxsim_reconstruction_correlates_with_fp():
    """PLAID-reconstructed tokens must preserve MaxSim rank order on clusterable data.

    Real ColBERT tokens form tight modes in 128d (PLAID's whole premise); pure
    `torch.randn` is an adversarial input. We use a clustered synthetic source so
    the test measures algorithm correctness, not data pathology.
    """
    torch.manual_seed(0)
    d = 32
    n_docs = 40
    tokens_per_doc = 24
    # Each document draws tokens from a shared Gaussian-mixture vocabulary.
    vocab = _make_synth_tokens(n=200, d=d, n_clusters=20, seed=7)
    doc_token_idx = torch.randint(0, vocab.shape[0], (n_docs, tokens_per_doc))
    docs = [vocab[doc_token_idx[i]] + 0.05 * torch.randn(tokens_per_doc, d)
            for i in range(n_docs)]
    all_tokens = torch.cat(docs, dim=0)
    q = vocab[torch.randint(0, vocab.shape[0], (8,))] + 0.05 * torch.randn(8, d)

    cfg = PLAIDConfig(n_centroids=64, residual_bits=2, kmeans_iters=15,
                       kmeans_sample=all_tokens.shape[0])
    pq = PLAIDQuantizer(d=d, config=cfg)
    pq.train(all_tokens)

    fp_scores = torch.tensor([maxsim(q, doc).item() for doc in docs])
    plaid_scores = torch.tensor([
        maxsim(q, pq.encode_decode(doc)).item() for doc in docs
    ])
    fp_c = fp_scores - fp_scores.mean()
    pl_c = plaid_scores - plaid_scores.mean()
    corr = (fp_c @ pl_c) / (fp_c.norm() * pl_c.norm() + 1e-9)
    assert corr.item() > 0.9, f"expected strong MaxSim correlation, got {corr.item():.3f}"


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_save_load_roundtrip(tmp_path):
    d, n = 16, 512
    x = _make_synth_tokens(n, d, n_clusters=8)
    cfg = PLAIDConfig(n_centroids=8, residual_bits=2, kmeans_iters=5, kmeans_sample=n)
    pq = PLAIDQuantizer(d=d, config=cfg)
    pq.train(x)

    path = tmp_path / "plaid.pt"
    pq.save(path)
    loaded = PLAIDQuantizer.load(path)

    assert loaded.d == pq.d
    assert loaded.config == pq.config
    torch.testing.assert_close(loaded.centroids, pq.centroids)
    torch.testing.assert_close(loaded.bucket_edges, pq.bucket_edges)
    torch.testing.assert_close(loaded.bucket_centers, pq.bucket_centers)

    torch.testing.assert_close(pq.encode_decode(x), loaded.encode_decode(x))


def test_shared_centroids_bit_exact_vs_independent_fit():
    """Two quantizers differing only in `b` must produce identical centroids.

    This is the correctness claim behind the script's centroid-sharing
    optimization: given (tokens, seed, C, kmeans_iters), centroids are a pure
    function of the k-means run — `b` never enters that computation.
    """
    d, n = 16, 2048
    x = _make_synth_tokens(n, d, n_clusters=16)

    # Fit independently for b=2 and b=4.
    cfg2 = PLAIDConfig(n_centroids=16, residual_bits=2, kmeans_iters=10, kmeans_sample=n, seed=7)
    cfg4 = PLAIDConfig(n_centroids=16, residual_bits=4, kmeans_iters=10, kmeans_sample=n, seed=7)
    pq2 = PLAIDQuantizer(d=d, config=cfg2); pq2.train(x)
    pq4 = PLAIDQuantizer(d=d, config=cfg4); pq4.train(x)
    torch.testing.assert_close(pq2.centroids, pq4.centroids)

    # Now re-fit b=4 sharing pq2's centroids — must match the independent fit bit-exactly.
    pq4_shared = PLAIDQuantizer(d=d, config=cfg4)
    pq4_shared.train(x, centroids=pq2.centroids)
    torch.testing.assert_close(pq4_shared.centroids, pq4.centroids)
    torch.testing.assert_close(pq4_shared.bucket_edges, pq4.bucket_edges)
    torch.testing.assert_close(pq4_shared.bucket_centers, pq4.bucket_centers)


def test_shared_centroids_shape_mismatch_raises():
    cfg = PLAIDConfig(n_centroids=16, residual_bits=2, kmeans_iters=2, kmeans_sample=128)
    pq = PLAIDQuantizer(d=8, config=cfg)
    bogus = torch.randn(32, 8)  # C=32 doesn't match cfg.n_centroids=16
    with pytest.raises(ValueError, match="shared centroids shape"):
        pq.train(torch.randn(128, 8), centroids=bogus)


def test_encode_before_train_raises():
    pq = PLAIDQuantizer(d=16)
    with pytest.raises(RuntimeError, match="not trained"):
        pq.encode(torch.randn(4, 16))
