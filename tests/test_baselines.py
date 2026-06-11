"""PQ baseline compression method tests."""

import pytest
import torch
import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.subbit.baselines import (
    product_quantize,
    pq_decode,
    pq_maxsim,
    pq_lookup_tables,
    random_projection_binary,
    identity_truncation,
    pca_projection_binary,
)
from src.subbit.scoring import maxsim


# ---------------------------------------------------------------------------
# Product Quantization Tests
# ---------------------------------------------------------------------------


class TestProductQuantize:
    """Tests for FAISS Product Quantization baseline."""

    @pytest.fixture
    def corpus_data(self):
        """Synthetic corpus for PQ training (needs enough data for k-means)."""
        torch.manual_seed(42)
        return torch.randn(1000, 128)

    def test_codes_shape(self, corpus_data):
        """PQ codes should have shape (N, n_subquantizers)."""
        codes, pq = product_quantize(corpus_data, n_subquantizers=8, n_bits=8)
        assert codes.shape == (1000, 8)

    def test_codes_dtype(self, corpus_data):
        """PQ codes should be long tensor."""
        codes, pq = product_quantize(corpus_data, n_subquantizers=8, n_bits=8)
        assert codes.dtype == torch.long

    def test_codes_in_range(self, corpus_data):
        """With 8 bits, codes should be in [0, 255]."""
        codes, pq = product_quantize(corpus_data, n_subquantizers=8, n_bits=8)
        assert codes.min().item() >= 0
        assert codes.max().item() <= 255

    def test_decode_shape(self, corpus_data):
        """Decoded vectors should match original dimensionality."""
        codes, pq = product_quantize(corpus_data, n_subquantizers=8, n_bits=8)
        reconstructed = pq_decode(codes, pq)
        assert reconstructed.shape == (1000, 128)

    def test_decode_dtype(self, corpus_data):
        """Decoded vectors should be float tensors."""
        codes, pq = product_quantize(corpus_data, n_subquantizers=8, n_bits=8)
        reconstructed = pq_decode(codes, pq)
        assert reconstructed.dtype == torch.float32

    def test_round_trip_error_bounded(self, corpus_data):
        """PQ reconstruction error should be reasonable (not huge)."""
        codes, pq = product_quantize(corpus_data, n_subquantizers=8, n_bits=8)
        reconstructed = pq_decode(codes, pq)
        mse = (corpus_data - reconstructed).pow(2).mean().item()
        # MSE should be less than input variance (PQ should help)
        input_var = corpus_data.var().item()
        assert mse < input_var, f"PQ MSE ({mse:.4f}) > input variance ({input_var:.4f})"

    def test_fit_data_separate(self, corpus_data):
        """Should use fit_data for training when provided."""
        test_data = torch.randn(100, 128)
        codes, pq = product_quantize(
            test_data, n_subquantizers=8, n_bits=8, fit_data=corpus_data
        )
        assert codes.shape == (100, 8)

    def test_different_n_sub(self, corpus_data):
        """Different sub-quantizer counts should work."""
        for n_sub in [4, 8, 16]:
            codes, pq = product_quantize(
                corpus_data, n_subquantizers=n_sub, n_bits=8
            )
            assert codes.shape == (1000, n_sub)

    def test_maxsim_compatible(self, corpus_data):
        """PQ-reconstructed embeddings should work with maxsim scoring."""
        q = torch.randn(5, 128)
        d = corpus_data[:50]
        codes, pq = product_quantize(d, n_subquantizers=8, n_bits=8, fit_data=corpus_data)
        d_recon = pq_decode(codes, pq)
        score = maxsim(q, d_recon)
        assert score.shape == ()
        assert torch.isfinite(score)

    def test_adc_matches_reconstruction_maxsim(self, corpus_data):
        """ADC scoring should match decode-then-score exactly."""
        q = torch.randn(5, 128)
        d = corpus_data[:50]
        codes, pq = product_quantize(d, n_subquantizers=8, n_bits=8, fit_data=corpus_data)

        score_adc = pq_maxsim(q, codes, pq)
        score_recon = maxsim(q, pq_decode(codes, pq))

        assert score_adc.item() == pytest.approx(score_recon.item(), abs=1e-5)


# ---------------------------------------------------------------------------
# Random Projection Binary Tests
# ---------------------------------------------------------------------------


class TestRandomProjectionBinary:
    def test_output_shape(self):
        embs = torch.randn(10, 128)
        binary, R = random_projection_binary(embs, projected_dim=64)
        assert binary.shape == (10, 64)
        assert R.shape == (64, 128)

    def test_values_are_pm1(self):
        embs = torch.randn(20, 128)
        binary, _ = random_projection_binary(embs, projected_dim=32)
        assert torch.all((binary == 1.0) | (binary == -1.0))

    def test_deterministic_with_same_seed(self):
        embs = torch.randn(10, 128)
        b1, _ = random_projection_binary(embs, projected_dim=64, seed=42)
        b2, _ = random_projection_binary(embs, projected_dim=64, seed=42)
        assert torch.equal(b1, b2)

    def test_different_seed_gives_different_result(self):
        embs = torch.randn(10, 128)
        b1, _ = random_projection_binary(embs, projected_dim=64, seed=42)
        b2, _ = random_projection_binary(embs, projected_dim=64, seed=99)
        assert not torch.equal(b1, b2)


# ---------------------------------------------------------------------------
# Identity Truncation Tests
# ---------------------------------------------------------------------------


class TestIdentityTruncation:
    def test_output_shape(self):
        embs = torch.randn(10, 128)
        binary = identity_truncation(embs, projected_dim=64)
        assert binary.shape == (10, 64)

    def test_values_are_pm1(self):
        embs = torch.randn(20, 128)
        binary = identity_truncation(embs, projected_dim=32)
        assert torch.all((binary == 1.0) | (binary == -1.0))

    def test_truncates_first_r_dims(self):
        """Identity truncation should use the first r dimensions."""
        embs = torch.randn(5, 128)
        binary = identity_truncation(embs, projected_dim=64)
        expected = torch.sign(embs[:, :64])
        expected[expected == 0] = 1.0
        assert torch.equal(binary, expected)


# ---------------------------------------------------------------------------
# PCA Projection Binary Tests
# ---------------------------------------------------------------------------


class TestPCAProjectionBinary:
    def test_output_shape(self):
        embs = torch.randn(100, 128)
        binary, R, mean = pca_projection_binary(embs, projected_dim=64)
        assert binary.shape == (100, 64)
        assert R.shape == (64, 128)
        assert mean.shape == (128,)

    def test_values_are_pm1(self):
        embs = torch.randn(100, 128)
        binary, _, _ = pca_projection_binary(embs, projected_dim=32)
        assert torch.all((binary == 1.0) | (binary == -1.0))


# ---------------------------------------------------------------------------
# PQ Lookup Tables Tests
# ---------------------------------------------------------------------------


class TestPQLookupTables:
    @pytest.fixture
    def trained_pq(self):
        data = torch.randn(500, 128)
        _, pq = product_quantize(data, n_subquantizers=8, n_bits=8)
        return pq

    def test_output_shape(self, trained_pq):
        q = torch.randn(5, 128)
        tables = pq_lookup_tables(q, trained_pq)
        # (m_query_tokens, n_sub_quantizers, k_centroids)
        assert tables.shape == (5, 8, 256)

    def test_values_are_finite(self, trained_pq):
        q = torch.randn(3, 128)
        tables = pq_lookup_tables(q, trained_pq)
        assert torch.all(torch.isfinite(tables))

    def test_single_query_token(self, trained_pq):
        q = torch.randn(1, 128)
        tables = pq_lookup_tables(q, trained_pq)
        assert tables.shape == (1, 8, 256)
