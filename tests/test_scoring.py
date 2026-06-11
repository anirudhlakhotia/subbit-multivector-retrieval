"""MaxSim scoring tests."""

import pytest
import torch
import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.subbit.scoring import (
    maxsim,
    batched_maxsim,
    efficient_batched_maxsim,
    compute_storage_bytes,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_qd():
    """Simple query/doc pair with known MaxSim."""
    # query: 2 tokens, doc: 3 tokens, dim=4
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0, 0.0]])
    d = torch.tensor([[1.0, 0.0, 0.0, 0.0],   # perfect match for q[0]
                       [0.0, 0.0, 1.0, 0.0],   # no match
                       [0.0, 0.5, 0.5, 0.0]])   # partial match for q[1]
    return q, d


@pytest.fixture
def binary_vecs():
    """Random binary vectors in {-1, +1}."""
    torch.manual_seed(42)
    x = torch.randn(10, 64)
    return torch.sign(x).clamp(min=0).float() * 2 - 1  # {-1, +1}


# ---------------------------------------------------------------------------
# MaxSim Tests
# ---------------------------------------------------------------------------


class TestMaxSim:
    """Tests for the core MaxSim scoring function."""

    def test_known_score(self, simple_qd):
        """MaxSim should be 1.0 + 0.5 = 1.5 for the simple example."""
        q, d = simple_qd
        score = maxsim(q, d)
        # q[0] max match with d[0]: 1.0
        # q[1] max match with d[2]: 0.5
        assert torch.isclose(score, torch.tensor(1.5))

    def test_identical_vectors(self):
        """MaxSim of identical q and d should be num_query_tokens (unit vecs)."""
        q = torch.nn.functional.normalize(torch.randn(5, 16), dim=-1)
        score = maxsim(q, q)
        assert torch.isclose(score, torch.tensor(5.0), atol=1e-5)

    def test_orthogonal_vectors(self):
        """MaxSim of orthogonal q and d should be 0."""
        q = torch.eye(4)[:2]  # [1,0,0,0] and [0,1,0,0]
        d = torch.eye(4)[2:]  # [0,0,1,0] and [0,0,0,1]
        score = maxsim(q, d)
        assert torch.isclose(score, torch.tensor(0.0))

    def test_output_is_scalar_unbatched(self):
        """Unbatched MaxSim should return a scalar."""
        q = torch.randn(5, 32)
        d = torch.randn(10, 32)
        score = maxsim(q, d)
        assert score.dim() == 0

    def test_batched_shape(self):
        """Batched MaxSim should return (batch,) tensor."""
        q = torch.randn(4, 5, 32)
        d = torch.randn(4, 10, 32)
        scores = maxsim(q, d)
        assert scores.shape == (4,)

    def test_query_mask(self):
        """Masked query tokens should not contribute to score."""
        q = torch.ones(3, 4)
        d = torch.ones(2, 4)
        mask = torch.tensor([True, True, False])  # Only first 2 tokens

        score_full = maxsim(q, d)
        score_masked = maxsim(q, d, query_mask=mask)

        # Full: 3 tokens × max_sim=4.0 = 12.0
        # Masked: 2 tokens × max_sim=4.0 = 8.0
        assert torch.isclose(score_full, torch.tensor(12.0))
        assert torch.isclose(score_masked, torch.tensor(8.0))

    def test_doc_mask(self):
        """Masked doc tokens should be ignored in max pooling."""
        q = torch.tensor([[1.0, 0.0]])
        d = torch.tensor([[0.5, 0.0],   # sim = 0.5
                           [1.0, 0.0]])  # sim = 1.0 but masked
        d_mask = torch.tensor([True, False])

        score = maxsim(q, d, doc_mask=d_mask)
        assert torch.isclose(score, torch.tensor(0.5))

    def test_symmetric_binary_maxsim(self):
        """MaxSim on binary {-1,+1} vecs should equal dim - 2*hamming."""
        dim = 64
        q = torch.ones(1, dim)
        d = torch.ones(1, dim)
        d[0, :16] = -1  # 16 disagreements

        score = maxsim(q, d)
        # dot product of binary vecs: dim - 2*hamming = 64 - 32 = 32
        assert torch.isclose(score, torch.tensor(32.0))

    def test_asymmetric_maxsim(self):
        """Asymmetric scoring: float query × binary doc."""
        q = torch.randn(5, 32)  # float query
        d = torch.sign(torch.randn(10, 32))  # binary doc
        score = maxsim(q, d)
        assert score.dim() == 0  # Scalar
        assert torch.isfinite(score)


# ---------------------------------------------------------------------------
# Batched MaxSim Tests
# ---------------------------------------------------------------------------


class TestBatchedMaxSim:
    """Tests for batched MaxSim (cross-scoring all queries against all docs)."""

    def test_output_shape(self):
        """Should return (num_queries, num_docs) matrix."""
        q = torch.randn(3, 5, 16)
        d = torch.randn(7, 10, 16)
        scores = batched_maxsim(q, d)
        assert scores.shape == (3, 7)

    def test_consistent_with_unbatched(self):
        """Batched scoring should match individual scoring."""
        q = torch.randn(2, 4, 16)
        d = torch.randn(3, 6, 16)

        batched_scores = batched_maxsim(q, d)

        for i in range(2):
            for j in range(3):
                individual = maxsim(q[i], d[j])
                assert torch.isclose(batched_scores[i, j], individual, atol=1e-5)

    def test_efficient_matches_naive(self):
        """Efficient batched should match naive batched."""
        q = torch.randn(3, 5, 16)
        d = torch.randn(8, 10, 16)

        naive = batched_maxsim(q, d)
        efficient = efficient_batched_maxsim(q, d, chunk_size=4)

        assert torch.allclose(naive, efficient, atol=1e-4)


# ---------------------------------------------------------------------------
# Storage Computation Tests
# ---------------------------------------------------------------------------


class TestStorageComputation:
    """Tests for storage size calculations."""

    def test_basic_computation(self):
        """Check storage computation for known values."""
        result = compute_storage_bytes(
            num_tokens=1_000_000,
            projected_dim=64,
            include_doc_lens=False,
        )
        # 1M tokens × 64/8 bytes = 8MB
        assert result["token_storage_bytes"] == 1_000_000 * 8
        assert result["bytes_per_token"] == 8
        assert result["bits_per_dim"] == pytest.approx(0.5)

    def test_with_doc_overhead(self):
        """Storage with doc length overhead should be larger."""
        without = compute_storage_bytes(1_000_000, 64, include_doc_lens=False)
        with_overhead = compute_storage_bytes(1_000_000, 64, include_doc_lens=True)
        assert with_overhead["total_bytes"] > without["token_storage_bytes"]


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


class TestScoringIntegration:
    """Integration tests combining model + scoring."""

    def test_model_output_scores_correctly(self):
        """Model encode + MaxSim should produce valid scores."""
        from src.subbit.model import SubBitModel

        model = SubBitModel(128, 64, init_method="random_orthogonal")
        model.eval()

        q = torch.randn(5, 128)
        d = torch.randn(20, 128)

        with torch.no_grad():
            q_proj = model.encode_query(q)
            d_bin = model.encode_document(d)
            score = maxsim(q_proj, d_bin)

        assert torch.isfinite(score)
        assert score.dim() == 0

    def test_asymmetric_vs_symmetric(self):
        """Asymmetric and symmetric scoring should differ (asymmetric uses float query)."""
        from src.subbit.model import SubBitModel

        model = SubBitModel(128, 64, init_method="random_orthogonal")
        model.eval()

        q = torch.randn(5, 128)
        d = torch.randn(20, 128)

        with torch.no_grad():
            q_asym = model.encode_query(q, symmetric=False)
            q_sym = model.encode_query(q, symmetric=True)
            d_bin = model.encode_document(d)

            score_asym = maxsim(q_asym, d_bin)
            score_sym = maxsim(q_sym, d_bin)

        # They should generally differ
        # (could be equal in degenerate cases, but very unlikely with random data)
        assert not torch.isclose(score_asym, score_sym, atol=1e-3)

    def test_r128_identity_approximates_full_precision(self):
        """At r=128 with identity init and use_scale=False, q_proj == q exactly.

        (The default use_scale=True multiplies q_proj by a learned token-dependent
        factor S(q) ∈ [1.0, 1.5], so identity alone no longer yields q_proj == q.)
        """
        from src.subbit.model import SubBitModel

        model = SubBitModel(128, 128, init_method="identity", use_scale=False)
        model.eval()

        q = torch.randn(3, 128)
        d = torch.randn(10, 128)

        with torch.no_grad():
            q_proj = model.encode_query(q)  # Identity projection = same as q
            # q_proj should equal q
            assert torch.allclose(q_proj, q, atol=1e-5)
