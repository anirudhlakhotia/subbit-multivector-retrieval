"""Unit tests for the current SubBit loss.

The active loss is `boundary_guard_loss` — Top-K Basin Shaping (MSE) plus an
Adversarial Sentinel (MSE anchoring the strongest student outsider to its
teacher score). All prior heuristic losses (InfoNCE, distillation KL,
token-winner CE, triplet ranking, etc.) were physically removed; tests for
them have been dropped.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.subbit.losses import (  # noqa: E402
    SubBitLoss,
    boundary_guard_loss,
    orthogonal_regularization,
)
from src.subbit.model import SubBitModel  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def model():
    """Small model for testing."""
    return SubBitModel(
        input_dim=128,
        projected_dim=64,
        init_method="random_orthogonal",
    )


@pytest.fixture
def batch_data():
    """Batch of (query, pos_doc, neg_doc) ColBERT-like embeddings."""
    torch.manual_seed(42)
    batch_size = 4
    q = torch.randn(batch_size, 8, 128)
    d_pos = torch.randn(batch_size, 20, 128)
    d_neg = torch.randn(batch_size, 20, 128)

    q = nn.functional.normalize(q, dim=-1)
    d_pos = nn.functional.normalize(d_pos, dim=-1)
    d_neg = nn.functional.normalize(d_neg, dim=-1)

    return q, d_pos, d_neg


# ---------------------------------------------------------------------------
# boundary_guard_loss — the heart of the training objective
# ---------------------------------------------------------------------------


class TestBoundaryGuardLoss:
    def test_returns_two_nonnegative_tensors(self, model, batch_data):
        q, d_pos, _ = batch_data
        q_proj = model.encode_query(q)
        d_bin = model.encode_document(d_pos)

        topk_loss, guard_loss = boundary_guard_loss(q_proj, d_bin, q, d_pos, k=3)

        assert torch.isfinite(topk_loss)
        assert torch.isfinite(guard_loss)
        assert topk_loss.item() >= 0.0
        # Hinge is relu-based so guard is always >= 0 by construction.
        assert guard_loss.item() >= 0.0

    def test_differentiable(self, model, batch_data):
        q, d_pos, _ = batch_data
        q_proj = model.encode_query(q)
        d_bin = model.encode_document(d_pos)

        topk_loss, guard_loss = boundary_guard_loss(q_proj, d_bin, q, d_pos, k=3)
        (topk_loss + guard_loss).backward()

        assert model.R.weight.grad is not None
        assert model.R.weight.grad.abs().sum() > 0

    def test_skips_when_too_few_docs(self, model):
        """If a batch item has fewer docs than k+1, the loop should skip it cleanly."""
        q = torch.randn(1, 4, 128)
        d = torch.randn(1, 2, 128)  # only 2 docs — k=3 needs >= 4
        d_mask = torch.ones(1, 2, dtype=torch.bool)

        q_proj = model.encode_query(q)
        d_bin = model.encode_document(d)

        topk_loss, guard_loss = boundary_guard_loss(
            q_proj, d_bin, q, d, k=3, d_mask=d_mask,
        )
        # With no valid items, both terms should be zero (skipped).
        assert topk_loss.item() == 0.0
        assert guard_loss.item() == 0.0

    def test_with_masks(self, model, batch_data):
        q, d_pos, _ = batch_data
        batch_size = q.shape[0]
        q_proj = model.encode_query(q)
        d_bin = model.encode_document(d_pos)

        q_mask = torch.ones(batch_size, 8, dtype=torch.bool)
        q_mask[:, 6:] = False
        d_mask = torch.ones(batch_size, 20, dtype=torch.bool)
        d_mask[:, 15:] = False

        topk_loss, guard_loss = boundary_guard_loss(
            q_proj, d_bin, q, d_pos, k=3, q_mask=q_mask, d_mask=d_mask,
        )
        assert torch.isfinite(topk_loss)
        assert torch.isfinite(guard_loss)


# ---------------------------------------------------------------------------
# Orthogonal regularization
# ---------------------------------------------------------------------------


class TestOrthogonalRegularization:
    def test_low_for_orthogonal_init(self):
        m = SubBitModel(128, 64, init_method="random_orthogonal")
        loss = orthogonal_regularization(m)
        assert loss.item() < 1.0

    def test_differentiable(self):
        m = SubBitModel(128, 64, init_method="random_orthogonal")
        loss = orthogonal_regularization(m)
        loss.backward()
        assert m.R.weight.grad is not None


# ---------------------------------------------------------------------------
# SubBitLoss (combined forward)
# ---------------------------------------------------------------------------


class TestSubBitLoss:
    def test_forward_returns_expected_keys(self, model, batch_data):
        q, d_pos, d_neg = batch_data
        criterion = SubBitLoss()
        result = criterion(model, q, d_pos, d_neg)

        assert set(result.keys()) == {
            "total",
            "boundary_topk",
            "boundary_fp",
            "ortho",
        }

    def test_total_is_weighted_sum(self, model, batch_data):
        q, d_pos, d_neg = batch_data
        criterion = SubBitLoss(
            boundary_topk_weight=1.0,
            boundary_fp_weight=1.0,
            ortho_weight=0.001,
        )
        result = criterion(model, q, d_pos, d_neg)

        expected = (
            1.0 * result["boundary_topk"].item()
            + 1.0 * result["boundary_fp"].item()
            + 0.001 * result["ortho"].item()
        )
        assert result["total"].item() == pytest.approx(expected, abs=1e-4)

    def test_weights_can_zero_terms(self, model, batch_data):
        q, d_pos, d_neg = batch_data
        criterion = SubBitLoss(
            boundary_topk_weight=0.0,
            boundary_fp_weight=0.0,
            ortho_weight=0.0,
        )
        result = criterion(model, q, d_pos, d_neg)
        assert result["total"].item() == pytest.approx(0.0)

    def test_loss_decreases_with_training(self, batch_data):
        q, d_pos, d_neg = batch_data
        m = SubBitModel(128, 64, init_method="random_orthogonal")
        criterion = SubBitLoss()
        optimizer = torch.optim.Adam(m.parameters(), lr=0.01)

        losses = []
        for _ in range(20):
            optimizer.zero_grad()
            result = criterion(m, q, d_pos, d_neg)
            result["total"].backward()
            optimizer.step()
            losses.append(result["total"].item())

        avg_first = sum(losses[:5]) / 5
        avg_last = sum(losses[-5:]) / 5
        assert avg_last <= avg_first, (
            f"Loss did not decrease: first5={avg_first:.4f} last5={avg_last:.4f}"
        )

    def test_gradient_magnitude_reasonable(self, model, batch_data):
        q, d_pos, d_neg = batch_data
        criterion = SubBitLoss()
        result = criterion(model, q, d_pos, d_neg)
        result["total"].backward()

        grad_norm = model.R.weight.grad.norm().item()
        assert 1e-8 < grad_norm < 1e6, f"gradient norm {grad_norm} unreasonable"

    def test_with_masks(self, model, batch_data):
        q, d_pos, d_neg = batch_data
        batch_size = q.shape[0]

        q_mask = torch.ones(batch_size, 8, dtype=torch.bool)
        q_mask[:, 6:] = False
        d_pos_mask = torch.ones(batch_size, 20, dtype=torch.bool)
        d_pos_mask[:, 15:] = False
        d_neg_mask = torch.ones(batch_size, 20, dtype=torch.bool)

        criterion = SubBitLoss()
        result = criterion(
            model, q, d_pos, d_neg,
            q_mask=q_mask, d_pos_mask=d_pos_mask, d_neg_mask=d_neg_mask,
        )
        assert torch.isfinite(result["total"])
        result["total"].backward()
        assert model.R.weight.grad is not None

    def test_ste_flags(self, model):
        """STE defaults follow the paper config; both are overridable."""
        default = SubBitLoss()
        assert default.ste_query is True
        assert default.ste_doc is True

        custom = SubBitLoss(ste_query=True, ste_doc=False)
        assert custom.ste_query is True
        assert custom.ste_doc is False


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestLossEdgeCases:
    def test_single_sample_batch(self, model):
        q = torch.randn(1, 5, 128)
        d_pos = torch.randn(1, 10, 128)
        d_neg = torch.randn(1, 10, 128)

        criterion = SubBitLoss()
        result = criterion(model, q, d_pos, d_neg)
        assert torch.isfinite(result["total"])

    def test_single_token_query(self, model):
        q = torch.randn(2, 1, 128)
        d_pos = torch.randn(2, 10, 128)
        d_neg = torch.randn(2, 10, 128)

        criterion = SubBitLoss()
        result = criterion(model, q, d_pos, d_neg)
        assert torch.isfinite(result["total"])
