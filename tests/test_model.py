"""Unit tests for SubBitModel: forward pass, STE gradients, save/load."""

import pytest
import torch
import torch.nn as nn
import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.subbit.model import SubBitModel, ste_sign, hard_sign, StraightThroughSign, InitMethod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def model_64():
    """SubBitModel with r=64, random orthogonal init."""
    return SubBitModel(
        input_dim=128,
        projected_dim=64,
        init_method="random_orthogonal",
        orthogonal_constraint=False,
    )


@pytest.fixture
def model_32():
    """SubBitModel with r=32, random orthogonal init."""
    return SubBitModel(
        input_dim=128,
        projected_dim=32,
        init_method="random_orthogonal",
        orthogonal_constraint=False,
    )


@pytest.fixture
def model_128():
    """SubBitModel with r=128 (no compression), identity init."""
    return SubBitModel(
        input_dim=128,
        projected_dim=128,
        init_method="identity",
        orthogonal_constraint=False,
    )


# ---------------------------------------------------------------------------
# STE Sign Tests
# ---------------------------------------------------------------------------


class TestSTESign:
    """Tests for the Straight-Through Estimator sign function."""

    def test_forward_produces_signs(self):
        """sign(x) should produce {-1, +1} values."""
        x = torch.randn(100)
        y = ste_sign(x)
        assert torch.all((y == 1.0) | (y == -1.0))

    def test_forward_correct_signs(self):
        """Positive inputs → +1, negative inputs → -1."""
        x = torch.tensor([3.0, -2.0, 0.5, -0.1])
        y = ste_sign(x)
        assert torch.allclose(y, torch.tensor([1.0, -1.0, 1.0, -1.0]))

    def test_zero_mapped_to_positive(self):
        """Zeros should be mapped to +1."""
        x = torch.tensor([0.0, 0.0, 0.0])
        y = ste_sign(x)
        assert torch.all(y == 1.0)

    def test_gradient_is_identity(self):
        """STE gradient should pass through as identity (∂sign/∂x ≈ 1)."""
        x = torch.randn(50, requires_grad=True)
        y = ste_sign(x)
        loss = y.sum()
        loss.backward()
        # Gradient should be all ones (identity pass-through)
        assert torch.allclose(x.grad, torch.ones_like(x))

    def test_gradient_flows_through_chain(self):
        """Gradient should flow through sign in a computation chain."""
        x = torch.randn(10, requires_grad=True)
        # y = sign(2*x) → gradient should be 2 (chain rule with identity STE)
        y = ste_sign(2 * x)
        loss = y.sum()
        loss.backward()
        assert torch.allclose(x.grad, 2 * torch.ones_like(x))

    def test_gradient_with_downstream_ops(self):
        """Gradient should flow correctly when sign is in the middle of a graph."""
        x = torch.randn(5, 10, requires_grad=True)
        W = torch.randn(10, 3)
        # Forward: sign(x) @ W → sum
        binary = ste_sign(x)
        output = binary @ W
        loss = output.sum()
        loss.backward()
        # x.grad should not be None or zero
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_output_detached_from_input_value(self):
        """The forward output (sign) shouldn't affect gradient magnitude."""
        x = torch.tensor([100.0, -100.0, 0.001, -0.001], requires_grad=True)
        y = ste_sign(x)
        loss = y.sum()
        loss.backward()
        # All gradients should be 1 regardless of input magnitude
        assert torch.allclose(x.grad, torch.ones(4))

    def test_hard_sign_blocks_gradients(self):
        """Non-STE sign should have zero gradient."""
        x = torch.randn(32, requires_grad=True)
        y = hard_sign(x)
        y.sum().backward()
        assert torch.allclose(x.grad, torch.zeros_like(x))


# ---------------------------------------------------------------------------
# Model Forward Pass Tests
# ---------------------------------------------------------------------------


class TestModelForward:
    """Tests for SubBitModel forward pass shapes and values."""

    def test_project_shape(self, model_64):
        """project() should reduce last dimension to projected_dim."""
        x = torch.randn(10, 128)
        proj = model_64.project(x)
        assert proj.shape == (10, 64)

    def test_project_batched_shape(self, model_64):
        """project() should handle batched inputs."""
        x = torch.randn(4, 10, 128)
        proj = model_64.project(x)
        assert proj.shape == (4, 10, 64)

    def test_binarize_output_values(self, model_64):
        """binarize() output should be in {-1, +1}."""
        projected = torch.randn(10, 64)
        binary = model_64.binarize(projected)
        assert binary.shape == (10, 64)
        assert torch.all((binary == 1.0) | (binary == -1.0))

    def test_encode_document_shape(self, model_64):
        """encode_document() should return (n, r) binary tensor."""
        doc = torch.randn(50, 128)
        encoded = model_64.encode_document(doc)
        assert encoded.shape == (50, 64)
        assert torch.all((encoded == 1.0) | (encoded == -1.0))

    def test_encode_document_batched(self, model_64):
        """encode_document() should handle batched input."""
        doc = torch.randn(4, 50, 128)
        encoded = model_64.encode_document(doc)
        assert encoded.shape == (4, 50, 64)

    def test_encode_query_asymmetric(self, model_64):
        """encode_query() asymmetric should return float tensor."""
        query = torch.randn(10, 128)
        encoded = model_64.encode_query(query, symmetric=False)
        assert encoded.shape == (10, 64)
        # Should NOT be binary — should have non-integer values
        assert not torch.all((encoded == 1.0) | (encoded == -1.0))

    def test_encode_query_symmetric(self, model_64):
        """encode_query() symmetric should return binary tensor."""
        query = torch.randn(10, 128)
        encoded = model_64.encode_query(query, symmetric=True)
        assert encoded.shape == (10, 64)
        assert torch.all((encoded == 1.0) | (encoded == -1.0))

    def test_different_projected_dims(self, model_32, model_64, model_128):
        """Models with different r should produce different output shapes."""
        x = torch.randn(10, 128)
        assert model_32.encode_document(x).shape == (10, 32)
        assert model_64.encode_document(x).shape == (10, 64)
        assert model_128.encode_document(x).shape == (10, 128)


# ---------------------------------------------------------------------------
# Gradient Flow Tests
# ---------------------------------------------------------------------------


class TestGradientFlow:
    """Tests that gradients flow correctly through the model."""

    def test_gradient_reaches_R(self, model_64):
        """Gradient should reach R through the STE binarization."""
        doc = torch.randn(10, 128)
        encoded = model_64.encode_document(doc)
        loss = encoded.sum()
        loss.backward()

        assert model_64.R.weight.grad is not None
        assert model_64.R.weight.grad.abs().sum() > 0

    def test_gradient_shape(self, model_64):
        """R.weight.grad should have shape (projected_dim, input_dim)."""
        doc = torch.randn(10, 128)
        loss = model_64.encode_document(doc).sum()
        loss.backward()
        assert model_64.R.weight.grad.shape == (64, 128)

    def test_encode_document_without_ste_blocks_R_grad(self, model_64):
        """Hard-sign doc ablation should not backpropagate through sign."""
        doc = torch.randn(10, 128)
        loss = model_64.encode_document(doc, use_ste=False).sum()
        loss.backward()
        assert model_64.R.weight.grad is not None
        assert torch.allclose(model_64.R.weight.grad, torch.zeros_like(model_64.R.weight.grad))

    def test_gradient_with_loss(self, model_64):
        """Full forward-backward with a realistic loss should work."""
        q = torch.randn(5, 128)
        d = torch.randn(20, 128)

        q_proj = model_64.encode_query(q)
        d_bin = model_64.encode_document(d)

        # MaxSim-like loss
        sim = q_proj @ d_bin.T  # (5, 20)
        max_sim = sim.max(dim=1).values  # (5,)
        loss = -max_sim.sum()
        loss.backward()

        assert model_64.R.weight.grad is not None
        assert model_64.R.weight.grad.abs().sum() > 0

    def test_optimizer_step_changes_R(self, model_64):
        """An optimizer step should modify R's weights."""
        R_before = model_64.R.weight.data.clone()

        optimizer = torch.optim.Adam(model_64.parameters(), lr=0.01)
        doc = torch.randn(10, 128)
        loss = model_64.encode_document(doc).sum()
        loss.backward()
        optimizer.step()

        R_after = model_64.R.weight.data
        assert not torch.allclose(R_before, R_after), "R should change after optimizer step"


# ---------------------------------------------------------------------------
# Initialization Tests
# ---------------------------------------------------------------------------


class TestInitialization:
    """Tests for different initialization methods."""

    def test_random_orthogonal_init(self):
        """Random orthogonal init should produce near-orthonormal rows."""
        model = SubBitModel(128, 64, init_method="random_orthogonal")
        R = model.R.weight.data  # (64, 128)
        # R @ R^T should be close to identity
        product = R @ R.T
        assert torch.allclose(product, torch.eye(64), atol=1e-5)

    def test_identity_init(self):
        """Identity init should use first r rows of identity."""
        model = SubBitModel(128, 64, init_method="identity")
        R = model.R.weight.data
        expected = torch.eye(128)[:64]
        assert torch.allclose(R, expected)

    def test_pca_init_deferred(self):
        """PCA init should be deferred until fit_pca() is called."""
        model = SubBitModel(128, 64, init_method="pca")
        assert model._pca_fitted is False

    def test_pca_fit(self):
        """fit_pca() should update R with PCA components."""
        model = SubBitModel(128, 32, init_method="pca")
        # Generate data with known structure
        data = torch.randn(1000, 128)
        R_before = model.R.weight.data.clone()

        model.fit_pca(data)

        assert model._pca_fitted is True
        R_after = model.R.weight.data
        assert R_after.shape == (32, 128)
        # R should have changed
        assert not torch.allclose(R_before, R_after)

    def test_pca_init_with_numpy(self):
        """fit_pca() should accept numpy arrays."""
        model = SubBitModel(128, 32, init_method="pca")
        data = np.random.randn(1000, 128).astype(np.float32)
        model.fit_pca(data)
        assert model._pca_fitted is True

    def test_pca_wrong_dim_raises(self):
        """fit_pca() should raise if embedding dim doesn't match."""
        model = SubBitModel(128, 32, init_method="pca")
        data = torch.randn(100, 64)  # Wrong dim
        with pytest.raises(ValueError, match="Embedding dim"):
            model.fit_pca(data)


# ---------------------------------------------------------------------------
# Stiefel Constraint Tests
# ---------------------------------------------------------------------------


class TestStiefelConstraint:
    """Tests for the orthogonal (Stiefel) constraint on R."""

    def test_apply_orthogonal_constraint(self):
        """After constraint, R should have orthonormal rows."""
        model = SubBitModel(128, 64, init_method="random_orthogonal",
                                    orthogonal_constraint=True)
        # Perturb R to break orthogonality
        model.R.weight.data += 0.1 * torch.randn_like(model.R.weight.data)

        # Verify it's no longer orthogonal
        R = model.R.weight.data
        product = R @ R.T
        assert not torch.allclose(product, torch.eye(64), atol=1e-3)

        # Apply constraint
        model.apply_orthogonal_constraint()

        R = model.R.weight.data
        product = R @ R.T
        assert torch.allclose(product, torch.eye(64), atol=1e-5)

    def test_stiefel_preserves_shape(self):
        """Stiefel projection shouldn't change R's shape."""
        model = SubBitModel(128, 32, init_method="random_orthogonal",
                                    orthogonal_constraint=True)
        shape_before = model.R.weight.data.shape
        model.apply_orthogonal_constraint()
        assert model.R.weight.data.shape == shape_before


# ---------------------------------------------------------------------------
# Properties Tests
# ---------------------------------------------------------------------------


class TestProperties:
    """Tests for computed properties."""

    def test_bits_per_dim(self):
        model = SubBitModel(128, 64)
        assert model.bits_per_dim == 0.5

    def test_bits_per_dim_32(self):
        model = SubBitModel(128, 32)
        assert model.bits_per_dim == 0.25

    def test_bytes_per_token(self):
        model = SubBitModel(128, 64)
        assert model.bytes_per_token == 8.0

    def test_bytes_per_token_16(self):
        model = SubBitModel(128, 16)
        assert model.bytes_per_token == 2.0

    def test_parameter_count(self):
        """Default (use_scale=True) adds W_scale = 128+1 params on top of R."""
        model = SubBitModel(128, 64)
        total = sum(p.numel() for p in model.parameters())
        assert total == 64 * 128 + 128 + 1  # R + W_scale.weight + W_scale.bias

    def test_parameter_count_no_scale(self):
        """Without use_scale, parameters are exactly R: projected_dim × input_dim."""
        model = SubBitModel(128, 64, use_scale=False)
        total = sum(p.numel() for p in model.parameters())
        assert total == 64 * 128


# ---------------------------------------------------------------------------
# Save / Load Tests
# ---------------------------------------------------------------------------


class TestSaveLoad:
    """Tests for model serialization."""

    def test_save_and_load(self, tmp_path, model_64):
        """Saved model should load with identical weights and config."""
        save_path = tmp_path / "model.pt"
        model_64.save(save_path)

        loaded = SubBitModel.load(save_path)
        assert loaded.input_dim == model_64.input_dim
        assert loaded.projected_dim == model_64.projected_dim
        assert loaded.orthogonal_constraint == model_64.orthogonal_constraint
        assert torch.allclose(loaded.R.weight.data, model_64.R.weight.data)

    def test_load_produces_same_output(self, tmp_path, model_64):
        """Loaded model should produce identical outputs."""
        save_path = tmp_path / "model.pt"
        model_64.save(save_path)
        loaded = SubBitModel.load(save_path)

        x = torch.randn(10, 128)
        model_64.eval()
        loaded.eval()
        with torch.no_grad():
            out1 = model_64.encode_document(x)
            out2 = loaded.encode_document(x)
        assert torch.allclose(out1, out2)

    def test_save_creates_directory(self, tmp_path, model_64):
        """save() should create parent directories if needed."""
        save_path = tmp_path / "nested" / "dir" / "model.pt"
        model_64.save(save_path)
        assert save_path.exists()


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case tests."""

    def test_single_token(self, model_64):
        """Should work with a single token."""
        x = torch.randn(1, 128)
        encoded = model_64.encode_document(x)
        assert encoded.shape == (1, 64)

    def test_r_equals_d(self):
        """r=128 (no compression) should still work correctly."""
        model = SubBitModel(128, 128, init_method="identity")
        x = torch.randn(10, 128)
        encoded = model.encode_document(x)
        assert encoded.shape == (10, 128)

    def test_very_small_r(self):
        """r=8 should work."""
        model = SubBitModel(128, 8, init_method="random_orthogonal")
        x = torch.randn(10, 128)
        encoded = model.encode_document(x)
        assert encoded.shape == (10, 8)
        assert model.bits_per_dim == pytest.approx(0.0625)

    def test_deterministic_output(self, model_64):
        """Same input should always produce same output (no randomness)."""
        model_64.eval()
        x = torch.randn(10, 128)
        with torch.no_grad():
            out1 = model_64.encode_document(x)
            out2 = model_64.encode_document(x)
        assert torch.allclose(out1, out2)


# ---------------------------------------------------------------------------
# Save / Load Roundtrip
# ---------------------------------------------------------------------------


class TestSaveLoadRoundtrip:
    def test_save_load_preserves_weights(self, tmp_path):
        model = SubBitModel(128, 64, init_method="random_orthogonal")
        # Perturb weights so they're not default
        with torch.no_grad():
            model.R.weight.add_(torch.randn_like(model.R.weight) * 0.01)

        path = tmp_path / "model.pt"
        model.save(path)

        loaded = SubBitModel.load(path)
        assert torch.allclose(model.R.weight, loaded.R.weight)

    def test_save_load_preserves_config(self, tmp_path):
        model = SubBitModel(128, 32, init_method="identity", orthogonal_constraint=True)
        path = tmp_path / "model.pt"
        model.save(path)

        loaded = SubBitModel.load(path)
        assert loaded.input_dim == 128
        assert loaded.projected_dim == 32
        assert loaded.orthogonal_constraint is True

    def test_save_load_preserves_encoding(self, tmp_path):
        """Loaded model produces identical encodings."""
        model = SubBitModel(128, 64, init_method="random_orthogonal")
        model.eval()
        x = torch.randn(10, 128)

        path = tmp_path / "model.pt"
        model.save(path)
        loaded = SubBitModel.load(path)
        loaded.eval()

        with torch.no_grad():
            orig_q = model.encode_query(x)
            loaded_q = loaded.encode_query(x)
            orig_d = model.encode_document(x)
            loaded_d = loaded.encode_document(x)

        assert torch.allclose(orig_q, loaded_q)
        assert torch.allclose(orig_d, loaded_d)

    def test_from_file_slices_larger_checkpoint_for_low_r_warm_start(self, tmp_path):
        source = SubBitModel(128, 64, init_method="random_orthogonal")
        path = tmp_path / "source.pt"
        source.save(path)

        target = SubBitModel(
            128,
            8,
            init_method="from_file",
            init_path=str(path),
        )

        assert target.R.weight.shape == (8, 128)
        assert torch.allclose(target.R.weight, source.R.weight[:8])

    def test_save_load_from_file_checkpoint_is_self_contained(self, tmp_path):
        source = SubBitModel(128, 64, init_method="random_orthogonal")
        source_path = tmp_path / "source.pt"
        source.save(source_path)
        model = SubBitModel(
            128,
            8,
            init_method="from_file",
            init_path=str(source_path),
        )
        checkpoint = tmp_path / "from_file.pt"
        model.save(checkpoint)

        loaded = SubBitModel.load(checkpoint)

        assert loaded.init_method == model.init_method
        assert torch.allclose(loaded.R.weight, model.R.weight)
