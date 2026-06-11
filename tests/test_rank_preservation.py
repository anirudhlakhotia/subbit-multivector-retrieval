"""Tests for rank_preservation: structure-preservation diagnostics."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.subbit.rank_preservation import (
    PairwiseScores,
    compute_pairwise_scores,
    compute_rank_preservation,
    topk_overlap,
)


# ---------------------------------------------------------------------------
# topk_overlap unit
# ---------------------------------------------------------------------------


def test_topk_overlap_identity():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert topk_overlap(a, a, k=3) == pytest.approx(1.0)
    assert topk_overlap(a, a, k=1) == pytest.approx(1.0)


def test_topk_overlap_fully_reversed():
    # Top-3 of [1,2,3,4,5] is {2,3,4}; top-3 of its reverse [5,4,3,2,1]
    # is {0,1,2}. Intersection is {2} → 1/3.
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = a[::-1].copy()
    assert topk_overlap(a, b, k=3) == pytest.approx(1 / 3)


def test_topk_overlap_k_exceeds_n():
    a = np.array([0.1, 0.2, 0.3])
    b = np.array([0.3, 0.2, 0.1])
    assert topk_overlap(a, b, k=100) == pytest.approx(1.0)


def test_topk_overlap_half_shared():
    a = np.array([10.0, 9.0, 1.0, 2.0])   # top-2 → {0, 1}
    b = np.array([10.0, 1.0, 9.0, 2.0])   # top-2 → {0, 2}
    assert topk_overlap(a, b, k=2) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# compute_rank_preservation unit
# ---------------------------------------------------------------------------


def _make_scores(fp32: np.ndarray, subbit: np.ndarray) -> PairwiseScores:
    q_ids = [f"q{i}" for i in range(fp32.shape[0])]
    d_ids = [f"d{j}" for j in range(fp32.shape[1])]
    return PairwiseScores(
        fp32=torch.from_numpy(fp32).float(),
        subbit=torch.from_numpy(subbit).float(),
        query_ids=q_ids,
        doc_ids=d_ids,
    )


def test_identity_scores_give_perfect_agreement():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(5, 50))
    scores = _make_scores(a, a.copy())
    rep = compute_rank_preservation(scores, k_values=(1, 5, 50))
    assert rep.spearman_rho["mean"] == pytest.approx(1.0)
    assert rep.kendall_tau["mean"] == pytest.approx(1.0)
    assert rep.topk_overlap[1]["mean"] == pytest.approx(1.0)
    assert rep.topk_overlap[5]["mean"] == pytest.approx(1.0)
    assert rep.topk_overlap[50]["mean"] == pytest.approx(1.0)
    assert rep.score_correlation_pearson == pytest.approx(1.0)


def test_reversed_scores_give_perfect_anticorrelation():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(4, 40))
    b = -a
    scores = _make_scores(a, b)
    rep = compute_rank_preservation(scores, k_values=(10,))
    assert rep.spearman_rho["mean"] == pytest.approx(-1.0)
    assert rep.kendall_tau["mean"] == pytest.approx(-1.0)
    assert rep.score_correlation_pearson == pytest.approx(-1.0)


def test_monotone_transform_preserves_rank_correlation():
    # Squaring a strictly positive monotone signal preserves order.
    rng = np.random.default_rng(1)
    a = np.abs(rng.normal(size=(3, 30))) + 1.0
    b = a**3  # strictly monotone on positives
    scores = _make_scores(a, b)
    rep = compute_rank_preservation(scores, k_values=(5,))
    assert rep.spearman_rho["mean"] == pytest.approx(1.0)
    assert rep.kendall_tau["mean"] == pytest.approx(1.0)
    # Pearson on a non-linear transform is NOT 1.0 — expected.
    assert rep.score_correlation_pearson < 1.0
    assert rep.score_correlation_pearson > 0.5


def test_skips_constant_score_queries():
    a = np.array([[1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0]])
    b = np.array([[0.5, 0.5, 0.5, 0.5], [1.0, 2.0, 3.0, 4.0]])
    scores = _make_scores(a, b)
    rep = compute_rank_preservation(scores, k_values=(2,))
    # Only the second query contributed.
    assert rep.spearman_rho["n"] == 1
    assert rep.spearman_rho["mean"] == pytest.approx(1.0)


def test_per_query_dump_round_trips():
    rng = np.random.default_rng(42)
    a = rng.normal(size=(4, 20))
    b = a + 0.1 * rng.normal(size=(4, 20))
    scores = _make_scores(a, b)
    rep = compute_rank_preservation(
        scores, k_values=(3, 10), keep_per_query=True
    )
    assert len(rep.per_query) == 4
    for entry in rep.per_query.values():
        assert "spearman_rho" in entry
        assert "kendall_tau" in entry
        assert set(entry["topk_overlap"].keys()) == {"3", "10"}


def test_shape_mismatch_raises():
    a = np.zeros((3, 10))
    b = np.zeros((3, 11))
    with pytest.raises(ValueError):
        compute_rank_preservation(_make_scores(a, b))


# ---------------------------------------------------------------------------
# compute_pairwise_scores integration (synthetic stores)
# ---------------------------------------------------------------------------


class _DictStore:
    def __init__(self, data: dict[str, torch.Tensor]):
        self.data = data

    def get(self, key: str) -> torch.Tensor:
        if key not in self.data:
            raise KeyError(key)
        return self.data[key]


def test_compute_pairwise_scores_matches_manual_maxsim():
    torch.manual_seed(0)
    d_in, d_out = 16, 8
    q_tokens, d_tokens = 4, 6
    n_q, n_d = 3, 5

    queries = {
        f"q{i}": torch.randn(q_tokens, d_in)
        for i in range(n_q)
    }
    docs = {
        f"d{j}": torch.randn(d_tokens, d_in)
        for j in range(n_d)
    }

    # Deterministic "learned" projection; sign-binarise docs.
    R = torch.randn(d_out, d_in)

    def encode_query(x):
        return x @ R.T

    def encode_document(x):
        proj = x @ R.T
        proj[proj == 0] = 1.0
        return torch.sign(proj)

    q_store = _DictStore(queries)
    d_store = _DictStore(docs)
    query_ids = list(queries.keys())
    doc_ids = list(docs.keys())

    scores = compute_pairwise_scores(
        q_store, d_store, query_ids, doc_ids,
        encode_query=encode_query,
        encode_document=encode_document,
        device="cpu",
        progress=False,
    )

    assert scores.fp32.shape == (n_q, n_d)
    assert scores.subbit.shape == (n_q, n_d)

    # Verify a single pair by hand.
    q = queries["q1"]
    d = docs["d2"]
    expected_fp32 = (q @ d.T).max(dim=-1).values.sum().item()
    expected_sub = (encode_query(q) @ encode_document(d).T).max(dim=-1).values.sum().item()
    qi = query_ids.index("q1")
    dj = doc_ids.index("d2")
    assert scores.fp32[qi, dj].item() == pytest.approx(expected_fp32, rel=1e-5)
    assert scores.subbit[qi, dj].item() == pytest.approx(expected_sub, rel=1e-5)


def test_missing_docs_are_skipped():
    torch.manual_seed(0)
    q_store = _DictStore({"q0": torch.randn(3, 8)})
    d_store = _DictStore({"d0": torch.randn(4, 8), "d1": torch.randn(4, 8)})
    R = torch.randn(4, 8)

    def encode_query(x):
        return x @ R.T

    def encode_document(x):
        proj = x @ R.T
        proj[proj == 0] = 1.0
        return torch.sign(proj)

    # Ask for a missing doc id; it should be skipped gracefully.
    scores = compute_pairwise_scores(
        q_store, d_store,
        query_ids=["q0"],
        doc_ids=["d0", "d_missing", "d1"],
        encode_query=encode_query,
        encode_document=encode_document,
        device="cpu",
        progress=False,
    )
    assert scores.doc_ids == ["d0", "d1"]
    assert scores.fp32.shape == (1, 2)
