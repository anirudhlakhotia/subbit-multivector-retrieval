"""Unit tests for evaluation metrics (MRR, NDCG, Recall, rank correlation)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.subbit.evaluation import (
    compute_mrr,
    compute_ndcg,
    compute_recall,
    compute_per_query_ndcg,
    compute_rank_correlation,
)


# ---------------------------------------------------------------------------
# Fixtures — shared ranking / qrels data
# ---------------------------------------------------------------------------

@pytest.fixture
def perfect_data():
    """Rankings where the single relevant doc is ranked first."""
    qrels = {
        "q1": {"d1": 1},
        "q2": {"d3": 1},
    }
    rankings = {
        "q1": ["d1", "d2", "d3"],
        "q2": ["d3", "d1", "d2"],
    }
    return rankings, qrels


@pytest.fixture
def imperfect_data():
    """Rankings where relevant doc is NOT rank 1."""
    qrels = {
        "q1": {"d2": 1},          # relevant at rank 2
        "q2": {"d3": 1},          # relevant at rank 3
        "q3": {"d1": 1, "d4": 1}, # two relevant docs, at rank 1 and out of top-3
    }
    rankings = {
        "q1": ["d1", "d2", "d3", "d4", "d5"],
        "q2": ["d1", "d2", "d3", "d4", "d5"],
        "q3": ["d1", "d2", "d3", "d4", "d5"],
    }
    return rankings, qrels


@pytest.fixture
def graded_data():
    """Rankings with graded relevance (rel=0,1,2,3)."""
    qrels = {
        "q1": {"d1": 3, "d2": 2, "d3": 1, "d4": 0},
    }
    # Perfect ordering: d1(3) > d2(2) > d3(1) > d4(0)
    rankings_perfect = {"q1": ["d1", "d2", "d3", "d4"]}
    # Reversed ordering: d4(0) > d3(1) > d2(2) > d1(3)
    rankings_reversed = {"q1": ["d4", "d3", "d2", "d1"]}
    return rankings_perfect, rankings_reversed, qrels


# ---------------------------------------------------------------------------
# MRR Tests
# ---------------------------------------------------------------------------

class TestComputeMRR:
    def test_perfect_ranking_gives_mrr_1(self, perfect_data):
        rankings, qrels = perfect_data
        assert compute_mrr(rankings, qrels, k=10) == 1.0

    def test_relevant_at_rank_2(self):
        rankings = {"q1": ["d_irrel", "d_rel"]}
        qrels = {"q1": {"d_rel": 1}}
        assert compute_mrr(rankings, qrels, k=10) == 0.5

    def test_relevant_beyond_k_gives_zero(self):
        rankings = {"q1": ["d1", "d2", "d3", "d_rel"]}
        qrels = {"q1": {"d_rel": 1}}
        assert compute_mrr(rankings, qrels, k=3) == 0.0

    def test_no_relevant_docs_gives_zero(self):
        rankings = {"q1": ["d1", "d2"]}
        qrels = {"q1": {"d_other": 1}}
        assert compute_mrr(rankings, qrels, k=10) == 0.0

    def test_empty_rankings(self):
        assert compute_mrr({}, {}, k=10) == 0.0

    def test_mixed_queries(self, imperfect_data):
        rankings, qrels = imperfect_data
        # q1: relevant at rank 2 → 1/2, q2: rank 3 → 1/3, q3: rank 1 → 1/1
        expected = (0.5 + 1/3 + 1.0) / 3
        assert abs(compute_mrr(rankings, qrels, k=10) - expected) < 1e-6

    def test_query_not_in_qrels_ignored(self):
        rankings = {"q1": ["d1"], "q_unknown": ["d2"]}
        qrels = {"q1": {"d1": 1}}
        assert compute_mrr(rankings, qrels, k=10) == 1.0


# ---------------------------------------------------------------------------
# Recall Tests
# ---------------------------------------------------------------------------

class TestComputeRecall:
    def test_perfect_recall(self, perfect_data):
        rankings, qrels = perfect_data
        assert compute_recall(rankings, qrels, k=10) == 1.0

    def test_partial_recall(self):
        rankings = {"q1": ["d1", "d2"]}
        qrels = {"q1": {"d1": 1, "d3": 1}}  # d3 not retrieved
        assert compute_recall(rankings, qrels, k=10) == 0.5

    def test_recall_at_k_cutoff(self):
        rankings = {"q1": ["d1", "d2", "d_rel"]}
        qrels = {"q1": {"d_rel": 1}}
        assert compute_recall(rankings, qrels, k=2) == 0.0
        assert compute_recall(rankings, qrels, k=3) == 1.0

    def test_zero_recall_when_nothing_relevant(self):
        rankings = {"q1": ["d1", "d2"]}
        qrels = {"q1": {"d3": 1}}
        assert compute_recall(rankings, qrels, k=10) == 0.0

    def test_multiple_relevant(self, imperfect_data):
        rankings, qrels = imperfect_data
        # q3 has d1 (rank 1) and d4 (rank 4), at k=3: recall = 1/2
        recall_q3 = compute_recall({"q3": rankings["q3"]}, {"q3": qrels["q3"]}, k=3)
        assert abs(recall_q3 - 0.5) < 1e-6


# ---------------------------------------------------------------------------
# NDCG Tests
# ---------------------------------------------------------------------------

class TestComputeNDCG:
    def test_perfect_graded_ranking(self, graded_data):
        rankings_perfect, _, qrels = graded_data
        ndcg = compute_ndcg(rankings_perfect, qrels, k=4)
        assert abs(ndcg - 1.0) < 1e-6

    def test_reversed_ranking_less_than_perfect(self, graded_data):
        rankings_perfect, rankings_reversed, qrels = graded_data
        ndcg_perfect = compute_ndcg(rankings_perfect, qrels, k=4)
        ndcg_reversed = compute_ndcg(rankings_reversed, qrels, k=4)
        assert ndcg_reversed < ndcg_perfect

    def test_binary_relevance_at_rank_1(self, perfect_data):
        rankings, qrels = perfect_data
        # Binary relevance, relevant at rank 1 → DCG = (2^1-1)/log2(2) = 1.0
        # IDCG = same → NDCG = 1.0
        ndcg = compute_ndcg(rankings, qrels, k=10)
        assert abs(ndcg - 1.0) < 1e-6

    def test_empty_gives_zero(self):
        assert compute_ndcg({}, {}, k=10) == 0.0


# ---------------------------------------------------------------------------
# Per-Query NDCG Tests
# ---------------------------------------------------------------------------

class TestComputePerQueryNDCG:
    def test_returns_dict_per_query(self, imperfect_data):
        rankings, qrels = imperfect_data
        per_q = compute_per_query_ndcg(rankings, qrels, k=10)
        assert isinstance(per_q, dict)
        assert set(per_q.keys()) == {"q1", "q2", "q3"}

    def test_mean_matches_global(self, imperfect_data):
        rankings, qrels = imperfect_data
        per_q = compute_per_query_ndcg(rankings, qrels, k=10)
        global_ndcg = compute_ndcg(rankings, qrels, k=10)
        mean_per_q = sum(per_q.values()) / len(per_q)
        assert abs(mean_per_q - global_ndcg) < 1e-6


# ---------------------------------------------------------------------------
# Rank Correlation Tests
# ---------------------------------------------------------------------------

class TestComputeRankCorrelation:
    def test_identical_scores_give_rho_1(self):
        scores = {"q1": {"d1": 0.9, "d2": 0.5, "d3": 0.1}}
        result = compute_rank_correlation(scores, scores)
        assert abs(result["spearman_rho"] - 1.0) < 1e-6

    def test_reversed_scores_give_negative_rho(self):
        scores_a = {"q1": {"d1": 0.9, "d2": 0.5, "d3": 0.1}}
        scores_b = {"q1": {"d1": 0.1, "d2": 0.5, "d3": 0.9}}
        result = compute_rank_correlation(scores_a, scores_b)
        assert result["spearman_rho"] < 0
