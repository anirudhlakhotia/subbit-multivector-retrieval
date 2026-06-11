"""Most MaxSim Winners Flip, Retrieval Survives: Low-Margin Substitution in Sign-Coded Late Interaction."""

__version__ = "0.1.0"

from .model import SubBitModel, InitMethod, ste_sign
from .losses import SubBitLoss, orthogonal_regularization, boundary_guard_loss
from .scoring import maxsim
from .evaluation import compute_mrr, compute_recall, compute_ndcg, compute_per_query_ndcg, compute_rank_correlation, evaluate_retrieval, rerank_candidates, score_document
from .training import Trainer
from .encoders import MultiVectorEncoder, ColBERTEncoder, PrecomputedEncoder

__all__ = [
    "SubBitModel",
    "SubBitLoss",
    "InitMethod",
    "ste_sign",
    "orthogonal_regularization",
    "boundary_guard_loss",
    "maxsim",
    "compute_mrr",
    "compute_recall",
    "compute_ndcg",
    "compute_per_query_ndcg",
    "compute_rank_correlation",
    "evaluate_retrieval",
    "rerank_candidates",
    "score_document",
    "Trainer",
    "MultiVectorEncoder",
    "ColBERTEncoder",
    "PrecomputedEncoder",
]
