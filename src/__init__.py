"""Most MaxSim Winners Flip, Retrieval Survives: Low-Margin Substitution in Sign-Coded Late Interaction."""

__version__ = "0.1.0"

# Re-export everything from the canonical subbit package
from .subbit import (
    SubBitModel,
    SubBitLoss,
    InitMethod,
    ste_sign,
    orthogonal_regularization,
    boundary_guard_loss,
    maxsim,
    compute_mrr,
    compute_recall,
    compute_ndcg,
    evaluate_retrieval,
    Trainer,
)
