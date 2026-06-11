#!/usr/bin/env python3
"""Smoke test: verify all imports, class instantiation, encode/score pipeline."""
import sys, torch, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# --- Core imports ---
from src.subbit.model import SubBitModel, InitMethod
from src.subbit.losses import SubBitLoss, orthogonal_regularization
from src.subbit.scoring import maxsim
from src.subbit.evaluation import compute_mrr, compute_recall, compute_ndcg
from src.subbit.training import Trainer

print("✓ All imports OK")

# --- Instantiation ---
model = SubBitModel(128, 64, init_method="random_orthogonal")
loss_fn = SubBitLoss()
print(f"✓ SubBitModel: {model.input_dim}d → {model.projected_dim}d ({model.bits_per_dim:.3f} bits/dim)")

# --- Encode / score pipeline ---
q = torch.randn(5, 128)
d = torch.randn(20, 128)
with torch.no_grad():
    q_proj = model.encode_query(q)       # float (5, 64)
    d_bin  = model.encode_document(d)    # ±1   (20, 64)
    score  = maxsim(q_proj, d_bin)
assert score.dim() == 0 and torch.isfinite(score), "score check failed"
print(f"✓ MaxSim score: {score.item():.4f}")

# --- Metrics ---
rankings = {"q1": ["d1", "d2", "d3"], "q2": ["d2", "d1", "d3"]}
qrels    = {"q1": {"d1": 1}, "q2": {"d2": 1}}
mrr  = compute_mrr(rankings, qrels, k=10)
r100 = compute_recall(rankings, qrels, k=100)
ndcg = compute_ndcg(rankings, qrels, k=10)
print(f"✓ Metrics — MRR@10={mrr:.4f}, R@100={r100:.4f}, NDCG@10={ndcg:.4f}")

# --- Save / load ---
import tempfile, os
with tempfile.TemporaryDirectory() as tmp:
    p = os.path.join(tmp, "model.pt")
    model.save(p)
    loaded = SubBitModel.load(p)
    assert loaded.projected_dim == model.projected_dim
print("✓ Save / load OK")

print("\n✅  Smoke test PASSED — refactor is clean.")
