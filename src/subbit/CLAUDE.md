# CLAUDE.md - src/subbit

Canonical SubBit package for the paper reproduction code. Public API is
exported via `src/subbit/__init__.py` and re-exported at `src/__init__.py`.
No backward-compat aliases or non-paper model variants are active here.

## Modules

- **`model.py`** - `SubBitModel(input_dim=128, projected_dim=r, init_method,
  use_scale=True, orthogonal_constraint=True, freeze_R=False)`, `InitMethod`,
  `ste_sign`, and `hard_sign`. The paper model stores each document token as
  `sign(R d)` and scores projected fp32 query tokens, optionally multiplied by
  the bounded query-side scale head.
- **`losses.py`** - `SubBitLoss` wraps boundary-guard top-k basin shaping,
  adversarial sentinel anchoring, and optional Stiefel regularization. Forward
  returns only `{"total", "boundary_topk", "boundary_fp", "ortho"}`.
- **`scoring.py`** - `maxsim`, batched MaxSim helpers, storage accounting,
  and RaBitQ scoring helpers used by the paper scripts.
- **`data.py`** - `EmbeddingStore`, datasets, collation, qrel loading,
  synthetic debug data, and embedding-cache path resolution.
- **`training.py`** - `Trainer` for the canonical r=64 paper recipe.
  `Trainer.train()` returns the paper-ready record with best metric, eval
  history, final losses, measurement block, training time, and checkpoint path.
- **`evaluation.py`** - metrics, ranking helpers, reranking, and baseline
  evaluation. Runtime measurement metadata is stored under private
  `_measurement` keys and lifted by the trainer.
- **`baselines.py`** - random projection, PCA, identity truncation, and
  PQ baselines.
- **`baselines_plaid.py`** - PLAID-style centroid/residual baseline used by the
  paper table.
- **`encoders.py`** - `MultiVectorEncoder` protocol plus `ColBERTEncoder` and
  `PrecomputedEncoder`.
- **`measurement.py`** - latency, memory, metadata, and storage instrumentation
  for reviewer-ready runs.
- **`rank_preservation.py`** - rank-fidelity helpers (Spearman, Kendall,
  overlap@K) used by the paper diagnostics.
- **`utils.py`** - seeding, device selection, Stiefel projection, and logging
  helpers.

## When adding paper code

Keep new code tied to a figure, table, metric, or reproduction command in
`REPRODUCIBILITY.md`.
