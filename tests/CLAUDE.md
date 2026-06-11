# CLAUDE.md - tests/

Pytest suite for the active paper path. Run with `pytest tests/ -v`.

## Files

- `test_model.py` - `SubBitModel` shapes, STE gradients through `sign`, init
  methods, Stiefel projection, query scale head, and save/load roundtrip.
- `test_losses.py` - `SubBitLoss` return keys (`total`, `boundary_topk`,
  `boundary_fp`, `ortho`), gradient flow, boundary-guard math, and orthogonal
  regularization.
- `test_scoring.py` - MaxSim variants, batched scoring, storage accounting,
  and r=128 identity parity.
- `test_baselines.py`, `test_baseline_correctness.py`,
  `test_baselines_plaid.py`, `test_data.py`, `test_evaluation.py`,
  `test_rank_preservation.py`, `test_scale_sweep_subset.py`,
  and `test_training.py` - active module-specific coverage.
- `smoke_test.py` - end-to-end import, encode/score, and save/load smoke
  coverage.

## Conventions

- Every fixture uses `input_dim=128` (the ColBERTv2 dimension). Change at the
  fixture level, never per-test.
- Synthetic data is `torch.randn`. sklearn PCA can print harmless
  `RuntimeWarning: divide by zero / invalid value in matmul`.
- Tests import canonical names (`SubBitModel`, `SubBitLoss`). Non-paper model
  variants and backward-compat aliases are not part of this repo.
- `use_scale=True` is the default. Tests that need pure-R behavior must pass
  `use_scale=False`.
- CPU is used throughout; no GPU/MPS is required.
