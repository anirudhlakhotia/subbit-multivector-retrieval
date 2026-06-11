# CLAUDE.md - utils/

Utilities for release checks and paper artifact accounting. Most runnable paper
entry points live in `training/`, `evaluation/`, `diagnostics/`, `figures/`,
`latency/`, `statistics/`, or `full_scale/`.

## Active Utilities

| Script | Role |
|---|---|
| `precompute_itq_R.py` | Rebuild the ITQ projection artifact if the raw cache is available. |
| `sign_only_constbert_mps.py` | ConstBERT sign-only helper for artifact regeneration. |
