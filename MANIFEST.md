# Repository Manifest

Most MaxSim Winners Flip, Retrieval Survives: Low-Margin Substitution in Sign-Coded Late Interaction

## Code

- `src/subbit/` — model, loss, trainer, scoring, baselines, data, metrics, measurement, and rank-preservation helpers.
- `configs/base.yaml` — training config.
- `configs/debug.yaml` — small synthetic debug override.
- `training/` — training and embedding-generation entry points.
- `evaluation/` — baseline, ITQ, PQ/OPQ, RaBitQ, rank preservation, symmetric scoring, two-stage rerank, and MS MARCO evaluator scripts.
- `diagnostics/` — mechanism and margin diagnostics.
- `figures/` — figure builders.
- `latency/` — latency benchmarking harness.
- `plaid/` — PLAID helper scripts.
- `statistics/` — bootstrap and CI scripts.
- `utils/` — paper output builder (rebuilds the curated JSONs, including storage accounting), ITQ projection helper, and ConstBERT sign-coding helper.
- `full_scale/` — Modal jobs for full-scale paper tables.
- `tests/` — unit tests.

## Artifacts

- `artifacts/checkpoints/` — trained SubBit checkpoints and ablation variants.
- `artifacts/paper/results/` — curated paper table, figure, and prose-claim JSONs.
- `outputs/paper/` — symlinks to curated paper results.

The full 8.8M fp32 embedding mmap and Modal volume contents are external
prerequisites for rerunning the Modal jobs.
