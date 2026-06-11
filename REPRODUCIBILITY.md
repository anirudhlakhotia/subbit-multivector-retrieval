# Reproducibility Guide

This guide lists the commands and artifacts needed to reproduce the reported
paper numbers. Paths are relative to the release root.

## Prerequisites

The local scripts require MS MARCO embedding caches on disk. These are not
tracked in this repository and must be placed at:

```
data/embeddings/msmarco/100k/       (base corpus)
data/embeddings/msmarco/100k_aug/   (augmented queries, canonical eval slice)
```

Each directory must contain `doc_embeddings.pt`, `query_embeddings.pt` (or
`query_embeddings_aug.pt`), and `qrels.tsv`.

## Environment

```bash
uv sync --extra dev
uv run --extra dev pytest tests/ -v
```

For plain pip:

```bash
python -m pip install -e ".[dev]"
pytest tests/ -v
```

Most scripts default to `--device auto` (MPS on Apple Silicon). For machines
without MPS or where MPS NDArray size limits cause crashes, pass `--device cpu`
on every command. CPU mode is fully functional, just slower.

## Training

Canonical r=64 training:

```bash
python training/train.py --config configs/base.yaml run.name=scale_50k
```

The canonical checkpoint is:

```
artifacts/checkpoints/50k_topk/best.pt
```

Published paper-facing JSONs are under:

```
artifacts/paper/results/
```

The raw output paths in the commands below are local reproduction products.
When the raw local result files are present, run
`python utils/build_paper_outputs.py` to rebuild the curated paper JSONs.

## 100k storage-quality table

Main fp32 / identity / random / PCA / trained rows:

```bash
python evaluation/run_baseline_comparison.py \
  --config configs/base.yaml \
  --checkpoints artifacts/checkpoints/50k_topk/best.pt \
  --regimes 64 \
  --eval-mode float \
  --output outputs/aug_eval/baseline_100k_aug_r64.json \
  data.embeddings_dir=data/embeddings/msmarco/100k_aug
```

PLAID rows and trained no-scale row:

```bash
python evaluation/run_baseline_comparison.py \
  --config configs/base.yaml \
  --checkpoints artifacts/checkpoints/50k_topk/best.pt \
  --regimes 64 \
  --eval-mode float \
  --skip-baselines outputs/aug_eval/baseline_100k_aug_r64.json \
  --include-plaid \
  --plaid-centroids 32768 \
  --plaid-residual-bits 1 2 4 \
  --output outputs/aug_eval/table_plaid_noscale_aug.json \
  --no-scale-ablation \
  data.embeddings_dir=data/embeddings/msmarco/100k_aug
```

Other 100k baselines:

```bash
python evaluation/eval_itq_baseline.py \
  --config configs/base.yaml \
  --embeddings-dir data/embeddings/msmarco/100k_aug \
  --r 64 \
  --output outputs/aug_eval/itq_100k_aug.json

python evaluation/eval_pq_msmarco_dev.py \
  --embeddings-dir data/embeddings/msmarco/100k_aug \
  --output outputs/aug_eval/pq_opq_100k_aug.json

python evaluation/eval_rabitq_100k_aug.py \
  --output outputs/aug_eval/rabitq_100k_aug.json \
  --device cpu
```

Raw provenance artifacts:

```
artifacts/paper/results/table_01_msmarco_storage_quality.json
artifacts/paper/results/table_06_plaid_storage_quality.json
```

## Rank preservation

Learned asymmetric row (Table 3):

```bash
python evaluation/evaluate_rank_preservation.py \
  --variants learned_asymmetric \
  --output outputs/aug_eval/preservation_aug.json
```

Random / PCA / identity / symmetric baselines:

```bash
python evaluation/evaluate_rank_preservation.py \
  --variants random_projection pca_projection identity_truncation \
  --output outputs/aug_eval/preservation_aug_rest.json
```

Paper provenance:

```
artifacts/paper/results/table_03_rank_preservation.json
```

## Mechanism diagnostics

```bash
python diagnostics/geometry_fidelity_vs_argmax.py

python diagnostics/mechanism_decomposition.py \
  --checkpoint artifacts/checkpoints/50k_topk/best.pt

python diagnostics/diagnose_worst_queries.py
python diagnostics/diagnose_missed_vs_resolved.py
```

Paper provenance:

```
artifacts/paper/results/figure_01_three_levels.json
artifacts/paper/results/figure_02_argmax_mechanism.json
```

## Symmetric scoring

```bash
python evaluation/eval_symmetric_vs_asymmetric.py
```

Output: `outputs/aug_eval/symmetric_aug.json`.

## Sign-only encoding

```bash
python evaluation/evaluate_sign_only.py --device cpu
```

Output: `outputs/sign_only_eval.json`.

## Two-stage rerank

```bash
python evaluation/evaluate_two_stage_rerank.py \
  --checkpoint artifacts/checkpoints/50k_topk/best.pt \
  --embeddings-dir data/embeddings/msmarco/100k_aug \
  --device cpu
```

The random-R rerank comparison uses a separate checkpoint:

```bash
python evaluation/evaluate_two_stage_rerank.py \
  --checkpoint artifacts/checkpoints/ablation_random_plus_scale/best.pt \
  --embeddings-dir data/embeddings/msmarco/100k_aug \
  --device cpu
```

Paper provenance:

```
artifacts/paper/results/table_04_two_stage_retrieval.json
```

## Latency

```bash
python latency/bench_latency_interleaved.py \
  --group table \
  --rounds 5 \
  --max-queries 500 \
  --output outputs/latency/latency_interleaved_table.json

python latency/bench_latency_interleaved.py \
  --group rerank \
  --rounds 5 \
  --max-queries 500 \
  --rerank-cutoffs 100 256 1024 \
  --output outputs/latency/latency_interleaved_rerank.json
```

Official PLAID helpers:

```bash
python plaid/build_plaid_official_subset.py
python plaid/plaid_official_index.py
python plaid/plaid_official_search.py
```

Paper provenance:

```
artifacts/paper/results/table_05_rerank_latency.json
artifacts/paper/results/table_07_retrieval_latency.json
```

## Figures

```bash
python figures/plot_pareto_quality_storage.py
python figures/plot_three_levels.py
python figures/plot_mechanism_figures.py
python figures/plot_training_anatomy.py
```

Rendered figures are written to `figures/`.

## Full 8.8M and BEIR (Modal)

These runs need Modal credentials and a configured Modal volume. Scripts live
under `full_scale/`:

```bash
modal run full_scale/eval_subbit_full.py::run
modal run full_scale/eval_rand_plus_scale_full.py::run
modal run full_scale/eval_identity_plus_scale_full.py::run
modal run full_scale/eval_subbit_r128_full.py::run
modal run full_scale/eval_rand128_plus_scale_full.py::run
modal run full_scale/eval_identity128_plus_scale_full.py::run
modal run full_scale/eval_itq_full.py::run
modal run full_scale/eval_rabitq_full.py::run
modal run full_scale/eval_rerank_8m_full.py::run
modal run full_scale/eval_beir9_full_r64.py::run
```

Bootstrap and CI:

```bash
python statistics/bootstrap_8m_r64.py
python statistics/bootstrap_8m_r128.py
python statistics/compute_beir_ci.py
```

Paper provenance:

```
artifacts/paper/results/prose_01_low_rank_controls.json
artifacts/paper/results/prose_02_full_scale_rerank.json
artifacts/paper/results/prose_03_r128_rotation_control.json
artifacts/paper/results/table_02_beir_ndcg.json
```
