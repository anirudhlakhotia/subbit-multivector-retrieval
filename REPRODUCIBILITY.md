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

Each directory must contain `doc_embeddings.pt`, `query_embeddings.pt`, and
`qrels.tsv`. The loader only resolves the canonical name `query_embeddings.pt`;
in `100k_aug/` that file is the 32-token augmented query cache.

## Generating the embedding caches

If you do not already have the caches above, produce them with frozen ColBERTv2
(weights fetched from HuggingFace; MS MARCO read through `ir_datasets`):

```bash
python training/encode_corpus.py --config configs/base.yaml
```

This encodes the MS MARCO passage slice and dev queries set in
`configs/base.yaml` and writes `doc_embeddings.pt` and `query_embeddings.pt` to
`data.embeddings_dir` (see `--help` for `--output-dir`, `--max-passages`, and the
full-corpus `--use-mmap` options).

One step is easy to miss: the canonical evals use 32-token augmented dev queries,
but `encode_corpus.py` stores queries with augmentation stripped (about 8 real
tokens), which scores around 0.817 MRR@10 instead of the augmented 0.864. Build
the augmented cache that the `100k_aug/` slice reads with:

```bash
python training/encode_queries_augmented.py \
  --out data/embeddings/msmarco/100k_aug/query_embeddings.pt
```

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

Most scripts accept `--device cpu`. On Apple Silicon the default
(`--device auto`) selects MPS, where large-tensor `torch.topk` can crash or
misrank, so pass `--device cpu` on every command that accepts it. CPU mode is
fully functional, just slower. Two exceptions: `run_baseline_comparison.py` has
no `--device` flag and reads the device from the config, so append
`hardware.device=cpu` to its command line instead; the figure and `statistics/`
scripts have no device option (they run on CPU / read JSON).

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
  data.embeddings_dir=data/embeddings/msmarco/100k_aug \
  hardware.device=cpu
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
  --subbit-ablate-scale \
  data.embeddings_dir=data/embeddings/msmarco/100k_aug \
  hardware.device=cpu
```

Other 100k baselines:

```bash
python evaluation/eval_itq_baseline.py \
  --config configs/base.yaml \
  --embeddings-dir data/embeddings/msmarco/100k_aug \
  --r 64 \
  --device cpu \
  --output outputs/aug_eval/itq_100k_aug.json

python evaluation/eval_pq_msmarco_dev.py \
  --embeddings-dir data/embeddings/msmarco/100k_aug \
  --device cpu \
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
  --device cpu \
  --output outputs/aug_eval/preservation_aug.json
```

Random / PCA / identity / symmetric baselines:

```bash
python evaluation/evaluate_rank_preservation.py \
  --variants random_projection pca_projection identity_truncation \
  --device cpu \
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
  --checkpoint artifacts/checkpoints/50k_topk/best.pt \
  --emb-dir data/embeddings/msmarco/100k_aug \
  --residual-output outputs/c6_flip_residuals_aug100k_r64.npz

python diagnostics/diagnose_worst_queries.py
python diagnostics/diagnose_missed_vs_resolved.py
```

The `--residual-output` above writes `outputs/c6_flip_residuals_aug100k_r64.npz`,
which `figures/plot_mechanism_figures.py` reads, so run this before that figure.
These diagnostics run on CPU and take no `--device` flag.

Paper provenance:

```
artifacts/paper/results/figure_01_three_levels.json
artifacts/paper/results/figure_02_argmax_mechanism.json
```

## Symmetric scoring

```bash
python evaluation/eval_symmetric_vs_asymmetric.py --device cpu
```

Output: `outputs/aug_eval/symmetric_aug.json`.

## Sign-only encoding

```bash
python evaluation/evaluate_sign_only.py --device cpu
```

Output: `outputs/sign_only_eval.json`. The ColBERTv2 half uses the bundled
`100k` cache; the ConstBERT half needs a separate
`data/embeddings/constbert/100k` cache that is not bundled in this release and
is skipped automatically when absent.

## Two-stage rerank

```bash
python evaluation/evaluate_two_stage_rerank.py \
  --checkpoint artifacts/checkpoints/50k_topk/best.pt \
  --embeddings-dir data/embeddings/msmarco/100k_aug \
  --device cpu \
  --output outputs/aug_eval/rerank_aug_fullfp32.json
```

The random-R rerank comparison uses a separate checkpoint:

```bash
python evaluation/evaluate_two_stage_rerank.py \
  --checkpoint artifacts/checkpoints/ablation_random_plus_scale/best.pt \
  --embeddings-dir data/embeddings/msmarco/100k_aug \
  --device cpu \
  --output outputs/aug_eval/rerank_aug_randR.json
```

Paper provenance:

```
artifacts/paper/results/table_04_two_stage_retrieval.json
```

## Latency

Run these on AC power: the harness refuses to benchmark on battery (it exits
with a message to pass `--allow-battery`), and battery numbers are throttled and
will not match the paper.

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

Official PLAID helpers. The official PLAID indexing engine needs `faiss` and the
full ColBERT indexing stack, which this repo's environment does not install; set
up a separate `.venv-plaid` with `colbert-ai` and `faiss` and run these three
with that interpreter. `build_plaid_official_subset.py` also reads the raw
MS MARCO passage collection from the local `ir_datasets` cache
(`~/.ir_datasets/msmarco-passage/`), which `ir_datasets` downloads on first use.
The index and search steps require an index name:

```bash
python plaid/build_plaid_official_subset.py
python plaid/plaid_official_index.py  --name msmarco100k.nbits2
python plaid/plaid_official_search.py --index msmarco100k.nbits2
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

Rendered figures are written to `paper/figures/`, where the manuscript's
`\includegraphics{figures/...}` paths resolve them.

## Full 8.8M and BEIR (Modal)

These runs need Modal credentials and a configured Modal volume. Scripts live
under `full_scale/`:

```bash
modal run --detach full_scale/eval_subbit_full.py::run
modal run --detach full_scale/eval_rand_plus_scale_full.py::run
modal run --detach full_scale/eval_identity_plus_scale_full.py::run
modal run --detach full_scale/eval_subbit_r128_full.py::run
modal run --detach full_scale/eval_rand128_plus_scale_full.py::run
modal run --detach full_scale/eval_identity128_plus_scale_full.py::run
modal run --detach full_scale/eval_itq_full.py::run
modal run --detach full_scale/eval_rabitq_full.py::run
modal run --detach full_scale/eval_rerank_8m_full.py::run   # run AFTER eval_subbit_full (reads its stage-1 sidecar)
modal run --detach full_scale/eval_beir9_full_r64.py::run
```

`eval_rerank_8m_full.py` consumes the stage-1 per-query sidecar written by
`eval_subbit_full.py`, so run the subbit job first.

**Per-query sidecars are not shipped.** The bootstrap and CI scripts below read
per-query score sidecars (`*.per_query.pt`, ~1.4 GB total) that the Modal jobs
above write to the Modal volume. They are too large for git and are not in this
repository; the curated result JSONs under `artifacts/paper/results/` already
contain the final numbers. To re-run the scripts, first run the Modal jobs,
download their `*.per_query.pt` sidecars into `artifacts/sidecars/full_msmarco/`
(and `artifacts/sidecars/beir_ci/r64/`), and symlink them into
`outputs/full_msmarco/` to match the paths the scripts expect. `bootstrap_8m_r128.py`
specifically needs `rand_r128_full_msmarco.per_query.pt` and
`sign_d_full_msmarco.per_query.pt` from the r128 Modal jobs.

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
