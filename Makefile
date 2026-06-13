UV ?= uv
PYTHON ?= $(UV) run python
PYTEST ?= $(UV) run --extra dev pytest

.PHONY: test train-50k eval-100k eval-100k-plaid rerank-100k latency-table latency-rerank figures

test:
	$(PYTEST) tests/ -v

train-50k:
	$(PYTHON) training/train.py --config configs/base.yaml run.name=scale_50k

eval-100k:
	$(PYTHON) evaluation/run_baseline_comparison.py --config configs/base.yaml \
	  --checkpoints artifacts/checkpoints/50k_topk/best.pt \
	  --regimes 64 --eval-mode float \
	  --output outputs/aug_eval/baseline_100k_aug_r64.json \
	  data.embeddings_dir=data/embeddings/msmarco/100k_aug hardware.device=cpu

eval-100k-plaid:
	$(PYTHON) evaluation/run_baseline_comparison.py --config configs/base.yaml \
	  --checkpoints artifacts/checkpoints/50k_topk/best.pt \
	  --regimes 64 --eval-mode float \
	  --skip-baselines outputs/aug_eval/baseline_100k_aug_r64.json \
	  --include-plaid --plaid-centroids 32768 --plaid-residual-bits 1 2 4 \
	  --output outputs/aug_eval/table_plaid_noscale_aug.json --subbit-ablate-scale \
	  data.embeddings_dir=data/embeddings/msmarco/100k_aug hardware.device=cpu

rerank-100k:
	$(PYTHON) evaluation/evaluate_two_stage_rerank.py \
	  --checkpoint artifacts/checkpoints/50k_topk/best.pt \
	  --embeddings-dir data/embeddings/msmarco/100k_aug --device cpu \
	  --output outputs/aug_eval/rerank_aug_fullfp32.json

latency-table:
	$(PYTHON) latency/bench_latency_interleaved.py --group table --rounds 5 \
	  --max-queries 500 --output outputs/latency/latency_interleaved_table.json

latency-rerank:
	$(PYTHON) latency/bench_latency_interleaved.py --group rerank --rounds 5 \
	  --max-queries 500 --rerank-cutoffs 100 256 1024 \
	  --output outputs/latency/latency_interleaved_rerank.json

figures:
	$(PYTHON) figures/plot_pareto_quality_storage.py
	$(PYTHON) figures/plot_three_levels.py
	$(PYTHON) figures/plot_mechanism_figures.py
	$(PYTHON) figures/plot_training_anatomy.py
