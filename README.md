# Most MaxSim Winners Flip, Retrieval Survives: Low-Margin Substitution in Sign-Coded Late Interaction

[Anirudh Lakhotia](https://github.com/anirudhlakhotia), Nischal Helagally Shantharaju

Couchbase, Inc., USA

*Accepted at ReNeuIR 2026*

[Paper (PDF)](paper/paper_reneuir.pdf)

## Abstract

Late-interaction retrieval appears unusually robust to extreme compression. We study a sub-bit sign-coded ColBERT representation that stores each 128-dimensional document token as the 64 signs of a projected vector (0.5 bits per original dimension, 8 B/tok). On a 100k-passage MS MARCO diagnostic slice, sign coding changes the MaxSim winner for roughly 70% of query tokens, yet relevant-document recall remains near the fp32 ceiling and MRR@10 is recoverable by a short fp32 rerank. We find that sign coding substantially disrupts token-level structure while leaving retrieval effectiveness largely intact. We trace this robustness to low-margin substitution: although sign coding frequently changes the winning document token, the replacement usually has a similar fp32 score, so the resulting loss in the MaxSim sum remains small. The remaining retrieval loss is therefore concentrated in head ordering among near-tied candidates rather than in losing relevant documents from the candidate set. The mechanism also suggests that projection learning has limited room to help once errors are dominated by low-margin substitutions. Consistent with this prediction, on the full 8.8M-passage MS MARCO corpus a trained projection provides no statistically reliable advantage in MRR@10 or Recall@1000 over a random orthogonal projection at matched storage. These findings suggest that extreme compression for late-interaction retrieval need not faithfully preserve token-level geometry. A simple projection-training-free recipe -- a random 64-dimensional projection, 8 B/tok sign-coded document storage, and fp32 reranking of the top 100 candidates -- recovers fp32 MRR@10.

## Reproducing the results

[REPRODUCIBILITY.md](REPRODUCIBILITY.md) lists the exact command for every table
and figure in the paper. [MANIFEST.md](MANIFEST.md) maps the repository layout,
and the [Makefile](Makefile) wraps the common runs (`make test`, `make eval-100k`,
`make rerank-100k`, `make figures`).

```bash
uv sync --extra dev                 # or: pip install -e ".[dev]"
uv run --extra dev pytest tests/ -v
```

The MS MARCO embedding caches are an external prerequisite and are not tracked
here; see the Prerequisites section of REPRODUCIBILITY.md for where to place
them. Trained checkpoints and the curated per-table result JSONs are under
`artifacts/`. On Apple Silicon, run on CPU: the eval scripts auto-select MPS,
where large-tensor `torch.topk` is unreliable. REPRODUCIBILITY.md gives the
per-script flag (`--device cpu` for most scripts, `hardware.device=cpu` for the
baseline-table script, which has no `--device` flag).
