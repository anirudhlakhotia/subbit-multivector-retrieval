"""Iterative Quantization (Gong & Lazebnik, CVPR 2011) baseline at r=64.

Pipeline matches the existing PCA / random-projection / identity binary
baselines, with R replaced by an ITQ-rotated PCA basis:

  1. Fit PCA on a sample of training-corpus tokens → R_PCA (r × d).
  2. Project the same sample: V = X · R_PCA^T  (N × r).
  3. Iteratively learn an r × r orthogonal rotation Q minimising
     ||sign(V Q) − V Q||_F^2 (Procrustes step):
       a. B = sign(V Q)
       b. SVD(V^T B) = U Σ V_t^T
       c. Q ← V_t U^T
  4. The deployed projection becomes R = Q^T · R_PCA  (r × d).
  5. Encode docs as sign(R d) and queries as either R q (asymmetric) or
     sign(R q) (symmetric); rank with the canonical harness.

Companion to evaluation/run_baseline_comparison.py: shares the rank_all
implementation so MRR@10, R@100, R@1000 reproduce the paper conventions.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from omegaconf import OmegaConf

from src.subbit.data import EmbeddingStore, load_qrels, resolve_embedding_cache_path
from src.subbit.evaluation import compute_mrr, compute_recall
from evaluation.run_baseline_comparison import rank_all

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def fit_pca(sample: np.ndarray, r: int) -> np.ndarray:
    from sklearn.decomposition import PCA
    pca = PCA(n_components=r)
    pca.fit(sample)
    return pca.components_.astype(np.float32)  # (r, d)


def fit_itq(V: np.ndarray, n_iters: int = 50, seed: int = 42) -> np.ndarray:
    """Run ITQ on PCA-projected data V (N, r). Returns Q (r, r) orthogonal."""
    rng = np.random.default_rng(seed)
    r = V.shape[1]
    Q, _ = np.linalg.qr(rng.standard_normal((r, r)).astype(np.float64))
    V64 = V.astype(np.float64)
    prev_loss = float("inf")
    for it in range(n_iters):
        B = np.sign(V64 @ Q)
        B[B == 0] = 1.0
        # Closest orthogonal Q maximising tr(B^T V Q)  = tr(M Q) with M = V^T B.
        M = V64.T @ B  # (r, r)
        U, _, Vt = np.linalg.svd(M)
        Q = Vt.T @ U.T
        loss = float(np.linalg.norm(B - V64 @ Q, ord="fro") ** 2)
        if it == 0 or (it + 1) % 10 == 0 or it == n_iters - 1:
            log.info("  itq iter %2d  loss=%.4e  delta=%.4e",
                     it + 1, loss, prev_loss - loss)
        prev_loss = loss
    return Q.astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--embeddings-dir", default=None,
                    help="Override cfg.data.embeddings_dir (e.g. "
                         "data/embeddings/msmarco/100k_aug for augmented dev queries).")
    ap.add_argument("--query-embeddings", default=None,
                    help="Override the query cache path (e.g. an augmented "
                         "query_embeddings_aug.pt). Defaults to query_embeddings.pt in embeddings-dir.")
    ap.add_argument("--r", type=int, default=64)
    ap.add_argument("--pca-sample", type=int, default=50_000)
    ap.add_argument("--itq-iters", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-queries", type=int, default=-1)
    ap.add_argument("--device", default="mps",
                    help="cpu|mps|cuda — default mps to match the other "
                         "baseline rows in baseline_comparison.json")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    OmegaConf.set_struct(cfg, False)
    OmegaConf.resolve(cfg)
    embeddings_dir = Path(args.embeddings_dir) if args.embeddings_dir else Path(cfg.data.embeddings_dir)
    log.info("Loading stores from %s", embeddings_dir)

    device = torch.device(args.device)
    log.info("Using device=%s", device)
    query_cache = (Path(args.query_embeddings) if args.query_embeddings
                   else resolve_embedding_cache_path(embeddings_dir, "query"))
    log.info("Query cache: %s", query_cache)
    query_store = EmbeddingStore(query_cache, mode="dict")
    query_store.load()
    doc_store = EmbeddingStore(resolve_embedding_cache_path(embeddings_dir, "doc"),
                               mode="dict")
    doc_store.load()
    qrels = load_qrels(embeddings_dir / "qrels.tsv")
    doc_ids = doc_store.get_all_ids()
    log.info("  %d docs, %d qrels", len(doc_ids), len(qrels))

    # Sample tokens for PCA + ITQ fit (matches the PCA baseline harness).
    log.info("Sampling %d tokens for PCA + ITQ fit...", args.pca_sample)
    sample = doc_store.sample_embeddings(args.pca_sample)
    sample_np = sample.cpu().numpy().astype(np.float32)
    log.info("  sample shape=%s", sample_np.shape)

    log.info("Fitting PCA r=%d ...", args.r)
    t0 = time.time()
    R_pca = fit_pca(sample_np, args.r)  # (r, d)
    log.info("  PCA fit in %.1fs; explained-var trace=%.3f",
             time.time() - t0,
             float(np.var(sample_np @ R_pca.T) / np.var(sample_np)))

    log.info("Running ITQ for %d iterations on PCA-projected sample...",
             args.itq_iters)
    V = sample_np @ R_pca.T  # (N, r)
    t0 = time.time()
    Q = fit_itq(V, n_iters=args.itq_iters, seed=args.seed)  # (r, r)
    log.info("  ITQ fit in %.1fs", time.time() - t0)

    # Combined deployed projection: R = Q^T · R_PCA  (r × d).
    R_full = (Q.T @ R_pca).astype(np.float32)
    log.info("R_full shape=%s, ||R||_F=%.4f",
             R_full.shape, float(np.linalg.norm(R_full)))

    R_t = torch.from_numpy(R_full).to(device)

    def enc_q_asym(e: torch.Tensor) -> torch.Tensor:
        return e.to(device) @ R_t.T  # (..., r) float

    def enc_q_sym(e: torch.Tensor) -> torch.Tensor:
        proj = e.to(device) @ R_t.T
        b = torch.sign(proj)
        b[b == 0] = 1.0
        return b

    def enc_d(e: torch.Tensor) -> torch.Tensor:
        proj = e.to(device) @ R_t.T
        b = torch.sign(proj)
        b[b == 0] = 1.0
        return b

    out = {
        "harness": {
            "embeddings_dir": str(embeddings_dir),
            "n_docs": len(doc_ids),
            "n_qrels": len(qrels),
            "r": args.r,
            "pca_sample": args.pca_sample,
            "itq_iters": args.itq_iters,
            "seed": args.seed,
            "device": str(device),
        },
        "results": {},
    }

    for label, enc_q, binary_docs in [
        ("itq_asym", enc_q_asym, True),
        ("itq_sym",  enc_q_sym,  True),
    ]:
        log.info("Ranking with %s ...", label)
        t0 = time.time()
        rankings, _ = rank_all(
            query_store, doc_store, qrels, enc_q, enc_d,
            doc_ids, device, args.max_queries,
            eval_mode="float", binary_docs=binary_docs, bytes_per_token=args.r // 8,
        )
        wall = time.time() - t0
        mrr10 = compute_mrr(rankings, qrels, k=10)
        r100 = compute_recall(rankings, qrels, k=100)
        r1000 = compute_recall(rankings, qrels, k=1000)
        log.info("  %s: MRR@10=%.4f  R@100=%.4f  R@1000=%.4f  (%.1fs)",
                 label, mrr10, r100, r1000, wall)
        out["results"][label] = {
            "mrr@10": float(mrr10),
            "recall@100": float(r100),
            "recall@1000": float(r1000),
            "wall_seconds": float(wall),
            "n_queries": len(rankings),
        }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    log.info("Wrote %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
