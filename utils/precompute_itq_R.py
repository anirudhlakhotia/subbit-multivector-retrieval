"""Precompute the ITQ projection R at r=64 and save as a tensor file
loadable via SubBitModel(init_method='from_file', init_path=...).

Mirrors the PCA + ITQ pipeline in evaluation/eval_itq_baseline.py exactly so
the resulting R is the same projection that Table 2's "ITQ projection"
row uses. Saves R as a torch tensor of shape (projected_dim, input_dim).

Usage
-----
  python utils/precompute_itq_R.py \
      --config configs/base.yaml \
      --r 64 \
      --output outputs/itq_init/R_r64.pt
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from omegaconf import OmegaConf

from src.subbit.data import EmbeddingStore, resolve_embedding_cache_path

# Reuse the exact ITQ + PCA fitters used by the baseline harness so
# the precomputed R is bit-identical to Table 2's "ITQ projection" row.
from evaluation.eval_itq_baseline import fit_itq, fit_pca

log = logging.getLogger(__name__)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--r", type=int, default=64)
    ap.add_argument("--pca-sample", type=int, default=50_000)
    ap.add_argument("--itq-iters", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")

    cfg = OmegaConf.load(args.config)
    OmegaConf.set_struct(cfg, False)
    OmegaConf.resolve(cfg)
    embeddings_dir = Path(cfg.data.embeddings_dir)
    log.info("Loading doc store from %s", embeddings_dir)

    doc_store = EmbeddingStore(resolve_embedding_cache_path(embeddings_dir, "doc"),
                               mode="dict")
    doc_store.load()

    log.info("Sampling %d tokens for PCA + ITQ fit", args.pca_sample)
    sample = doc_store.sample_embeddings(args.pca_sample)
    sample_np = sample.cpu().numpy().astype(np.float32)

    log.info("Fitting PCA r=%d", args.r)
    t0 = time.time()
    R_pca = fit_pca(sample_np, args.r)
    log.info("  PCA fit in %.1fs", time.time() - t0)

    log.info("Running ITQ for %d iterations", args.itq_iters)
    V = sample_np @ R_pca.T
    t0 = time.time()
    Q = fit_itq(V, n_iters=args.itq_iters, seed=args.seed)
    log.info("  ITQ fit in %.1fs", time.time() - t0)

    R_full = (Q.T @ R_pca).astype(np.float32)
    log.info("R_full shape=%s  ||R||_F=%.4f",
             R_full.shape, float(np.linalg.norm(R_full)))

    R_tensor = torch.from_numpy(R_full).contiguous()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(R_tensor, out_path)
    log.info("wrote %s (shape=%s, dtype=%s)", out_path,
             tuple(R_tensor.shape), R_tensor.dtype)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
