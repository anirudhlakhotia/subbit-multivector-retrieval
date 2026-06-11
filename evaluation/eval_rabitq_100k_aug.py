"""Correct RaBitQ baseline at 100k, on the augmented ColBERTv2 cache.

The paper's tab:pareto RaBitQ row (r=128, 24 B/tok) MUST use the same data as the
other rows: ColBERTv2 token embeddings, the 6,980 MS MARCO dev-small queries with
standard ColBERT augmentation. This script computes RaBitQ with the same verified
math as ``full_scale/eval_rabitq_full.py`` (the 8.8M version)
on the cached ``100k_aug`` embeddings, ranking with the canonical asymmetric MaxSim
plus the RaBitQ per-token correction.

RaBitQ (Gao et al. 2024), r=128:
  R = random orthogonal 128x128 (seed 42, QR);  y = R d;  b = sign(y);
  norm = ||y||;  vdot = <y, b> / (||y|| * sqrt(128));
  correction = norm / (vdot * sqrt(128));   <q, d> ~= correction * <R q, b>.

Usage:
  python evaluation/eval_rabitq_100k_aug.py \
      --embeddings-dir data/embeddings/msmarco/100k_aug \
      --output outputs/aug_eval/rabitq_100k_aug.json --device cpu
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

from src.subbit.data import EmbeddingStore, load_qrels, resolve_embedding_cache_path
from src.subbit.evaluation import compute_mrr, compute_recall

DIM = 128
SQRT_DIM = float(DIM ** 0.5)  # 11.3137...
RABITQ_BYTES_PER_TOK = 24     # 16 B packed sign (128 bits) + 4 B norm + 4 B vdot

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Testable RaBitQ primitives (match full_scale/eval_rabitq_full.py exactly).
# ---------------------------------------------------------------------------
def random_orthogonal(dim: int, seed: int, device: torch.device) -> torch.Tensor:
    """Fixed random-orthogonal rotation R (dim x dim) with orthonormal rows.

    Matches full_scale/eval_rabitq_full.py: torch.manual_seed(seed); Q,_=qr(randn); R=Q.T.
    """
    torch.manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(dim, dim))
    return Q.T.to(device).float()


def rabitq_encode(y: torch.Tensor):
    """Given rotated doc tokens y = R d  (..., DIM), return (binary, correction).

    binary in {-1, +1}^DIM; correction is the per-token factor such that
    <q, d> = <R q, y> ~= correction * <R q, binary>.
    """
    norm = y.norm(dim=-1)
    binary = torch.where(y >= 0, torch.ones_like(y), -torch.ones_like(y))
    inner = (y * binary).sum(dim=-1)
    vdot = inner / (norm.clamp_min(1e-10) * SQRT_DIM)
    correction = norm / (vdot.clamp_min(1e-10) * SQRT_DIM)
    return binary, correction


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embeddings-dir", default="data/embeddings/msmarco/100k_aug")
    ap.add_argument("--query-embeddings", default=None,
                    help="Override query cache path (default: query_embeddings.pt in embeddings-dir).")
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--doc-chunk", type=int, default=4096)
    ap.add_argument("--query-batch", type=int, default=64)
    ap.add_argument("--top-k", type=int, default=1000)
    ap.add_argument("--max-queries", type=int, default=-1)
    ap.add_argument("--device", default="cpu",
                    help="cpu|mps|cuda. CPU avoids the MPS NDArray>INT_MAX limit on "
                         "the full 100k padded tensor; default cpu for safety.")
    args = ap.parse_args()

    device = torch.device(args.device)
    emb_dir = Path(args.embeddings_dir)
    log.info("RaBitQ r=%d on %s (device=%s)", DIM, emb_dir, device)

    q_cache = (Path(args.query_embeddings) if args.query_embeddings
               else resolve_embedding_cache_path(emb_dir, "query"))
    log.info("Query cache: %s", q_cache)
    query_store = EmbeddingStore(q_cache, mode="dict"); query_store.load()
    doc_store = EmbeddingStore(resolve_embedding_cache_path(emb_dir, "doc"), mode="dict"); doc_store.load()
    qrels = load_qrels(emb_dir / "qrels.tsv")
    doc_ids = doc_store.get_all_ids()
    log.info("  %d docs, %d qrels", len(doc_ids), len(qrels))

    R = random_orthogonal(DIM, args.seed, device)
    orth_err = (R @ R.T - torch.eye(DIM, device=device)).abs().max().item()
    log.info("R: shape=%s, ||R||_F=%.4f, max|RR^T-I|=%.2e (seed=%d)",
             tuple(R.shape), R.norm().item(), orth_err, args.seed)
    assert orth_err < 1e-4, f"R not orthogonal (max|RR^T-I|={orth_err})"

    # ---- Queries (dev qrel set), project by R (no scale head) ----
    qids = [q for q in qrels if q in set(query_store.get_all_ids())]
    qids.sort()
    if 0 < args.max_queries < len(qids):
        qids = qids[:args.max_queries]
    Qn = len(qids)
    m_q = query_store.get(qids[0]).shape[0]
    Q_proj = torch.stack([query_store.get(q).to(device) for q in qids]).float() @ R.T  # (Qn, m, 128)
    log.info("  queries: %d x %s", Qn, tuple(Q_proj.shape[1:]))

    # ---- Doc tokens, sort by length, chunk; RaBitQ-encode + corrected MaxSim ----
    lengths = np.array([doc_store.get(p).shape[0] for p in doc_ids], dtype=np.int64)
    order = np.argsort(lengths, kind="stable")
    pids = [doc_ids[i] for i in order]
    lengths = lengths[order]
    Dn = len(pids)
    top_scores = torch.full((Qn, args.top_k), -float("inf"), device=device)
    top_indices = torch.full((Qn, args.top_k), -1, dtype=torch.int64, device=device)

    t0 = time.time()
    for da in range(0, Dn, args.doc_chunk):
        db = min(da + args.doc_chunk, Dn)
        seg = pids[da:db]
        seg_len = torch.from_numpy(lengths[da:db]).to(device)
        seg_max = int(seg_len.max().item())
        seg_n = db - da
        D_raw = torch.zeros(seg_n, seg_max, DIM, dtype=torch.float32, device=device)
        for i, p in enumerate(seg):
            e = doc_store.get(p).to(device).float()
            D_raw[i, :e.shape[0]] = e
        y = D_raw @ R.T                          # (seg_n, seg_max, 128)
        binary, correction = rabitq_encode(y)    # (seg_n,seg_max,128), (seg_n,seg_max)
        valid = torch.arange(seg_max, device=device)[None, :] < seg_len[:, None]
        binary = binary.to(torch.float32)
        del D_raw, y

        for qa in range(0, Qn, args.query_batch):
            qb = min(qa + args.query_batch, Qn)
            Q_b = Q_proj[qa:qb]                                  # (Qb, m, 128)
            sim = torch.einsum("qmr,dnr->qmdn", Q_b, binary)     # <Rq, b>
            sim = sim * correction[None, None, :, :]             # RaBitQ correction
            sim.masked_fill_(~valid[None, None, :, :], float("-inf"))
            scores = sim.max(dim=-1).values.sum(dim=1)           # (Qb, seg_n)
            del sim
            doc_idx = torch.arange(da, db, device=device).expand(qb - qa, -1)
            cs = torch.cat([top_scores[qa:qb], scores], dim=1)
            ci = torch.cat([top_indices[qa:qb], doc_idx], dim=1)
            top_scores[qa:qb], pos = cs.topk(args.top_k, dim=1)
            top_indices[qa:qb] = ci.gather(1, pos)
        del binary, correction, valid
        if (da // args.doc_chunk + 1) % 5 == 0:
            el = time.time() - t0
            done = db
            print(f"  {done:,}/{Dn:,} docs  {done/max(el,1):.0f} docs/s  "
                  f"ETA {(Dn-done)/max(done/max(el,1),1)/60:.1f}m", flush=True)

    # ---- Metrics ----
    top_idx = top_indices.cpu().numpy()
    rankings = {}
    for qi, qid in enumerate(qids):
        rankings[qid] = [pids[idx] for idx in top_idx[qi] if idx >= 0]
    mrr10 = compute_mrr(rankings, qrels, k=10)
    r100 = compute_recall(rankings, qrels, k=100)
    r1000 = compute_recall(rankings, qrels, k=1000)
    log.info("RaBitQ asym: MRR@10=%.4f  R@100=%.4f  R@1000=%.4f  (%.1fs)",
             mrr10, r100, r1000, time.time() - t0)

    out = {
        "method": "rabitq_asym",
        "embeddings_dir": str(emb_dir),
        "encoder": "colbert-ir/colbertv2.0",
        "r": DIM, "bytes_per_token": RABITQ_BYTES_PER_TOK,
        "scoring": "asymmetric (float Rq x sign(Rd) with norm/vdot correction)",
        "seed": args.seed, "n_docs": Dn, "n_queries": Qn,
        "metrics": {"mrr@10": float(mrr10), "recall@100": float(r100), "recall@1000": float(r1000)},
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    log.info("Wrote %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
