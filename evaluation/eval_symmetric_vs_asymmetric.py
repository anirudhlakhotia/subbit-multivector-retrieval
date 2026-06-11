"""Measure the MRR@10 / R@100 / R@1000 cost of symmetric (sign(R·q) × sign(R·d))
scoring at r=64 on the 100k augmented corpus.

Scoring pipeline:
  q_b = sign(R·q)                — query side binarised
  d_b = sign(R·d)                — same as the existing SubBit doc encoding
  score(q, d) = Σ_i max_j (q_b[i] · d_b[j])  — equivalent to popcount MaxSim

Doc encoding is identical to the asymmetric case, so any delta is strictly
attributable to query-side binarisation.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.subbit.data import EmbeddingStore, load_qrels, resolve_embedding_cache_path
from src.subbit.evaluation import compute_mrr, compute_recall
from src.subbit.model import SubBitModel
from src.subbit.utils import ensure_dir, get_device, seed_everything, setup_logging

logger = logging.getLogger(__name__)

# Asymmetric baseline (trained SubBit r=64, doc=sign(Rd), query=Rq)
# for the 100k augmented corpus.  Verified against:
#   outputs/aug_eval/baseline_100k_aug_r64.json (SubBit r=64 entry)
#   outputs/aug_eval/rerank_aug_fullfp32.json (stage1_binary entry)
ASYM_CACHED = {
    "mrr@10": 0.8496173306954092,
    "recall@100": 0.990974212034384,
    "recall@1000": 0.9971585482330467,
    "source": "baseline_100k_aug_r64.json → SubBit r=64 (trained) entry",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--embeddings-dir", type=Path, default=Path("data/embeddings/msmarco/100k_aug"))
    p.add_argument("--checkpoint", type=Path, default=Path("artifacts/checkpoints/50k_topk/best.pt"))
    p.add_argument("--output", type=Path, default=Path("outputs/aug_eval/symmetric_aug.json"))
    p.add_argument("--max-queries", type=int, default=-1)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    seed_everything(args.seed)
    device = torch.device(args.device) if args.device else get_device("auto")
    logger.info("device=%s", device)

    # ---- Data ------------------------------------------------------------
    qs = EmbeddingStore(resolve_embedding_cache_path(args.embeddings_dir, "query"), mode="dict")
    qs.load()
    ds = EmbeddingStore(resolve_embedding_cache_path(args.embeddings_dir, "doc"), mode="dict")
    ds.load()
    qrels = load_qrels(args.embeddings_dir / "qrels.tsv")
    known = set(qs.get_all_ids())
    query_ids = [q for q in qrels.keys() if q in known]
    if 0 < args.max_queries < len(query_ids):
        query_ids = query_ids[: args.max_queries]
    doc_ids = ds.get_all_ids()
    logger.info("queries=%d  docs=%d", len(query_ids), len(doc_ids))

    # ---- Model -----------------------------------------------------------
    model = SubBitModel.load(str(args.checkpoint), device=device)
    model.to(device).eval()

    # ---- Encode docs once (±1 codes) -------------------------------------
    t0 = time.time()
    encoded = {}
    for did in doc_ids:
        try:
            emb = ds.get(did).to(device)
        except (KeyError, FileNotFoundError):
            continue
        with torch.no_grad():
            encoded[did] = model.encode_document(emb, use_ste=False)
    logger.info("Encoded %d docs in %.1fs", len(encoded), time.time() - t0)

    valid = list(encoded.keys())
    max_len = max(e.shape[0] for e in encoded.values())
    r = encoded[valid[0]].shape[-1]
    doc_padded = torch.zeros(len(valid), max_len, r, device=device)
    doc_mask = torch.zeros(len(valid), max_len, dtype=torch.bool, device=device)
    for i, did in enumerate(valid):
        d = encoded[did]
        doc_padded[i, : d.shape[0]] = d
        doc_mask[i, : d.shape[0]] = True

    # ---- Symmetric scoring sweep -----------------------------------------
    rankings: dict[str, list[str]] = {}
    t0 = time.time()
    with torch.no_grad():
        for qid in query_ids:
            try:
                q_emb = qs.get(qid).to(device)
            except (KeyError, FileNotFoundError):
                continue
            q_b = model.encode_query(q_emb, symmetric=True, use_ste=False)  # ±1
            sim = torch.einsum("md,cnd->cmn", q_b, doc_padded)
            sim = sim.masked_fill(~doc_mask.unsqueeze(1), float("-inf"))
            scores = sim.max(dim=-1).values.sum(dim=-1)
            k_eff = min(1000, scores.shape[0])
            _, idx = torch.topk(scores, k=k_eff)
            rankings[qid] = [valid[i] for i in idx.cpu().tolist()]
    elapsed = time.time() - t0

    sym = {
        "mrr@10": compute_mrr(rankings, qrels, 10),
        "recall@100": compute_recall(rankings, qrels, 100),
        "recall@1000": compute_recall(rankings, qrels, 1000),
        "rank_time_sec": elapsed,
        "mean_latency_ms": 1000.0 * elapsed / max(1, len(rankings)),
    }
    logger.info(
        "symmetric    MRR@10 = %.4f   R@100 = %.4f   R@1000 = %.4f   (rank=%.1fs, %.2fms/q)",
        sym["mrr@10"], sym["recall@100"], sym["recall@1000"], elapsed, sym["mean_latency_ms"],
    )
    logger.info(
        "asymmetric*  MRR@10 = %.4f   R@100 = %.4f   R@1000 = %.4f   (cached)",
        ASYM_CACHED["mrr@10"], ASYM_CACHED["recall@100"], ASYM_CACHED["recall@1000"],
    )

    delta = {
        "mrr@10_abs": sym["mrr@10"] - ASYM_CACHED["mrr@10"],
        "mrr@10_rel_%": 100.0 * (sym["mrr@10"] - ASYM_CACHED["mrr@10"]) / max(1e-9, ASYM_CACHED["mrr@10"]),
        "recall@100_abs": sym["recall@100"] - ASYM_CACHED["recall@100"],
        "recall@1000_abs": sym["recall@1000"] - ASYM_CACHED["recall@1000"],
    }
    logger.info(
        "Δ symmetric − asymmetric:  MRR@10 %+.4f (%+.2f%%)   R@100 %+.4f   R@1000 %+.4f",
        delta["mrr@10_abs"], delta["mrr@10_rel_%"],
        delta["recall@100_abs"], delta["recall@1000_abs"],
    )

    ensure_dir(args.output.parent)
    payload = {
        "config": {
            "embeddings_dir": str(args.embeddings_dir),
            "checkpoint": str(args.checkpoint),
            "r": r,
            "num_queries": len(query_ids),
            "num_docs": len(valid),
            "device": str(device),
            "seed": args.seed,
        },
        "symmetric": sym,
        "asymmetric_cached": ASYM_CACHED,
        "delta_symmetric_minus_asymmetric": delta,
    }
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
