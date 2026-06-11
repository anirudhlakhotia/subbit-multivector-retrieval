#!/usr/bin/env python3
"""Dev-matched PQ-8x8 baseline for the paper.

PQ at 8 sub-quantizers x 8 bits = 64 bits = 8 B/tok, the canonical
matched-budget non-binary baseline against SubBit r=64. Codebook is
trained on tokens sampled from the same train-split docs SubBit uses,
so there is no eval-corpus leakage. Scoring uses asymmetric distance
computation (ADC): float query sub-vectors probe per-centroid lookup
tables; document tokens stay in compact code form.

Output mirrors the other 100k baseline evaluators:
    outputs/pq_8x8_100k/results.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.subbit.baselines import TorchProductQuantizer
from src.subbit.data import EmbeddingStore, load_qrels, resolve_embedding_cache_path
from src.subbit.evaluation import compute_mrr, compute_ndcg, compute_recall
from src.subbit.utils import seed_everything


PQ_M = 8                # sub-quantizers
PQ_BITS = 8             # bits per sub-quantizer
PQ_BYTES_PER_TOKEN = (PQ_M * PQ_BITS) // 8  # 8

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _load_config(path: str) -> OmegaConf:
    cfg = OmegaConf.load(path)
    OmegaConf.set_struct(cfg, False)
    OmegaConf.resolve(cfg)
    return cfg


def _train_doc_ids(triples_path: Path) -> set[str]:
    """Read train triples and return the set of doc ids that appear as
    positive or negative. Mirrors the leakage-avoidance pattern used for
    PCA init in training/train.py.
    """
    if not triples_path.exists():
        log.warning("Triples file %s not found; PQ will train on all docs", triples_path)
        return set()
    ids: set[str] = set()
    with triples_path.open("r") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            ids.add(str(parts[1]))
            ids.add(str(parts[2]))
    log.info("Derived %d unique train-split doc ids from %s", len(ids), triples_path)
    return ids


def _train_pq(
    doc_store: EmbeddingStore,
    train_doc_ids: set[str],
    *,
    n_train_tokens: int,
    seed: int,
) -> TorchProductQuantizer:
    if train_doc_ids:
        sample = doc_store.sample_embeddings(n_train_tokens, seed=seed, ids=list(train_doc_ids))
    else:
        sample = doc_store.sample_embeddings(n_train_tokens, seed=seed)
    log.info("Training PQ-%dx%d on %d sampled tokens", PQ_M, PQ_BITS, sample.shape[0])
    pq = TorchProductQuantizer(d=128, m=PQ_M, n_bits=PQ_BITS)
    pq.train(sample.float())
    return pq


def _encode_corpus(
    doc_store: EmbeddingStore,
    doc_ids: list[str],
    pq: TorchProductQuantizer,
    *,
    chunk_size: int,
) -> tuple[list[str], list[tuple[torch.Tensor, torch.Tensor]]]:
    """Encode every doc's tokens to PQ codes, padded into chunks.

    Returns:
        valid_doc_ids: doc ids actually present in the store.
        chunks: list of (codes_pad, mask) pairs.
            codes_pad: (n_docs, max_len, PQ_M) uint8 (codes are in [0, 256)).
            mask:      (n_docs, max_len) bool.
    """
    valid_doc_ids: list[str] = []
    chunks: list[tuple[torch.Tensor, torch.Tensor]] = []

    log.info("Encoding %d documents with PQ-%dx%d", len(doc_ids), PQ_M, PQ_BITS)
    for start in tqdm(range(0, len(doc_ids), chunk_size), desc="encode docs"):
        batch_ids = doc_ids[start : start + chunk_size]
        batch_codes: list[torch.Tensor] = []
        kept_ids: list[str] = []

        for pid in batch_ids:
            try:
                d = doc_store.get(pid).float()
            except (KeyError, FileNotFoundError):
                continue
            codes = pq.encode(d)              # (n_tokens, PQ_M) long
            batch_codes.append(codes.to(torch.uint8))
            kept_ids.append(str(pid))

        if not kept_ids:
            continue

        max_len = max(t.shape[0] for t in batch_codes)
        n_docs = len(kept_ids)
        codes_pad = torch.zeros(n_docs, max_len, PQ_M, dtype=torch.uint8)
        mask = torch.zeros(n_docs, max_len, dtype=torch.bool)
        for i, codes in enumerate(batch_codes):
            n_tok = codes.shape[0]
            codes_pad[i, :n_tok] = codes
            mask[i, :n_tok] = True

        valid_doc_ids.extend(kept_ids)
        chunks.append((codes_pad, mask))

    return valid_doc_ids, chunks


def _score_queries(
    query_store: EmbeddingStore,
    qrels: dict,
    valid_doc_ids: list[str],
    chunks: list[tuple[torch.Tensor, torch.Tensor]],
    pq: TorchProductQuantizer,
    *,
    device: torch.device,
    max_queries: int,
    top_k: int,
) -> tuple[dict[str, list[str]], float]:
    query_ids = list(qrels.keys())
    if 0 < max_queries < len(query_ids):
        query_ids = query_ids[:max_queries]

    rankings: dict[str, list[str]] = {}
    codebooks = pq.codebooks.to(device)         # (PQ_M, 256, ds)
    t0 = time.perf_counter()

    for qid in tqdm(query_ids, desc="rank PQ asym (ADC)"):
        try:
            q = query_store.get(qid).float().to(device)   # (m_q, 128)
        except (KeyError, FileNotFoundError):
            continue

        with torch.no_grad():
            # Build LUT: (m_q, PQ_M, 256). For each query token, each
            # sub-quantizer, dot-product against every centroid.
            q_sub = q.reshape(q.shape[0], PQ_M, pq.ds)
            tables = torch.einsum("qmd,mkd->qmk", q_sub, codebooks)
            # tables[q, sub_idx, code_id] = q_sub[q, sub_idx, :] . codebooks[sub_idx, code_id, :]

            all_scores = torch.empty(len(valid_doc_ids), device=device)
            offset = 0
            for codes_cpu, mask_cpu in chunks:
                codes = codes_cpu.to(device=device, dtype=torch.long)   # (n_docs, max_len, PQ_M)
                mask = mask_cpu.to(device=device)                       # (n_docs, max_len)
                n_docs, max_len, _ = codes.shape

                # Per (q, doc, tok), sum over sub-quantizers of LUT-gathered scores.
                # tables: (m_q, PQ_M, 256)
                # codes:  (n_docs, max_len, PQ_M)
                # Use advanced indexing per sub-quantizer to keep memory bounded.
                sim = torch.zeros(q.shape[0], n_docs, max_len, device=device)
                for sub_idx in range(PQ_M):
                    # tables[:, sub_idx, :] is (m_q, 256); index by codes[:, :, sub_idx] (n_docs, max_len)
                    sub_scores = tables[:, sub_idx, :][:, codes[:, :, sub_idx]]
                    # shape: (m_q, n_docs, max_len)
                    sim = sim + sub_scores

                sim = sim.masked_fill(~mask.unsqueeze(0), float("-inf"))
                # MaxSim: per-query-token max over doc tokens, sum over query tokens.
                scores = sim.max(dim=-1).values.sum(dim=0)
                all_scores[offset : offset + n_docs] = scores
                offset += n_docs

            actual_k = min(top_k, all_scores.numel())
            top_idx = all_scores.topk(actual_k).indices.cpu().tolist()
            rankings[str(qid)] = [valid_doc_ids[i] for i in top_idx]

    elapsed = time.perf_counter() - t0
    return rankings, elapsed


def _metrics(rankings: dict[str, list[str]], qrels: dict, elapsed: float) -> dict:
    return {
        "mrr@10": float(compute_mrr(rankings, qrels, k=10)),
        "ndcg@10": float(compute_ndcg(rankings, qrels, k=10)),
        "recall@100": float(compute_recall(rankings, qrels, k=100)),
        "recall@1000": float(compute_recall(rankings, qrels, k=1000)),
        "n_queries": len(rankings),
        "wall_seconds": float(elapsed),
        "mean_ms_per_query": float(1000.0 * elapsed / max(1, len(rankings))),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--embeddings-dir", default=None,
                        help="Override cfg.data.embeddings_dir, e.g. data/embeddings/msmarco/100k_aug.")
    parser.add_argument("--output", default="outputs/pq_8x8_100k/results.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-train-tokens", type=int, default=50000,
                        help="Tokens sampled from train-split docs to fit the PQ codebook")
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--max-docs", type=int, default=-1)
    parser.add_argument("--max-queries", type=int, default=-1)
    parser.add_argument("--top-k", type=int, default=1000)
    args = parser.parse_args()

    cfg = _load_config(args.config)
    embeddings_dir = Path(args.embeddings_dir or cfg.data.embeddings_dir)
    device = torch.device(args.device)
    seed_everything(args.seed)

    query_store = EmbeddingStore(resolve_embedding_cache_path(embeddings_dir, "query"), mode="dict")
    doc_store = EmbeddingStore(resolve_embedding_cache_path(embeddings_dir, "doc"), mode="dict")
    query_store.load()
    doc_store.load()
    qrels = load_qrels(embeddings_dir / "qrels.tsv")
    doc_ids = [str(pid) for pid in doc_store.get_all_ids()]
    if 0 < args.max_docs < len(doc_ids):
        doc_ids = doc_ids[: args.max_docs]

    # Mirror train.py resolution: prefer the cached triples that live next
    # to the encoded embeddings, fall back to the global config path.
    triples_path = embeddings_dir / "triples.tsv"
    if not triples_path.exists() and "train_triples" in cfg.data:
        triples_path = Path(cfg.data.train_triples)
    train_doc_ids = _train_doc_ids(triples_path)
    # Restrict to docs actually present in the store.
    if train_doc_ids:
        present = set(doc_ids)
        train_doc_ids = train_doc_ids & present
        log.info("%d train-split docs intersect the encoded corpus", len(train_doc_ids))

    pq = _train_pq(
        doc_store,
        train_doc_ids,
        n_train_tokens=args.n_train_tokens,
        seed=args.seed,
    )

    valid_doc_ids, chunks = _encode_corpus(
        doc_store,
        doc_ids,
        pq,
        chunk_size=args.chunk_size,
    )
    log.info("Encoded %d valid documents in %d chunks", len(valid_doc_ids), len(chunks))

    asym_rankings, asym_time = _score_queries(
        query_store,
        qrels,
        valid_doc_ids,
        chunks,
        pq,
        device=device,
        max_queries=args.max_queries,
        top_k=args.top_k,
    )

    output = {
        "config": {
            "embeddings_dir": str(embeddings_dir),
            "pq_m": PQ_M,
            "pq_bits": PQ_BITS,
            "bytes_per_token": PQ_BYTES_PER_TOKEN,
            "n_train_tokens": args.n_train_tokens,
            "n_train_docs_used": len(train_doc_ids) if train_doc_ids else None,
            "seed": args.seed,
            "device": str(device),
            "chunk_size": args.chunk_size,
            "max_docs": args.max_docs,
            "max_queries": args.max_queries,
            "top_k": args.top_k,
            "n_docs": len(valid_doc_ids),
        },
        "pq_asym": _metrics(asym_rankings, qrels, asym_time),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    log.info("Wrote %s", out_path)
    log.info("PQ-%dx%d (%d B/tok) MRR@10 = %.4f  NDCG@10 = %.4f  R@100 = %.4f  R@1000 = %.4f",
             PQ_M, PQ_BITS, PQ_BYTES_PER_TOKEN,
             output["pq_asym"]["mrr@10"],
             output["pq_asym"]["ndcg@10"],
             output["pq_asym"]["recall@100"],
             output["pq_asym"]["recall@1000"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
