"""Evaluation metrics for multi-vector retrieval.

Implements:
  - MRR@K (Mean Reciprocal Rank)
  - Recall@K
  - NDCG@K
  - Rank correlation (Spearman's rho)
  - Per-query NDCG
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
from scipy import stats
from tqdm import tqdm

from .model import SubBitModel
from .scoring import maxsim
from .data import EmbeddingStore

logger = logging.getLogger(__name__)


def compute_mrr(
    rankings: dict[str, list[str]],
    qrels: dict[str, dict[str, int]],
    k: int = 10,
) -> float:
    """Compute Mean Reciprocal Rank at K.

    Args:
        rankings: dict mapping qid → ranked list of doc IDs.
        qrels: dict mapping qid → {pid → relevance}.
        k: Cutoff depth.

    Returns:
        MRR@K score.
    """
    mrr_sum = 0.0
    num_queries = 0

    for qid, ranked_docs in rankings.items():
        if qid not in qrels:
            continue
        num_queries += 1
        relevant = set(pid for pid, rel in qrels[qid].items() if rel > 0)

        for rank, pid in enumerate(ranked_docs[:k], start=1):
            if pid in relevant:
                mrr_sum += 1.0 / rank
                break

    return mrr_sum / max(num_queries, 1)


def compute_recall(
    rankings: dict[str, list[str]],
    qrels: dict[str, dict[str, int]],
    k: int = 1000,
) -> float:
    """Compute Recall at K.

    Args:
        rankings: dict mapping qid → ranked list of doc IDs.
        qrels: dict mapping qid → {pid → relevance}.
        k: Cutoff depth.

    Returns:
        Recall@K score.
    """
    recall_sum = 0.0
    num_queries = 0

    for qid, ranked_docs in rankings.items():
        if qid not in qrels:
            continue
        num_queries += 1
        relevant = set(pid for pid, rel in qrels[qid].items() if rel > 0)
        if not relevant:
            continue

        retrieved = set(ranked_docs[:k])
        recall_sum += len(relevant & retrieved) / len(relevant)

    return recall_sum / max(num_queries, 1)


def compute_ndcg(
    rankings: dict[str, list[str]],
    qrels: dict[str, dict[str, int]],
    k: int = 10,
) -> float:
    """Compute NDCG at K.

    Args:
        rankings: dict mapping qid → ranked list of doc IDs.
        qrels: dict mapping qid → {pid → relevance}.
        k: Cutoff depth.

    Returns:
        NDCG@K score.
    """
    ndcg_sum = 0.0
    num_queries = 0

    for qid, ranked_docs in rankings.items():
        if qid not in qrels:
            continue
        num_queries += 1

        # DCG
        dcg = 0.0
        for rank, pid in enumerate(ranked_docs[:k], start=1):
            rel = qrels[qid].get(pid, 0)
            dcg += (2**rel - 1) / np.log2(rank + 1)

        # Ideal DCG
        ideal_rels = sorted(qrels[qid].values(), reverse=True)[:k]
        idcg = sum((2**rel - 1) / np.log2(rank + 1) for rank, rel in enumerate(ideal_rels, 1))

        ndcg_sum += dcg / max(idcg, 1e-10)

    return ndcg_sum / max(num_queries, 1)


def compute_per_query_ndcg(
    rankings: dict[str, list[str]],
    qrels: dict[str, dict[str, int]],
    k: int = 10,
) -> dict[str, float]:
    """Compute NDCG@K per query (returns dict, not mean).

    Args:
        rankings: dict mapping qid -> ranked list of doc IDs.
        qrels: dict mapping qid -> {pid -> relevance}.
        k: Cutoff depth.

    Returns:
        dict mapping qid -> NDCG@K score for that query.
    """
    per_query = {}
    for qid, ranked_docs in rankings.items():
        if qid not in qrels:
            continue
        dcg = 0.0
        for rank, pid in enumerate(ranked_docs[:k], start=1):
            rel = qrels[qid].get(pid, 0)
            dcg += (2**rel - 1) / np.log2(rank + 1)
        ideal_rels = sorted(qrels[qid].values(), reverse=True)[:k]
        idcg = sum((2**rel - 1) / np.log2(rank + 1) for rank, rel in enumerate(ideal_rels, 1))
        per_query[qid] = dcg / max(idcg, 1e-10)
    return per_query


def compute_rank_correlation(
    scores_a: dict[str, dict[str, float]],
    scores_b: dict[str, dict[str, float]],
) -> dict:
    """Compute Spearman's rank correlation between two scoring systems.

    Operates on per-query score dicts (all docs), not just ranked lists,
    to give a more informative correlation.

    Args:
        scores_a: dict mapping qid -> {did -> score} (e.g. FP128 scores).
        scores_b: dict mapping qid -> {did -> score} (e.g. binary scores).

    Returns:
        dict with spearman_rho (mean across queries) and n_queries.
    """
    rhos = []
    for qid in scores_a:
        if qid not in scores_b:
            continue
        common_dids = sorted(set(scores_a[qid].keys()) & set(scores_b[qid].keys()))
        if len(common_dids) < 3:
            continue
        vec_a = np.array([scores_a[qid][did] for did in common_dids])
        vec_b = np.array([scores_b[qid][did] for did in common_dids])
        if np.std(vec_a) < 1e-12 or np.std(vec_b) < 1e-12:
            continue
        rho, _ = stats.spearmanr(vec_a, vec_b)
        if not np.isnan(rho):
            rhos.append(rho)
    return {
        "spearman_rho": float(np.mean(rhos)) if rhos else 0.0,
        "n_queries": len(rhos),
    }


def evaluate_retrieval(
    model: SubBitModel,
    query_store: EmbeddingStore,
    doc_store: EmbeddingStore,
    qrels: dict[str, dict[str, int]],
    doc_ids: list[str] | None = None,
    device: torch.device = torch.device("cpu"),
    metrics: list[str] = None,
    max_queries: int = -1,
    top_k: int = 1000,
) -> dict:
    """Full evaluation pipeline: rank all docs for each query, compute metrics.

    Args:
        model: Trained SubBitModel.
        query_store: Query embedding store.
        doc_store: Document embedding store.
        qrels: Relevance judgments.
        doc_ids: List of candidate document IDs (None = all in store).
        device: Torch device.
        metrics: List of metrics to compute (e.g., ["mrr@10", "recall@1000"]).
        max_queries: Limit queries for speed.
        top_k: Maximum ranking depth.

    Returns:
        dict of metric_name → score.
    """
    if metrics is None:
        metrics = ["mrr@10", "recall@100", "recall@1000"]

    if doc_ids is None:
        doc_ids = doc_store.get_all_ids()

    query_ids = list(qrels.keys())
    if 0 < max_queries < len(query_ids):
        query_ids = query_ids[:max_queries]

    # Pre-encode all documents in a single batched matmul (no per-doc Python
    # loop). Avoids the O(N_docs) kernel-launch overhead that dominated the
    # eval pipeline on MPS at 100k corpus scale.
    logger.info(f"Pre-encoding {len(doc_ids)} documents for fast evaluation...")
    model.eval()
    model = model.to(device)

    # First pass: collect raw embeddings + lengths. Skips missing IDs.
    raw_chunks: list[torch.Tensor] = []
    lengths: list[int] = []
    valid_doc_ids: list[str] = []
    for pid in doc_ids:
        try:
            d = doc_store.get(pid)
            raw_chunks.append(d)
            lengths.append(int(d.shape[0]))
            valid_doc_ids.append(pid)
        except (KeyError, FileNotFoundError):
            pass

    if not valid_doc_ids:
        return {}

    N = len(valid_doc_ids)
    max_doc_len = max(lengths)
    input_dim = raw_chunks[0].shape[-1]

    raw_padded = torch.zeros(N, max_doc_len, input_dim, dtype=raw_chunks[0].dtype)
    for i, d in enumerate(raw_chunks):
        raw_padded[i, : d.shape[0]] = d

    # Run the model's document encoder over the padded corpus in chunks. The
    # output dim is discovered from a dummy forward so this works for any
    # projected_dim used by the paper experiments.
    encode_chunk = 4096
    with torch.no_grad():
        probe = model.encode_document(raw_padded[:1, : lengths[0]].to(device))
    out_dim = probe.shape[-1]
    out_dtype = probe.dtype

    # Store doc tensor in fp16 to halve memory + ~2x compute on M-series MPS.
    # Mathematically equivalent for sign-quantized SubBit (values are exactly
    # ±1 or small integer levels — fp16 representable losslessly). For fp32
    # outputs (e.g. r-projected oracle paths) the rounding is below MRR noise.
    eval_dtype = torch.float16 if device.type in ("mps", "cuda") else out_dtype

    # MPS rejects any single NDArray with > INT_MAX (~2.15B) total elements;
    # at r=128 the unified (N, T, r) tensor exceeds that. Fall back to a
    # chunk-list layout when needed. The list-of-chunks path is bit-identical
    # to the unified path (same fp16 values, same chunk boundaries used in
    # the scoring loop below); it just avoids the single >INT_MAX allocation.
    INT_MAX = (1 << 31) - 1
    total_elements = N * max_doc_len * out_dim
    use_chunked_storage = (
        device.type == "mps" and total_elements > INT_MAX
    )

    doc_mask = torch.zeros(N, max_doc_len, dtype=torch.bool, device=device)
    for i, n in enumerate(lengths):
        doc_mask[i, :n] = True

    doc_vecs = None
    doc_vec_chunks: list[torch.Tensor] | None = None
    chunk_offsets: list[tuple[int, int]] | None = None

    with torch.no_grad():
        if use_chunked_storage:
            doc_vec_chunks = []
            chunk_offsets = []
            for s in tqdm(range(0, N, encode_chunk), desc="Encoding docs"):
                e = min(s + encode_chunk, N)
                chunk = raw_padded[s:e].to(device)              # (C, T, in_dim)
                encoded = model.encode_document(chunk)           # (C, T, r)
                mask_chunk = doc_mask[s:e].unsqueeze(-1).to(encoded.dtype)
                doc_vec_chunks.append((encoded * mask_chunk).to(eval_dtype))
                chunk_offsets.append((s, e))
        else:
            doc_vecs = torch.zeros(N, max_doc_len, out_dim,
                                   dtype=eval_dtype, device=device)
            for s in tqdm(range(0, N, encode_chunk), desc="Encoding docs"):
                e = min(s + encode_chunk, N)
                chunk = raw_padded[s:e].to(device)              # (C, T, in_dim)
                encoded = model.encode_document(chunk)           # (C, T, r)
                # Zero-out positions that were padding so they don't contribute
                # to MaxSim sums.
                mask_chunk = doc_mask[s:e].unsqueeze(-1).to(encoded.dtype)
                doc_vecs[s:e] = (encoded * mask_chunk).to(eval_dtype)

    del raw_padded, raw_chunks
    layout = "chunked" if use_chunked_storage else "unified"
    logger.info("Padded doc tensor: (%d, %d, %d) %s [%s]",
                N, max_doc_len, out_dim, eval_dtype, layout)

    # Rank documents for each query using a per-query loop with chunked
    # doc-side sweeps. Per-query was empirically faster than batched-query
    # on MPS at 100k corpus scale: batched-query creates a (Qb, C, m, T)
    # intermediate that thrashes unified memory, while per-query keeps the
    # intermediate at (C, m, T) which fits comfortably in cache. The big win
    # over the original loop is the fp16 doc tensor (above) and the batched
    # doc encoding pre-pass.
    from .measurement import LatencyTracker, MemoryTracker

    rankings = {}
    latency = LatencyTracker()
    doc_chunk = 8192
    with MemoryTracker(device) as mem:
        for qid in tqdm(query_ids, desc="Evaluating"):
            try:
                q_embs = query_store.get(qid).to(device)
            except (KeyError, FileNotFoundError):
                continue

            with latency.measure(), torch.no_grad():
                q_encoded = model.encode_query(q_embs, symmetric=False).to(eval_dtype)

                scores_tensor = torch.empty(N, dtype=q_encoded.dtype,
                                            device=device)
                if use_chunked_storage:
                    # Iterate the chunk list; each chunk is bounded under
                    # INT_MAX so the einsum allocation is safe on MPS.
                    for chunk_tensor, (s, e) in zip(doc_vec_chunks, chunk_offsets):
                        sim = torch.einsum("md,cnd->cmn",
                                            q_encoded, chunk_tensor)
                        sim = sim.masked_fill(
                            ~doc_mask[s:e].unsqueeze(1), float('-inf'))
                        scores_tensor[s:e] = sim.max(dim=-1).values.sum(dim=-1)
                        del sim
                else:
                    num_docs = doc_vecs.shape[0]
                    for s in range(0, num_docs, doc_chunk):
                        e = min(s + doc_chunk, num_docs)
                        # (m, r) × (C, T, r) → (C, m, T) — same dtype throughout
                        sim = torch.einsum("md,cnd->cmn",
                                            q_encoded, doc_vecs[s:e])
                        sim = sim.masked_fill(
                            ~doc_mask[s:e].unsqueeze(1), float('-inf'))
                        scores_tensor[s:e] = sim.max(dim=-1).values.sum(dim=-1)
                        del sim

                actual_k = min(top_k, scores_tensor.shape[0])
                _, topk_idx = torch.topk(scores_tensor, k=actual_k)

            topk_idx_cpu = topk_idx.cpu().tolist()
            rankings[qid] = [valid_doc_ids[i] for i in topk_idx_cpu]

    # Compute metrics
    results = {}
    for metric_str in metrics:
        parts = metric_str.lower().split("@")
        metric_name = parts[0]
        k = int(parts[1]) if len(parts) > 1 else 10

        if metric_name == "mrr":
            results[metric_str] = compute_mrr(rankings, qrels, k)
        elif metric_name == "recall":
            results[metric_str] = compute_recall(rankings, qrels, k)
        elif metric_name == "ndcg":
            results[metric_str] = compute_ndcg(rankings, qrels, k)

    # Attach measurement block. Private keys (prefixed with '_') so downstream
    # metric loggers ignore them; the Trainer pops them before logging.
    from dataclasses import asdict
    results["_measurement"] = {
        "latency": asdict(latency.stats()),
        "memory": asdict(mem.snapshot()),
        "num_queries": len(rankings),
        "num_docs": len(valid_doc_ids),
        "top_k": top_k,
    }

    loggable = {k: v for k, v in results.items() if not k.startswith("_")}
    logger.info(f"Evaluation results: {loggable}")
    return results


# ---------------------------------------------------------------------------
# Reranking pipeline (dataset-agnostic)
# ---------------------------------------------------------------------------


def score_document(
    model: SubBitModel,
    q_encoded: torch.Tensor,
    doc_embs: torch.Tensor,
    scoring_mode: str = "binary",
) -> float:
    """Score a single document against an already-encoded query.

    Args:
        model: Trained SubBitModel.
        q_encoded: Projected query tensor (from ``model.encode_query``).
        doc_embs: Raw document token embeddings (num_tokens, input_dim).
        scoring_mode: ``"binary"`` (default) or ``"rabitq"``.

    Returns:
        Scalar similarity score.
    """
    from .scoring import maxsim_rabitq  # avoid circular at top level

    if scoring_mode == "binary":
        d_encoded = model.encode_document(doc_embs)
        return float(maxsim(q_encoded, d_encoded).item())
    if scoring_mode == "rabitq":
        d_encoded = model.encode_document_rabitq(doc_embs)
        return float(
            maxsim_rabitq(
                q_encoded,
                d_encoded["binary"],
                d_encoded["norm"],
                d_encoded["vdot"],
            ).item()
        )
    raise ValueError(f"Unsupported scoring mode: {scoring_mode}")


def rerank_candidates(
    model: SubBitModel,
    query_embs: dict[str, torch.Tensor],
    doc_embs: dict[str, torch.Tensor],
    qrels: dict[str, dict[str, int]],
    candidates: dict[str, list[str]],
    *,
    scoring_mode: str = "binary",
    device: torch.device | str = "cpu",
    top_k: int = 1000,
    max_queries: int = -1,
) -> tuple[dict[str, list[str]], list[tuple[str, str, int, float]], dict[str, int]]:
    """Rerank candidate documents for each query.

    This is the generic reranking function used by both MS MARCO and BEIR
    evaluation scripts.  It accepts arbitrary embeddings and candidates and
    is completely dataset-agnostic.

    Args:
        model: Trained SubBitModel.
        query_embs: {qid: (num_tokens, input_dim)} query embeddings.
        doc_embs: {pid: (num_tokens, input_dim)} document embeddings.
        qrels: {qid: {pid: relevance}} ground truth.
        candidates: {qid: [pid, ...]} candidate documents to score.
        scoring_mode: ``"binary"`` or ``"rabitq"``.
        device: Torch device.
        top_k: Return top-K ranked documents per query.
        max_queries: Limit number of queries (-1 = all).

    Returns:
        Tuple of (rankings, run_rows, stats).
    """
    device = torch.device(device) if isinstance(device, str) else device
    query_ids = [qid for qid in qrels if qid in query_embs and qid in candidates]
    if 0 < max_queries < len(query_ids):
        query_ids = query_ids[:max_queries]

    rankings: dict[str, list[str]] = {}
    run_rows: list[tuple[str, str, int, float]] = []
    stats = {
        "queries_evaluated": 0,
        "queries_missing_candidates": len(
            [qid for qid in qrels if qid in query_embs and qid not in candidates]
        ),
        "queries_missing_embeddings": len(
            [qid for qid in qrels if qid not in query_embs]
        ),
        "candidate_docs_scored": 0,
        "candidate_docs_missing_embeddings": 0,
    }

    model.eval()
    model = model.to(device)

    with torch.no_grad():
        for qid in tqdm(query_ids, desc="Reranking candidates"):
            q_encoded = model.encode_query(query_embs[qid].to(device))
            scores = []
            for did in candidates[qid]:
                d_emb = doc_embs.get(did)
                if d_emb is None:
                    stats["candidate_docs_missing_embeddings"] += 1
                    continue
                score = score_document(model, q_encoded, d_emb.to(device), scoring_mode)
                scores.append((did, score))

            scores.sort(key=lambda item: item[1], reverse=True)
            top_scores = scores[:top_k]
            rankings[qid] = [did for did, _ in top_scores]
            for rank, (did, score) in enumerate(top_scores, start=1):
                run_rows.append((qid, did, rank, score))

            stats["queries_evaluated"] += 1
            stats["candidate_docs_scored"] += len(scores)

    return rankings, run_rows, stats
