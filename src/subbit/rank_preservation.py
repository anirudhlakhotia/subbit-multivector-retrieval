"""Structure-preservation diagnostics for SubBit vs. fp32 MaxSim.

Answers the question *"what does the learned projection preserve?"* with three
complementary views over per-query (doc → score) vectors:

  1. **Spearman ρ** — monotonic agreement of raw scores.
  2. **Kendall τ** — pairwise ordering agreement (robust to ties, conservative).
  3. **Top-k overlap** — |A_k ∩ B_k| / k, the operational ranking-quality view.

Used as a workshop-paper diagnostic: empirical rank preservation is a stronger
signal of similarity-structure retention than error bounds alone.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

import numpy as np
import torch
from scipy import stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pairwise score computation
# ---------------------------------------------------------------------------


@dataclass
class PairwiseScores:
    """Per-(query, doc) score matrices for two scoring functions.

    Attributes:
        fp32:     (n_queries, n_docs) MaxSim under fp32 ColBERT scoring.
        subbit:   (n_queries, n_docs) MaxSim under SubBit (projected-binary) scoring.
        query_ids: ordered query IDs matching axis 0.
        doc_ids:   ordered doc IDs matching axis 1.
    """

    fp32: torch.Tensor
    subbit: torch.Tensor
    query_ids: list[str]
    doc_ids: list[str]


@torch.no_grad()
def compute_pairwise_scores(
    query_store,
    doc_store,
    query_ids: list[str],
    doc_ids: list[str],
    encode_query: Callable[[torch.Tensor], torch.Tensor],
    encode_document: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device | str = "cpu",
    progress: bool = True,
    doc_chunk: int = 4096,
) -> PairwiseScores:
    """Compute fp32 and SubBit MaxSim scores for every (q, d) pair.

    Pre-encodes and pads all doc tokens (raw fp32 + encoded SubBit) in
    **doc-chunks** of at most ``doc_chunk`` docs, to keep any single padded
    tensor below MPS's ``INT_MAX`` dimension-length limit — a 100k-doc
    corpus at max_len=168 × d=128 already exceeds that limit for the raw
    padded tensor. Scores are gathered per query by iterating through the
    chunks.

    Args:
        query_store: object with ``.get(qid) -> (m, 128) fp32``.
        doc_store:   object with ``.get(did) -> (n, 128) fp32``.
        query_ids:   queries to evaluate.
        doc_ids:     docs to score against.
        encode_query: SubBit query encoder, ``(m, 128) -> (m, r)`` float.
        encode_document: SubBit doc encoder, ``(n, 128) -> (n, r)`` ±1.
        device:     where to run the einsums.
        progress:   show a tqdm bar over queries.
        doc_chunk:  max docs per padded chunk (default 4096).
    """
    device = torch.device(device)

    raw_docs: list[torch.Tensor] = []
    sub_docs: list[torch.Tensor] = []
    kept_doc_ids: list[str] = []
    for did in doc_ids:
        try:
            emb = doc_store.get(did).to(device)
        except (KeyError, FileNotFoundError):
            continue
        raw_docs.append(emb)
        sub_docs.append(encode_document(emb).detach())
        kept_doc_ids.append(did)

    if not raw_docs:
        raise ValueError("No documents could be loaded for pairwise scoring")

    n_docs = len(raw_docs)
    d_raw = raw_docs[0].shape[-1]
    d_sub = sub_docs[0].shape[-1]

    # Build per-chunk padded tensors once; reuse them across queries.
    chunks: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for cstart in range(0, n_docs, doc_chunk):
        cend = min(cstart + doc_chunk, n_docs)
        segment_raw = raw_docs[cstart:cend]
        segment_sub = sub_docs[cstart:cend]
        seg_n = len(segment_raw)
        seg_max = max(t.shape[0] for t in segment_raw)

        raw_pad = torch.zeros(seg_n, seg_max, d_raw, device=device)
        sub_pad = torch.zeros(seg_n, seg_max, d_sub, device=device)
        seg_mask = torch.zeros(seg_n, seg_max, dtype=torch.bool, device=device)
        for i, (r, s) in enumerate(zip(segment_raw, segment_sub)):
            n = r.shape[0]
            raw_pad[i, :n] = r
            sub_pad[i, :n] = s
            seg_mask[i, :n] = True
        chunks.append((raw_pad, sub_pad, seg_mask))

    try:
        from tqdm import tqdm

        iterator = tqdm(query_ids, desc="pairwise scores", leave=False) if progress else query_ids
    except ImportError:
        iterator = query_ids

    fp32_rows: list[torch.Tensor] = []
    sub_rows: list[torch.Tensor] = []
    kept_query_ids: list[str] = []
    for qid in iterator:
        try:
            q_raw = query_store.get(qid).to(device)
        except (KeyError, FileNotFoundError):
            continue
        q_sub = encode_query(q_raw)

        fp_parts: list[torch.Tensor] = []
        sb_parts: list[torch.Tensor] = []
        for raw_pad, sub_pad, seg_mask in chunks:
            sim_fp = torch.einsum("md,cnd->cmn", q_raw, raw_pad)
            sim_fp = sim_fp.masked_fill(~seg_mask.unsqueeze(1), float("-inf"))
            fp_parts.append(sim_fp.max(dim=-1).values.sum(dim=-1).cpu())

            sim_sb = torch.einsum("md,cnd->cmn", q_sub, sub_pad)
            sim_sb = sim_sb.masked_fill(~seg_mask.unsqueeze(1), float("-inf"))
            sb_parts.append(sim_sb.max(dim=-1).values.sum(dim=-1).cpu())

        fp32_rows.append(torch.cat(fp_parts, dim=0))
        sub_rows.append(torch.cat(sb_parts, dim=0))
        kept_query_ids.append(qid)

    return PairwiseScores(
        fp32=torch.stack(fp32_rows, dim=0),
        subbit=torch.stack(sub_rows, dim=0),
        query_ids=kept_query_ids,
        doc_ids=kept_doc_ids,
    )


# ---------------------------------------------------------------------------
# Rank-preservation metrics
# ---------------------------------------------------------------------------


@dataclass
class RankPreservationReport:
    """Aggregate + distributional rank-preservation statistics.

    Per-query metrics are averaged (and reported with std / median) across
    queries. Top-k overlap is reported per k.
    """

    spearman_rho: dict[str, float]  # {mean, std, median, n}
    kendall_tau: dict[str, float]
    topk_overlap: dict[int, dict[str, float]]  # {k: {mean, std, median, n}}
    score_correlation_pearson: float  # flattened across all (q, d) pairs
    n_queries: int
    n_docs: int
    per_query: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "spearman_rho": self.spearman_rho,
            "kendall_tau": self.kendall_tau,
            "topk_overlap": {str(k): v for k, v in self.topk_overlap.items()},
            "score_correlation_pearson": self.score_correlation_pearson,
            "n_queries": self.n_queries,
            "n_docs": self.n_docs,
            "per_query": self.per_query,
        }


def _summarise(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n": int(arr.size),
    }


def topk_overlap(
    a: np.ndarray,
    b: np.ndarray,
    k: int,
) -> float:
    """Overlap |top-k(a) ∩ top-k(b)| / k for 1-D score vectors of equal length.

    When k >= len(a), both sets are the full universe and overlap is 1.0.
    Ties are broken by numpy's argpartition (arbitrary but deterministic).
    """
    n = a.shape[0]
    if k <= 0:
        return 0.0
    k_eff = min(k, n)
    if k_eff == n:
        return 1.0
    top_a = set(np.argpartition(-a, k_eff - 1)[:k_eff].tolist())
    top_b = set(np.argpartition(-b, k_eff - 1)[:k_eff].tolist())
    return len(top_a & top_b) / k_eff


def compute_rank_preservation(
    scores: PairwiseScores,
    k_values: Iterable[int] = (10, 100, 1000),
    keep_per_query: bool = False,
    min_docs: int = 3,
    tol: float = 1e-12,
) -> RankPreservationReport:
    """Summarise how SubBit MaxSim scores track fp32 MaxSim scores.

    Per-query:
      - Spearman ρ  — monotonic agreement over the full doc universe.
      - Kendall τ   — pairwise ordering agreement.
      - Top-k overlap for each k in ``k_values``.

    Globally:
      - Pearson correlation across the flattened (Q × D) matrix — a coarse
        score-calibration signal.

    Queries with near-constant scores (std < ``tol``) or fewer than
    ``min_docs`` docs are skipped, consistent with scipy convention.
    """
    a = scores.fp32.numpy().astype(np.float64)
    b = scores.subbit.numpy().astype(np.float64)
    if a.shape != b.shape:
        raise ValueError(f"fp32 {a.shape} and subbit {b.shape} score shapes disagree")

    n_queries, n_docs = a.shape
    k_values = sorted({int(k) for k in k_values})

    rhos: list[float] = []
    taus: list[float] = []
    overlaps: dict[int, list[float]] = {k: [] for k in k_values}
    per_query: dict[str, dict] = {}

    for i, qid in enumerate(scores.query_ids):
        va, vb = a[i], b[i]
        if va.shape[0] < min_docs:
            continue
        if va.std() < tol or vb.std() < tol:
            continue

        rho, _ = stats.spearmanr(va, vb)
        tau, _ = stats.kendalltau(va, vb)
        if np.isnan(rho) or np.isnan(tau):
            continue

        rhos.append(float(rho))
        taus.append(float(tau))
        ovs: dict[int, float] = {}
        for k in k_values:
            ov = topk_overlap(va, vb, k)
            overlaps[k].append(ov)
            ovs[k] = ov

        if keep_per_query:
            per_query[qid] = {
                "spearman_rho": float(rho),
                "kendall_tau": float(tau),
                "topk_overlap": {str(k): float(v) for k, v in ovs.items()},
            }

    # Global score calibration: Pearson over all (q, d) pairs.
    flat_a = a.reshape(-1)
    flat_b = b.reshape(-1)
    if flat_a.std() < tol or flat_b.std() < tol:
        global_pearson = 0.0
    else:
        global_pearson = float(np.corrcoef(flat_a, flat_b)[0, 1])

    return RankPreservationReport(
        spearman_rho=_summarise(rhos),
        kendall_tau=_summarise(taus),
        topk_overlap={k: _summarise(overlaps[k]) for k in k_values},
        score_correlation_pearson=global_pearson,
        n_queries=n_queries,
        n_docs=n_docs,
        per_query=per_query,
    )


__all__ = [
    "PairwiseScores",
    "RankPreservationReport",
    "compute_pairwise_scores",
    "compute_rank_preservation",
    "topk_overlap",
]
