"""MaxSim scoring for multi-vector retrieval.

Implements the core scoring primitives for the paper:
  - Full-precision MaxSim (ground truth)
  - Asymmetric MaxSim (float query × binary doc)
  - RaBitQ MaxSim with correction factors
  - Batched/scoring helpers
"""
from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core MaxSim Scoring
# ---------------------------------------------------------------------------


def maxsim(
    query_vecs: torch.Tensor,
    doc_vecs: torch.Tensor,
    query_mask: Optional[torch.Tensor] = None,
    doc_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute MaxSim score between query and document token embeddings.

    MaxSim = Σ_i max_j (q_i · d_j)  for each query token i over all doc tokens j.

    Args:
        query_vecs: (m, dim) or (batch, m, dim) query token embeddings.
        doc_vecs: (n, dim) or (batch, n, dim) document token embeddings.
        query_mask: Optional (m,) or (batch, m) bool mask for valid query tokens.
        doc_mask: Optional (n,) or (batch, n) bool mask for valid doc tokens.

    Returns:
        score: Scalar or (batch,) MaxSim scores.
    """
    is_batched = query_vecs.dim() == 3

    if is_batched:
        # (batch, m, n) similarity matrix
        sim_matrix = torch.bmm(query_vecs, doc_vecs.transpose(1, 2))
    else:
        # (m, n) similarity matrix
        sim_matrix = query_vecs @ doc_vecs.T

    # Mask out invalid document tokens (pad positions)
    if doc_mask is not None:
        if is_batched:
            # doc_mask: (batch, n) → (batch, 1, n)
            sim_matrix = sim_matrix.masked_fill(~doc_mask.unsqueeze(1), float("-inf"))
        else:
            sim_matrix = sim_matrix.masked_fill(~doc_mask.unsqueeze(0), float("-inf"))

    # Max over doc tokens for each query token
    max_sim_per_query = sim_matrix.max(dim=-1).values  # (m,) or (batch, m)

    # Mask out invalid query tokens
    if query_mask is not None:
        max_sim_per_query = max_sim_per_query * query_mask.float()

    # Sum over query tokens
    if is_batched:
        return max_sim_per_query.sum(dim=-1)  # (batch,)
    return max_sim_per_query.sum()  # scalar


def maxsim_rabitq(
    query_vecs: torch.Tensor,
    doc_binary: torch.Tensor,
    doc_norm: torch.Tensor,
    doc_vdot: torch.Tensor,
    query_mask: Optional[torch.Tensor] = None,
    doc_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """MaxSim with RaBitQ correction factors for unbiased distance estimation.

    Standard asymmetric MaxSim computes: sum_i max_j (Rq_i · sign(Rd_j))
    This is biased because sign() loses magnitude information.

    RaBitQ correction: for each doc token j, we stored:
      - norm_j  = ||Rd_j||
      - vdot_j  = (Rd_j / ||Rd_j||) · sign(Rd_j)

    The corrected similarity between query token i and doc token j:
      sim(i,j) = (Rq_i · sign(Rd_j)) * norm_j / vdot_j

    This recovers an unbiased estimate of Rq_i · Rd_j.

    Args:
        query_vecs: (m, r) float query embeddings in projected space.
        doc_binary: (n, r) binary doc embeddings in {-1, +1}.
        doc_norm:   (n,) norms of projected doc embeddings before sign.
        doc_vdot:   (n,) cosine between projected and quantized doc embeddings.
        query_mask: Optional (m,) bool mask for valid query tokens.
        doc_mask:   Optional (n,) bool mask for valid doc tokens.

    Returns:
        score: Scalar MaxSim score with RaBitQ corrections.
    """
    # Raw binary dot product: (m, n)
    sim_matrix = query_vecs @ doc_binary.T

    # RaBitQ correction: multiply each column j by norm_j / vdot_j
    correction = doc_norm / (doc_vdot + 1e-10)  # (n,)
    sim_matrix = sim_matrix * correction.unsqueeze(0)  # (m, n)

    # Mask invalid doc tokens
    if doc_mask is not None:
        sim_matrix = sim_matrix.masked_fill(~doc_mask.unsqueeze(0), float("-inf"))

    # Max over doc tokens per query token
    max_sim_per_query = sim_matrix.max(dim=-1).values  # (m,)

    # Mask invalid query tokens
    if query_mask is not None:
        max_sim_per_query = max_sim_per_query * query_mask.float()

    return max_sim_per_query.sum()


def batched_maxsim(
    query_vecs: torch.Tensor,
    doc_vecs: torch.Tensor,
    query_mask: Optional[torch.Tensor] = None,
    doc_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute MaxSim scores for a batch of queries against a batch of documents.

    Each query is scored against ALL documents (cross-scoring).

    Args:
        query_vecs: (num_queries, m, dim) query embeddings.
        doc_vecs: (num_docs, n, dim) document embeddings.
        query_mask: Optional (num_queries, m) masks.
        doc_mask: Optional (num_docs, n) masks.

    Returns:
        scores: (num_queries, num_docs) MaxSim scores.
    """
    num_queries = query_vecs.shape[0]
    num_docs = doc_vecs.shape[0]

    scores = torch.zeros(num_queries, num_docs, device=query_vecs.device)

    for i in range(num_queries):
        q = query_vecs[i]  # (m, dim)
        q_mask = query_mask[i] if query_mask is not None else None

        for j in range(num_docs):
            d = doc_vecs[j]  # (n, dim)
            d_mask = doc_mask[j] if doc_mask is not None else None
            scores[i, j] = maxsim(q, d, q_mask, d_mask)

    return scores


def efficient_batched_maxsim(
    query_vecs: torch.Tensor,
    doc_vecs: torch.Tensor,
    query_mask: Optional[torch.Tensor] = None,
    doc_mask: Optional[torch.Tensor] = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    """Memory-efficient batched MaxSim using chunked matrix multiplication.

    Scores each query against all documents using chunked ops to avoid
    materializing the full (num_queries × m × num_docs × n) tensor.

    Args:
        query_vecs: (num_queries, m, dim).
        doc_vecs: (num_docs, n, dim).
        query_mask: Optional (num_queries, m).
        doc_mask: Optional (num_docs, n).
        chunk_size: Number of documents to score per chunk.

    Returns:
        scores: (num_queries, num_docs).
    """
    num_queries, m, dim = query_vecs.shape
    num_docs, n, _ = doc_vecs.shape

    all_scores = []

    for chunk_start in range(0, num_docs, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num_docs)
        doc_chunk = doc_vecs[chunk_start:chunk_end]  # (chunk, n, dim)
        chunk_mask = doc_mask[chunk_start:chunk_end] if doc_mask is not None else None

        # (num_queries, m, dim) × (chunk, n, dim) → (num_queries, chunk, m, n)
        # sim[q, c, m, n] = query_vecs[q, m, :] · doc_chunk[c, n, :]
        sim = torch.einsum("qmd,cnd->qcmn", query_vecs, doc_chunk)
        # sim: (num_queries, chunk, m, n)

        # Mask invalid doc tokens
        if chunk_mask is not None:
            # chunk_mask: (chunk, n) → (1, chunk, 1, n)
            sim = sim.masked_fill(~chunk_mask[None, :, None, :], float("-inf"))

        # Max over doc tokens
        max_per_query_token = sim.max(dim=-1).values  # (num_queries, chunk, m)

        # Mask invalid query tokens
        if query_mask is not None:
            max_per_query_token = max_per_query_token * query_mask[:, None, :].float()

        # Sum over query tokens
        chunk_scores = max_per_query_token.sum(dim=-1)  # (num_queries, chunk)
        all_scores.append(chunk_scores)

    return torch.cat(all_scores, dim=1)  # (num_queries, num_docs)


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------


def compute_storage_bytes(
    num_tokens: int, projected_dim: int, include_doc_lens: bool = True,
    input_dim: int = 128, num_docs: int = 8_800_000,
) -> dict:
    """Compute storage requirements for a compressed corpus.

    Args:
        num_tokens: Total number of document tokens in the corpus.
        projected_dim: Subspace dimension r.
        include_doc_lens: Include storage for document length offsets.
        input_dim: Original embedding dimension (128 for ColBERTv2).
        num_docs: Number of documents (for doc-length offset overhead).

    Returns:
        dict with storage breakdowns in bytes, MB, and GB.
    """
    token_bytes = num_tokens * (projected_dim // 8)  # Packed binary tokens

    result = {
        "token_storage_bytes": token_bytes,
        "token_storage_mb": token_bytes / (1024**2),
        "token_storage_gb": token_bytes / (1024**3),
        "bytes_per_token": projected_dim // 8,
        "bits_per_dim": projected_dim / input_dim,
    }

    if include_doc_lens:
        doc_overhead = num_docs * 4  # 4 bytes per doc offset
        result["doc_overhead_bytes"] = doc_overhead
        result["total_bytes"] = token_bytes + doc_overhead
        result["total_gb"] = (token_bytes + doc_overhead) / (1024**3)

    return result
