"""Baseline compression methods for comparison.

All baselines take full-precision ColBERT embeddings and produce compressed
representations. These are NOT learned — they serve as reference points
on the Pareto frontier.

Baselines:
  1. Random projection + binary: Random orthogonal R + sign
  2. PCA projection + binary:    PCA R (no fine-tuning) + sign
  3. Identity truncation:        First r dims + sign (Matryoshka-style)
  4. Product quantization (PQ):  Pure-PyTorch ProductQuantizer (sub-vector k-means)
"""

import logging
from typing import Optional

import numpy as np
import torch
from sklearn.decomposition import PCA

from .scoring import maxsim

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Baseline encoding functions
# ---------------------------------------------------------------------------


def random_projection_binary(
    embeddings: torch.Tensor,
    projected_dim: int,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random orthogonal projection + binarization.

    Uses a random orthonormal R (no learning). Serves as the
    "how much does learning buy you?" baseline.

    Args:
        embeddings: (..., d) float tensor.
        projected_dim: Target dimension r.
        seed: Random seed for reproducibility.

    Returns:
        binary: (..., r) tensor with values in {-1, +1}.
        R: (r, d) projection matrix (for query-side use).
    """
    d = embeddings.shape[-1]
    torch.manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(d, projected_dim, device=embeddings.device))
    R = Q.T  # (r, d)

    projected = embeddings @ R.T  # (..., r)
    result = torch.sign(projected)
    result[result == 0] = 1.0

    return result, R


def pca_projection_binary(
    embeddings: torch.Tensor,
    projected_dim: int,
    fit_data: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PCA projection + binarization (no fine-tuning).

    Uses PCA to find the top-r variance directions, projects, then
    binarizes. This is the "PCA init without training" baseline.

    Args:
        embeddings: (..., d) float tensor to encode.
        projected_dim: Target dimension r.
        fit_data: (N, d) data to fit PCA on. If None, uses embeddings.

    Returns:
        binary: (..., r) tensor with values in {-1, +1}.
        R: (r, d) PCA projection matrix.
        mean: (d,) PCA mean — subtract from query before `@ R.T` so the query is
            centered identically (= sklearn pca.transform). Projecting without it
            is a fairness bug that handicaps PCA at low r.
    """
    original_shape = embeddings.shape
    d = embeddings.shape[-1]

    if fit_data is None:
        fit_data = embeddings.reshape(-1, d)
    else:
        fit_data = fit_data.reshape(-1, d)

    pca = PCA(n_components=projected_dim)
    pca.fit(fit_data.cpu().numpy())
    R = torch.tensor(pca.components_, dtype=torch.float32, device=embeddings.device)
    mean = torch.tensor(pca.mean_, dtype=torch.float32, device=embeddings.device)

    projected = (embeddings - mean) @ R.T  # (..., r); center = sklearn pca.transform
    result = torch.sign(projected)
    result[result == 0] = 1.0

    return result, R, mean


def identity_truncation(
    embeddings: torch.Tensor,
    projected_dim: int,
) -> torch.Tensor:
    """Identity truncation: keep first r dimensions, then binarize.

    Matryoshka-style baseline — assumes early dimensions are most important.

    Args:
        embeddings: (..., d) float tensor.
        projected_dim: Number of leading dimensions to keep.

    Returns:
        binary: (..., projected_dim) tensor with values in {-1, +1}.
    """
    truncated = embeddings[..., :projected_dim]
    result = torch.sign(truncated)
    result[result == 0] = 1.0
    return result


# ---------------------------------------------------------------------------
# Product Quantization (PQ) baseline
# ---------------------------------------------------------------------------


class TorchProductQuantizer:
    """Pure-PyTorch Product Quantizer (no FAISS dependency).

    Splits d-dim vectors into m sub-vectors of size ds = d/m,
    learns k centroids per sub-space via k-means, and encodes
    each sub-vector as its nearest centroid index.

    Attributes:
        d: Input dimensionality.
        m: Number of sub-quantizers.
        k: Number of centroids per sub-quantizer (2^n_bits).
        ds: Sub-vector dimensionality (d // m).
        codebooks: (m, k, ds) centroid tensor.
    """

    def __init__(self, d: int, m: int, n_bits: int = 8):
        assert d % m == 0, f"d={d} must be divisible by m={m}"
        self.d = d
        self.m = m
        self.k = 2 ** n_bits
        self.ds = d // m
        self.codebooks: Optional[torch.Tensor] = None  # (m, k, ds)

    def train(self, x: torch.Tensor, n_iter: int = 20) -> None:
        """Train codebooks via k-means on each sub-space.

        Args:
            x: (N, d) training vectors.
            n_iter: Number of k-means iterations.
        """
        N = x.shape[0]
        subs = x.reshape(N, self.m, self.ds)  # (N, m, ds)
        codebooks = torch.zeros(self.m, self.k, self.ds, dtype=x.dtype)

        for j in range(self.m):
            sub_j = subs[:, j, :]  # (N, ds)
            # Init centroids with random data points
            perm = torch.randperm(N)[: self.k]
            centroids = sub_j[perm].clone()  # (k, ds)

            for _ in range(n_iter):
                # Assign each point to nearest centroid
                dists = torch.cdist(sub_j, centroids)  # (N, k)
                labels = dists.argmin(dim=1)  # (N,)

                # Update centroids
                for c in range(self.k):
                    mask = labels == c
                    if mask.any():
                        centroids[c] = sub_j[mask].mean(dim=0)

            codebooks[j] = centroids

        self.codebooks = codebooks

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode vectors to centroid indices.

        Args:
            x: (N, d) input vectors.

        Returns:
            codes: (N, m) long tensor of centroid indices.
        """
        assert self.codebooks is not None, "Call train() first"
        N = x.shape[0]
        subs = x.reshape(N, self.m, self.ds)  # (N, m, ds)
        codes = torch.zeros(N, self.m, dtype=torch.long, device=x.device)
        
        codebooks = self.codebooks.to(x.device)

        for j in range(self.m):
            dists = torch.cdist(subs[:, j, :], codebooks[j])  # (N, k)
            codes[:, j] = dists.argmin(dim=1)

        return codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode centroid indices back to reconstructed vectors.

        Args:
            codes: (N, m) long tensor of centroid indices.

        Returns:
            reconstructed: (N, d) float tensor.
        """
        assert self.codebooks is not None, "Call train() first"
        N = codes.shape[0]
        parts = []
        codebooks = self.codebooks.to(codes.device)
        for j in range(self.m):
            parts.append(codebooks[j][codes[:, j]])  # (N, ds)
        return torch.cat(parts, dim=1)  # (N, d)


def product_quantize(
    embeddings: torch.Tensor,
    n_subquantizers: int = 8,
    n_bits: int = 8,
    fit_data: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, TorchProductQuantizer]:
    """Product Quantization (pure PyTorch, no FAISS dependency).

    Splits each d-dimensional vector into n_subquantizers sub-vectors
    and learns 2^n_bits centroids per sub-space via k-means.

    Storage: n_subquantizers * n_bits / 8 bytes per vector.
    With defaults (8 sub-q, 8 bits): 8 bytes/token.

    Args:
        embeddings: (..., d) float tensor to encode.
        n_subquantizers: Number of sub-vector partitions (d must be divisible).
        n_bits: Bits per sub-quantizer code (8 → 256 centroids).
        fit_data: (N, d) data to train codebook on. If None, uses embeddings.

    Returns:
        codes: (N, n_subquantizers) long tensor of centroid indices.
        pq: Trained TorchProductQuantizer for decode.
    """
    d = embeddings.shape[-1]
    flat = embeddings.reshape(-1, d).contiguous()

    if fit_data is None:
        train_data = flat
    else:
        train_data = fit_data.reshape(-1, d).contiguous()

    pq = TorchProductQuantizer(d, n_subquantizers, n_bits)
    pq.train(train_data)

    codes = pq.encode(flat)
    return codes, pq


def pq_decode(
    codes: torch.Tensor,
    pq: TorchProductQuantizer,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Decode PQ codes back to reconstructed float vectors.

    Args:
        codes: (N, n_subquantizers) centroid indices.
        pq: Trained TorchProductQuantizer.
        device: Target torch device.

    Returns:
        reconstructed: (N, d) float tensor.
    """
    return pq.decode(codes).to(device)


def pq_lookup_tables(
    query_embeddings: torch.Tensor,
    pq: TorchProductQuantizer,
) -> torch.Tensor:
    """Build asymmetric distance-computation lookup tables for a query.

    For each query token and each PQ sub-quantizer, precompute inner products
    against every centroid so document scoring can gather by code index without
    reconstructing float document vectors.

    Args:
        query_embeddings: (m, d) float query token embeddings.
        pq: Trained TorchProductQuantizer.

    Returns:
        tables: (m, pq.m, pq.k) lookup table tensor.
    """
    if query_embeddings.dim() != 2:
        raise ValueError(f"query_embeddings must have shape (m, d), got {tuple(query_embeddings.shape)}")

    device = query_embeddings.device
    codebooks = pq.codebooks.to(device)
    q_sub = query_embeddings.reshape(query_embeddings.shape[0], pq.m, pq.ds)
    return torch.einsum("qmd,mkd->qmk", q_sub, codebooks)


def pq_maxsim(
    query_embeddings: torch.Tensor,
    doc_codes: torch.Tensor,
    pq: TorchProductQuantizer,
    query_mask: Optional[torch.Tensor] = None,
    doc_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute asymmetric PQ MaxSim without reconstructing doc vectors.

    This is the true ADC baseline: float query sub-vectors probe centroid
    lookup tables, while documents stay in compact code form.
    """
    if query_embeddings.dim() != 2:
        raise ValueError(f"query_embeddings must have shape (m, d), got {tuple(query_embeddings.shape)}")
    if doc_codes.dim() != 2:
        raise ValueError(f"doc_codes must have shape (n, m_sub), got {tuple(doc_codes.shape)}")

    tables = pq_lookup_tables(query_embeddings, pq)  # (m_query, m_sub, k)
    codes = doc_codes.to(device=query_embeddings.device, dtype=torch.long)
    sim_matrix = torch.zeros(query_embeddings.shape[0], doc_codes.shape[0], device=query_embeddings.device)

    for sub_idx in range(pq.m):
        sim_matrix += tables[:, sub_idx][:, codes[:, sub_idx]]

    if doc_mask is not None:
        sim_matrix = sim_matrix.masked_fill(~doc_mask.unsqueeze(0), float("-inf"))

    max_sim_per_query = sim_matrix.max(dim=-1).values

    if query_mask is not None:
        max_sim_per_query = max_sim_per_query * query_mask.float()

    return max_sim_per_query.sum()



# ---------------------------------------------------------------------------
# Baseline MaxSim scoring
# ---------------------------------------------------------------------------


def score_with_projection(
    query_embs: torch.Tensor,
    doc_binary: torch.Tensor,
    R: torch.Tensor,
    symmetric: bool = False,
    mean: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Score using projection + binary document against query.

    Asymmetric (default): float query in projected space vs binary doc.
    Symmetric: binary query vs binary doc (Hamming in projected space).

    Args:
        query_embs: (m, d) float query embeddings.
        doc_binary: (n, r) binary document embeddings.
        R: (r, d) projection matrix.
        symmetric: If True, also binarize query.

    Returns:
        score: Scalar MaxSim score.
    """
    # For PCA, pass mean so the query is centered identically to the doc
    # ((x - mean) @ R.T = pca.transform). One-sided centering badly hurts MRR.
    qc = query_embs - mean if mean is not None else query_embs
    q_proj = qc @ R.T  # (m, r)
    if symmetric:
        q_proj = torch.sign(q_proj)
        q_proj[q_proj == 0] = 1.0
    return maxsim(q_proj, doc_binary)

