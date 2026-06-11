"""PLAID-style baseline: centroid + per-dim residual quantization.

Faithful re-implementation of the compression scheme from Santhanam et al.,
"PLAID: An Efficient Engine for Late Interaction Retrieval" (SIGIR 2022), built
on the residual codec of ColBERTv2 (Santhanam et al., NAACL 2022). Written from
the algorithm description in those papers — not a port of the official code —
so results can be attributed as an independent re-implementation.

Storage per doc token:
  - centroid_id:  ceil(log2(C)) bits, where C = n_centroids
  - residual:     d * b bits,         where b = residual_bits (per-dim bucket idx)

Default config (C=65536, b=2, d=128):
  16 + 256 = 272 bits = 34 bytes/token (~15× compression vs FP32).

Other canonical configs (keep for sweeps):
  b=1, C=65536: 16 + 128 = 144 bits = 18 bytes/token
  b=4, C=65536: 16 + 512 = 528 bits = 66 bytes/token

Scoring path
------------
`encode_decode` reconstructs float tokens (centroid + decoded residual); the
existing MaxSim path scores those directly. This is quality-identical to true
ADC scoring (which uses per-query lookup tables); speed benchmarks live in a
separate code path.

Isolation
---------
Self-contained — does not import from `src/subbit/baselines.py`. Pulled into
`evaluation/run_baseline_comparison.py` behind a `--include-plaid` flag so cached
baseline JSONs stay valid.
"""
from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PLAIDConfig:
    """Knobs for the PLAID-style compressor.

    Attributes:
        n_centroids:    C — number of k-means centroids (paper: 2^18 for full
                        MS MARCO; scale down for smaller corpora).
        residual_bits:  b — per-dim bucket-index width for the residual code.
                        Paper ships b ∈ {1, 2, 4}; b=2 is the ColBERTv2 default.
        kmeans_iters:   number of Lloyd iterations for centroid fitting.
        kmeans_sample:  cap on tokens used for centroid + bucket fitting.
        kmeans_chunk:   chunk size for the assignment step (controls peak mem).
        seed:           RNG seed for centroid init + sample draws.
    """

    n_centroids: int = 65536
    residual_bits: int = 2
    kmeans_iters: int = 20
    kmeans_sample: int = 1_000_000
    kmeans_chunk: int = 1024
    seed: int = 42

    @property
    def centroid_bits(self) -> int:
        return max(1, math.ceil(math.log2(max(2, self.n_centroids))))

    def bytes_per_token(self, d: int) -> float:
        """Exact bytes/token (fractional; not rounded up)."""
        return (self.centroid_bits + d * self.residual_bits) / 8.0


class PLAIDQuantizer:
    """Centroid + per-dim quantized residuals (PLAID / ColBERTv2-style).

    Usage:
        pq = PLAIDQuantizer(d=128, config=PLAIDConfig())
        pq.train(sampled_tokens)            # fits centroids + bucket boundaries
        ids, resid = pq.encode(x)           # lossy compression
        x_hat = pq.decode(ids, resid)       # reconstruction (for scoring)
        # or equivalently:
        x_hat = pq.encode_decode(x)
    """

    def __init__(self, d: int, config: Optional[PLAIDConfig] = None):
        self.d = d
        self.config = config or PLAIDConfig()
        self.centroids: Optional[torch.Tensor] = None       # (C, d)
        self.bucket_edges: Optional[torch.Tensor] = None    # (d, 2^b - 1)
        self.bucket_centers: Optional[torch.Tensor] = None  # (d, 2^b)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @torch.no_grad()
    def train(
        self,
        embeddings: torch.Tensor,
        centroids: Optional[torch.Tensor] = None,
    ) -> None:
        """Fit the PLAID state (centroids + per-dim residual buckets).

        Args:
            embeddings: (N, d) training tokens. Downsampled uniformly to
                `config.kmeans_sample` if larger.
            centroids: If provided, (C, d) pre-fit centroids. K-means is
                skipped and only the residual bucket boundaries/centers are
                fit. Use this to share k-means across a sweep over
                `residual_bits` (centroids are independent of b).
        """
        train = self._subsample(embeddings)

        if centroids is None:
            self.centroids = self._fit_centroids(train)
        else:
            if centroids.shape != (self.config.n_centroids, self.d):
                raise ValueError(
                    f"shared centroids shape {tuple(centroids.shape)} != "
                    f"({self.config.n_centroids}, {self.d})"
                )
            self.centroids = centroids.to(train.device)
            logger.info(
                "PLAID: reusing shared centroids (C=%d), fitting residual buckets only (b=%d)",
                self.config.n_centroids, self.config.residual_bits,
            )

        self._fit_residual_buckets(train)
        logger.info("PLAID fit complete. Bytes/token=%.2f", self.bytes_per_token)

    # --- helpers broken out of train() so centroids can be shared --------

    def _subsample(self, embeddings: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        device = embeddings.device
        if embeddings.dim() != 2 or embeddings.shape[-1] != self.d:
            raise ValueError(
                f"expected (N, {self.d}) tokens, got {tuple(embeddings.shape)}"
            )

        n = embeddings.shape[0]
        if n <= cfg.kmeans_sample:
            return embeddings

        # randperm on MPS can be slow/unsupported for huge N — sample on CPU then move.
        gen = torch.Generator(device="cpu")
        gen.manual_seed(cfg.seed)
        perm = torch.randperm(n, generator=gen)
        return embeddings[perm[: cfg.kmeans_sample].to(device)]

    @torch.no_grad()
    def _fit_centroids(self, train: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        device = train.device

        logger.info(
            "PLAID k-means: n=%d tokens, C=%d, iters=%d, d=%d",
            train.shape[0], cfg.n_centroids, cfg.kmeans_iters, self.d,
        )

        gen = torch.Generator(device="cpu")
        gen.manual_seed(cfg.seed)
        init_idx = torch.randperm(train.shape[0], generator=gen)[: cfg.n_centroids]
        centroids = train[init_idx.to(device)].clone().contiguous()

        for it in range(cfg.kmeans_iters):
            labels = self._assign(train, centroids)
            new_centroids = torch.zeros_like(centroids)
            counts = torch.zeros(cfg.n_centroids, device=device, dtype=train.dtype)
            new_centroids.index_add_(0, labels, train)
            counts.index_add_(
                0, labels, torch.ones(labels.shape[0], device=device, dtype=train.dtype)
            )
            nonempty = counts > 0
            new_centroids[nonempty] /= counts[nonempty].unsqueeze(1)
            # Empty clusters: carry the previous centroid forward (avoid NaN).
            new_centroids[~nonempty] = centroids[~nonempty]
            shift = (new_centroids - centroids).pow(2).mean().sqrt().item()
            centroids = new_centroids
            logger.debug("  k-means iter %d/%d  shift=%.4g", it + 1, cfg.kmeans_iters, shift)

        return centroids

    @torch.no_grad()
    def _fit_residual_buckets(self, train: torch.Tensor) -> None:
        cfg = self.config
        device = train.device
        assert self.centroids is not None

        labels = self._assign(train, self.centroids)
        residuals = train - self.centroids[labels]  # (N, d)

        n_buckets = 2 ** cfg.residual_bits
        if n_buckets > 1:
            q = torch.linspace(0, 1, n_buckets + 1, device=device)[1:-1]
            edges = torch.quantile(residuals, q, dim=0).T.contiguous()
        else:
            edges = torch.zeros(self.d, 0, device=device, dtype=residuals.dtype)

        centers = torch.zeros(self.d, n_buckets, device=device, dtype=residuals.dtype)
        bucket_idx = self._bucketize_per_dim(residuals, edges)
        for b in range(n_buckets):
            mask = (bucket_idx == b)
            denom = mask.sum(dim=0).clamp_min(1)
            centers[:, b] = (residuals * mask).sum(dim=0) / denom

        self.bucket_edges = edges
        self.bucket_centers = centers

    # ------------------------------------------------------------------
    # Encoding / decoding
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map tokens to (centroid_id, per-dim residual bucket index).

        Args:
            x: (..., d) float tokens.

        Returns:
            ids:     (...,) long — nearest centroid index in [0, C).
            resid:   (..., d) uint8 — per-dim bucket index in [0, 2^b).
        """
        self._require_trained()
        flat = x.reshape(-1, self.d)
        ids = self._assign(flat, self.centroids)
        residuals = flat - self.centroids[ids]
        bucket_idx = self._bucketize_per_dim(residuals, self.bucket_edges).to(torch.uint8)
        leading = x.shape[:-1]
        return ids.reshape(*leading), bucket_idx.reshape(*leading, self.d)

    @torch.no_grad()
    def decode(self, ids: torch.Tensor, bucket_idx: torch.Tensor) -> torch.Tensor:
        """Reconstruct float tokens from compressed codes."""
        self._require_trained()
        leading = ids.shape
        flat_ids = ids.reshape(-1)
        flat_idx = bucket_idx.reshape(-1, self.d).long()
        base = self.centroids[flat_ids]  # (N, d)
        dim_idx = torch.arange(self.d, device=ids.device).unsqueeze(0).expand_as(flat_idx)
        residual = self.bucket_centers[dim_idx, flat_idx]  # (N, d)
        return (base + residual).reshape(*leading, self.d)

    @torch.no_grad()
    def encode_decode(self, x: torch.Tensor) -> torch.Tensor:
        """Shortcut: `decode(*encode(x))`. Use this for quality evaluation."""
        ids, bucket_idx = self.encode(x)
        return self.decode(ids, bucket_idx)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _require_trained(self) -> None:
        if self.centroids is None or self.bucket_edges is None or self.bucket_centers is None:
            raise RuntimeError("PLAIDQuantizer not trained — call .train() first")

    @torch.no_grad()
    def _assign(self, x: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
        """Nearest-centroid assignment, chunked to bound peak memory."""
        n = x.shape[0]
        labels = torch.empty(n, dtype=torch.long, device=x.device)
        chunk = self.config.kmeans_chunk
        for i in range(0, n, chunk):
            end = min(i + chunk, n)
            # ||x - c||^2 = ||x||^2 - 2 x·c + ||c||^2; constant-per-row part drops out.
            d = torch.cdist(x[i:end], centroids)
            labels[i:end] = d.argmin(dim=1)
        return labels

    @torch.no_grad()
    def _bucketize_per_dim(self, residuals: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        """Per-dim bucketize: returns (N, d) long in [0, n_buckets)."""
        if edges.shape[1] == 0:
            return torch.zeros(residuals.shape, dtype=torch.long, device=residuals.device)
        out = torch.empty(residuals.shape, dtype=torch.long, device=residuals.device)
        for j in range(self.d):
            out[:, j] = torch.bucketize(residuals[:, j].contiguous(), edges[j].contiguous())
        return out

    def to(self, device: torch.device) -> "PLAIDQuantizer":
        """Move tables in-place and return self (fluent)."""
        if self.centroids is not None:
            self.centroids = self.centroids.to(device)
        if self.bucket_edges is not None:
            self.bucket_edges = self.bucket_edges.to(device)
        if self.bucket_centers is not None:
            self.bucket_centers = self.bucket_centers.to(device)
        return self

    @property
    def bytes_per_token(self) -> float:
        return self.config.bytes_per_token(self.d)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save fit state to disk. Keep the path's parent around — torch.save needs it."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._require_trained()
        torch.save(
            {
                "centroids": self.centroids.detach().cpu(),
                "bucket_edges": self.bucket_edges.detach().cpu(),
                "bucket_centers": self.bucket_centers.detach().cpu(),
                "d": self.d,
                "config": asdict(self.config),
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: Optional[torch.device] = None) -> "PLAIDQuantizer":
        state = torch.load(Path(path), map_location="cpu", weights_only=False)
        q = cls(d=state["d"], config=PLAIDConfig(**state["config"]))
        q.centroids = state["centroids"]
        q.bucket_edges = state["bucket_edges"]
        q.bucket_centers = state["bucket_centers"]
        if device is not None:
            q.to(device)
        return q


# ----------------------------------------------------------------------
# Integration adapter for evaluation/run_baseline_comparison.py
# ----------------------------------------------------------------------


def method_plaid(pq: PLAIDQuantizer, device: torch.device):
    """Return (encode_query, encode_doc, meta) compatible with `rank_all`.

    Scoring path: encode_doc reconstructs float tokens (centroid + decoded
    residual); the standard MaxSim einsum runs on top. This matches the quality
    a true ADC implementation would achieve.
    """
    pq.to(device)

    def enc_q(e: torch.Tensor) -> torch.Tensor:
        return e.to(device)  # queries stay float (asymmetric scoring)

    def enc_d(e: torch.Tensor) -> torch.Tensor:
        return pq.encode_decode(e.to(device))

    bpt = pq.bytes_per_token
    bpt_int = max(1, int(math.ceil(bpt)))
    compression = (pq.d * 4) // bpt_int
    return enc_q, enc_d, {
        "label": f"PLAID (C={pq.config.n_centroids}, b={pq.config.residual_bits})",
        "r": pq.d,
        "bytes_per_token": bpt_int,
        "bytes_per_token_exact": bpt,
        "compression": compression,
        "trained": False,  # centroids are fit, but no gradient-based learning
        "method": "plaid",
        "plaid_config": asdict(pq.config),
    }
