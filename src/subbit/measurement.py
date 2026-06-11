"""Reviewer-ready measurement utilities for training + eval runs.

Every SubBit experiment should record enough metadata and performance
numbers that a reader can (a) reproduce the run, and (b) understand both
quality AND efficiency. Papers in this domain (PLAID, EMVB, MUVERA,
ColBERTv2) consistently report:

  - Per-query latency (mean, P50, P95, P99) — CPU and/or GPU
  - Throughput (queries/sec)
  - Peak device memory during eval
  - Index size in GB (total tokens × bytes/token)
  - Model parameter count
  - Reproducibility metadata (git SHA, torch version, hardware)

This module provides a small set of primitives that the eval scripts call
without cluttering their logic:

  * :class:`LatencyTracker` — thin perf_counter wrapper that collects a list
    of per-query wall-clock times and reports quantiles.
  * :class:`MemoryTracker` — resets + reads peak device memory; CPU RSS
    is captured on best-effort via psutil (if installed).
  * :func:`collect_run_metadata` — one-call dump of all environment info.
  * :class:`EvalMeasurement` — aggregate dataclass written into results JSON.

Overhead is negligible (≪1 ms per query for latency, zero ongoing cost for
memory) so tracking is unconditional. If a rollout of the measurement code
ever starts distorting benchmarks we can gate on an env flag, but at this
scale it won't.
"""
from __future__ import annotations

import logging
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def _git_sha(short: bool = True) -> Optional[str]:
    """Return the current git commit SHA, or None if not in a repo."""
    try:
        args = ["git", "rev-parse", "--short" if short else "HEAD", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
        out = subprocess.check_output(args, stderr=subprocess.DEVNULL).decode().strip()
        return out or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _git_dirty() -> Optional[bool]:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode()
        return bool(out.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _torch_info() -> dict:
    try:
        import torch
        info = {
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
        }
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
            info["cuda_device_count"] = torch.cuda.device_count()
        # MPS check
        mps_avail = getattr(getattr(torch, "backends", None), "mps", None)
        info["mps_available"] = bool(mps_avail and mps_avail.is_available())
        return info
    except ImportError:
        return {}


def collect_run_metadata(config: Optional[dict] = None) -> dict:
    """Environment + provenance metadata. Written into every results JSON.

    Args:
        config: Optional run config (will be snapshot-included verbatim).
    """
    md: dict[str, Any] = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "hostname": platform.node(),
        "git_sha": _git_sha(short=True),
        "git_dirty": _git_dirty(),
    }
    md.update(_torch_info())
    if config is not None:
        md["config"] = config
    return md


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------


@dataclass
class LatencyStats:
    """Aggregated per-query latency in milliseconds."""

    count: int = 0
    total_ms: float = 0.0
    mean_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    throughput_qps: float = 0.0

    @classmethod
    def from_samples(cls, samples_ms: list[float]) -> "LatencyStats":
        if not samples_ms:
            return cls()
        sorted_s = sorted(samples_ms)
        n = len(sorted_s)

        def pct(p):
            idx = min(n - 1, max(0, int(round((p / 100.0) * (n - 1)))))
            return sorted_s[idx]

        total = sum(sorted_s)
        return cls(
            count=n,
            total_ms=total,
            mean_ms=total / n,
            p50_ms=pct(50),
            p95_ms=pct(95),
            p99_ms=pct(99),
            min_ms=sorted_s[0],
            max_ms=sorted_s[-1],
            throughput_qps=(n * 1000.0 / total) if total > 0 else 0.0,
        )


class LatencyTracker:
    """Collects per-query wall-clock times.

    Usage::

        tracker = LatencyTracker()
        for qid in queries:
            with tracker.measure():
                score(qid)
        stats = tracker.stats()
    """

    def __init__(self):
        self._samples_ms: list[float] = []

    def measure(self) -> "_LatencyCtx":
        return _LatencyCtx(self)

    def record(self, ms: float) -> None:
        self._samples_ms.append(ms)

    def stats(self) -> LatencyStats:
        return LatencyStats.from_samples(self._samples_ms)

    def reset(self) -> None:
        self._samples_ms.clear()

    @property
    def samples_ms(self) -> list[float]:
        return list(self._samples_ms)


def _device_sync() -> None:
    """Block until queued accelerator work finishes.

    MaxSim runs on an async backend (MPS/CUDA): kernels are *enqueued* and
    control returns immediately, so a bare ``time.perf_counter()`` measures
    launch/enqueue time, not execution. Without this sync the recorded latency
    can be ~10-60x below the true on-device cost. No-op on CPU.
    """
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            torch.mps.synchronize()
    except Exception:
        pass


class _LatencyCtx:
    def __init__(self, tracker: LatencyTracker):
        self._tracker = tracker
        self._t0 = 0.0

    def __enter__(self):
        _device_sync()  # drain prior queued work so it is not charged to this block
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        _device_sync()  # wait for THIS block's kernels to actually execute
        self._tracker.record((time.perf_counter() - self._t0) * 1000.0)
        return False


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


@dataclass
class MemorySnapshot:
    """Peak device memory + host RSS at the moment stats() is called."""

    device_type: str = "cpu"
    device_peak_mb: Optional[float] = None
    host_rss_mb: Optional[float] = None


class MemoryTracker:
    """Tracks peak memory on the given torch device.

    Resets counters on ``__enter__`` so the peak covers only work inside the
    with-block. On CPU we fall back to psutil RSS deltas (best effort).
    """

    def __init__(self, device):
        import torch
        self._torch = torch
        self._device = torch.device(device) if not isinstance(device, torch.device) else device
        self._host_before_mb: Optional[float] = None
        self._host_after_mb: Optional[float] = None
        self._device_peak_mb: Optional[float] = None

    def __enter__(self):
        torch = self._torch
        dt = self._device.type
        if dt == "cuda":
            torch.cuda.reset_peak_memory_stats(self._device)
        # MPS doesn't expose reset; we record delta in __exit__.
        self._host_before_mb = _host_rss_mb()
        return self

    def __exit__(self, exc_type, exc, tb):
        torch = self._torch
        dt = self._device.type
        if dt == "cuda":
            self._device_peak_mb = torch.cuda.max_memory_allocated(self._device) / (1024 ** 2)
        elif dt == "mps":
            # Best-effort: current_allocated is the live set (peak approximation).
            try:
                self._device_peak_mb = torch.mps.current_allocated_memory() / (1024 ** 2)
            except Exception:  # API not present on older torch
                self._device_peak_mb = None
        self._host_after_mb = _host_rss_mb()
        return False

    def snapshot(self) -> MemorySnapshot:
        host_peak = self._host_after_mb
        if host_peak is not None and self._host_before_mb is not None:
            host_peak = max(host_peak, self._host_before_mb)
        return MemorySnapshot(
            device_type=self._device.type,
            device_peak_mb=self._device_peak_mb,
            host_rss_mb=host_peak,
        )


def _host_rss_mb() -> Optional[float]:
    """Host resident set size in MB, via psutil if available, else None."""
    try:
        import psutil  # type: ignore
        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Aggregate eval measurement
# ---------------------------------------------------------------------------


@dataclass
class EvalMeasurement:
    """Per-method timing/memory/storage block stored in results JSON.

    Keys are chosen so a paper author can copy-paste into a LaTeX table:
      - ``scoring_latency``: just the score step (encode already done)
      - ``encode_seconds``: wall-clock to pre-encode the doc corpus
      - ``memory.device_peak_mb``: peak VRAM (CUDA) or live MPS allocation
      - ``index.total_bytes``: precomputed doc storage (for the Pareto plot)
    """

    scoring_latency: LatencyStats = field(default_factory=LatencyStats)
    encode_seconds: Optional[float] = None
    memory: MemorySnapshot = field(default_factory=MemorySnapshot)
    eval_mode: Optional[str] = None
    num_docs: Optional[int] = None
    num_queries: Optional[int] = None
    index_total_bytes: Optional[int] = None
    # Per-pass mean scoring latency (ms) when scoring is repeated for run-to-run
    # variance (see run_baseline_comparison --latency-repeats). Empty otherwise.
    latency_repeat_means_ms: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        # Friendly derived fields
        if self.index_total_bytes is not None:
            d["index_total_gb"] = self.index_total_bytes / (1024 ** 3)
        return d


def index_bytes_total(
    num_tokens: int, bytes_per_token: int, doc_len_overhead_per_doc: int = 4,
    num_docs: int = 0,
) -> int:
    """Compute total on-disk/in-memory index size in bytes.

    Mirrors :func:`scoring.compute_storage_bytes` but takes the precomputed
    (num_tokens, bytes/token) numbers so it can be called uniformly from
    any eval path. The doc-length overhead term is the per-doc offset/length
    housekeeping needed for variable-length packed storage.
    """
    return int(num_tokens) * int(bytes_per_token) + int(num_docs) * int(doc_len_overhead_per_doc)
