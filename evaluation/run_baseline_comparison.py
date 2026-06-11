#!/usr/bin/env python3
"""Baseline comparison experiment.

Answers the key reviewers' question: does training the projection actually help
over sign(e), random projection, PCA projection, and identity truncation at the
same bitwidth?

Baselines evaluated (all asymmetric: float query × binary doc):
  random_r{R}      r=32,64,128 — random orthogonal projection + sign
  pca_r{R}         r=32,64,128 — PCA projection + sign (no fine-tuning)
  identity_r{R}    r=32,64,128 — first r dims + sign
  subbit_r{R}      r=32,64,128 — trained SubBit (checkpoints required)
  fp128                        — full-precision upper bound

Metrics reported: MRR@10, NDCG@10, Recall@1000
Storage reported: bytes/token, compression vs FP128

Usage:
    # With trained checkpoints
    python evaluation/run_baseline_comparison.py \\
        --checkpoints outputs/checkpoints/r32.pt outputs/checkpoints/r64.pt outputs/checkpoints/r128.pt \\
        --config configs/base.yaml \\
        --output outputs/results/baseline_comparison.json

    # Quick debug run (no data required)
    python evaluation/run_baseline_comparison.py --debug --checkpoints <path>

    # Without checkpoints (baselines only, no SubBit rows)
    python evaluation/run_baseline_comparison.py --config configs/base.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Callable

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.subbit.baselines import (
    random_projection_binary,
    pca_projection_binary,
    identity_truncation,
    product_quantize,
    pq_maxsim,
    TorchProductQuantizer,
)
from src.subbit.baselines_plaid import PLAIDConfig, PLAIDQuantizer, method_plaid
from src.subbit.data import EmbeddingStore, load_qrels, create_debug_data, resolve_embedding_cache_path
from src.subbit.evaluation import compute_mrr, compute_ndcg, compute_recall
from src.subbit.measurement import (
    EvalMeasurement,
    LatencyTracker,
    MemoryTracker,
    collect_run_metadata,
    index_bytes_total,
)
from src.subbit.model import SubBitModel
from src.subbit.scoring import maxsim

try:
    import faiss  # optional; only used for --include-opq (OPQ learned rotation)
except ImportError:
    faiss = None
from src.subbit.utils import setup_logging, get_device, seed_everything, ensure_dir

logger = logging.getLogger(__name__)

REGIMES = [32, 64, 128]

# Inferred from data at runtime — set in main().
FP_DIM: int = 128  # default; overwritten by doc_store.dim


# ---------------------------------------------------------------------------
# Core ranking loop — shared by every method
# ---------------------------------------------------------------------------


# Set from --latency-repeats in main(). Float-path scoring is repeated this many
# times (corpus encoded once) to measure run-to-run latency variance.
LATENCY_REPEATS = 1


def rank_all(
    query_store: EmbeddingStore,
    doc_store: EmbeddingStore,
    qrels: dict,
    encode_query: Callable[[torch.Tensor], torch.Tensor],
    encode_doc: Callable[[torch.Tensor], torch.Tensor],
    doc_ids: list,
    device: torch.device,
    max_queries: int = -1,
    top_k: int = 1000,
    eval_mode: str = "float",
    binary_docs: bool = False,
    bytes_per_token: int = 0,
) -> tuple[dict[str, list[str]], EvalMeasurement]:
    """Score every (query, doc) pair — mirrors evaluate_retrieval pattern.

    Args:
        eval_mode: only "float" is active in this paper repo.
        binary_docs: Set True iff ``encode_doc`` returns {-1, +1} values.
    """
    if eval_mode != "float":
        raise ValueError(f"eval_mode must be float in this paper repo, got {eval_mode!r}")

    measurement = EvalMeasurement(eval_mode=eval_mode)
    latency = LatencyTracker()

    query_ids = list(qrels.keys())
    if 0 < max_queries < len(query_ids):
        query_ids = query_ids[:max_queries]

    # Pre-encode all docs on device (timed separately from scoring)
    logger.info(f"  Pre-encoding {len(doc_ids):,} docs...")
    import time as _time
    _t_enc = _time.perf_counter()
    encoded_docs: dict[str, torch.Tensor] = {}
    for pid in tqdm(doc_ids, desc="  encode docs", leave=False):
        try:
            d = doc_store.get(pid).to(device)
            with torch.no_grad():
                encoded_docs[pid] = encode_doc(d)
        except (KeyError, FileNotFoundError):
            pass
    measurement.encode_seconds = _time.perf_counter() - _t_enc

    if not encoded_docs:
        return {}, measurement

    # Bookkeeping for the Pareto plot
    measurement.num_docs = len(encoded_docs)
    total_tokens = int(sum(t.shape[0] for t in encoded_docs.values()))
    if bytes_per_token > 0:
        measurement.index_total_bytes = index_bytes_total(
            num_tokens=total_tokens,
            bytes_per_token=bytes_per_token,
            num_docs=len(encoded_docs),
        )

    # ------------------------------------------------------------------
    # Float path.
    # ------------------------------------------------------------------
    valid_doc_ids = sorted(list(encoded_docs.keys()), key=lambda pid: encoded_docs[pid].shape[0])
    if not valid_doc_ids:
        return {}, measurement

    logger.info("  Padding and batching all encoded docs into memory (length-sorted chunks)...")

    DOC_BATCH_SIZE = 25000
    batched_doc_vecs = []
    batched_doc_masks = []

    for i in range(0, len(valid_doc_ids), DOC_BATCH_SIZE):
        batch_ids = valid_doc_ids[i:i + DOC_BATCH_SIZE]
        max_doc_len = max(encoded_docs[pid].shape[0] for pid in batch_ids)
        dim = encoded_docs[batch_ids[0]].shape[1]

        doc_vecs = torch.zeros(len(batch_ids), max_doc_len, dim, device=device)
        doc_mask = torch.zeros(len(batch_ids), max_doc_len, dtype=torch.bool, device=device)

        for j, pid in enumerate(batch_ids):
            d = encoded_docs[pid]
            n_tokens = d.shape[0]
            doc_vecs[j, :n_tokens, :] = d
            doc_mask[j, :n_tokens] = True

        batched_doc_vecs.append(doc_vecs)
        batched_doc_masks.append(doc_mask)

    del encoded_docs

    # Corpus is encoded/batched ONCE above. The scoring loop below is repeated
    # to measure run-to-run latency variance without re-paying the encode cost.
    # When repeating (LATENCY_REPEATS>1) one untimed WARMUP pass runs first to
    # absorb the cold-start outlier (kernels/caches), then LATENCY_REPEATS timed
    # passes are recorded. Rankings are deterministic across passes; kept once.
    n_reps = max(1, LATENCY_REPEATS)
    do_warmup = n_reps > 1
    total_passes = (1 if do_warmup else 0) + n_reps
    repeat_means_ms = []
    rankings = {}
    stats = {}
    first_timed_pass = 1 if do_warmup else 0
    # Preload query embeddings to device ONCE. The .get()/.to() was already
    # OUTSIDE the timed block, so this does not change what is timed (still
    # encode_query + scoring) -- it only removes redundant per-pass disk I/O so
    # repeated passes measure scoring-kernel variance, not file-read variance.
    preloaded_q = []
    for qid in query_ids:
        try:
            preloaded_q.append((qid, query_store.get(qid).to(device)))
        except (KeyError, FileNotFoundError):
            continue
    with MemoryTracker(device) as mem:
        for pass_i in range(total_passes):
            is_warmup = do_warmup and pass_i == 0
            keep_rankings = pass_i == first_timed_pass
            latency_rep = LatencyTracker()
            if n_reps == 1:
                desc = "  ranking queries"
            elif is_warmup:
                desc = "  ranking queries (warmup, untimed)"
            else:
                desc = f"  ranking queries (lat rep {pass_i}/{n_reps})"
            for qid, q_embs in tqdm(preloaded_q, desc=desc, leave=False):
                with latency_rep.measure():
                    with torch.no_grad():
                        q_enc = encode_query(q_embs)

                        all_scores = []
                        for doc_vecs, doc_mask in zip(batched_doc_vecs, batched_doc_masks):
                            sim = torch.einsum("md,cnd->cmn", q_enc, doc_vecs)
                            sim = sim.masked_fill(~doc_mask.unsqueeze(1), float('-inf'))
                            scores = sim.max(dim=-1).values.sum(dim=-1)
                            all_scores.append(scores)

                        scores_tensor = torch.cat(all_scores, dim=0)

                        actual_k = min(top_k, scores_tensor.shape[0])
                        _, topk_idx = torch.topk(scores_tensor, k=actual_k)

                if keep_rankings:
                    rankings[qid] = [valid_doc_ids[i] for i in topk_idx.cpu().tolist()]
            stats = latency_rep.stats()
            if not is_warmup:
                repeat_means_ms.append(stats.mean_ms)

    measurement.memory = mem.snapshot()
    measurement.scoring_latency = stats  # full quantiles from the last pass
    if n_reps > 1:
        measurement.latency_repeat_means_ms = repeat_means_ms
    measurement.num_queries = len(rankings)
    return rankings, measurement


def compute_metrics(rankings: dict, qrels: dict) -> dict:
    return {
        "mrr@10": compute_mrr(rankings, qrels, k=10),
        "ndcg@10": compute_ndcg(rankings, qrels, k=10),
        "recall@100": compute_recall(rankings, qrels, k=100),
        "recall@1000": compute_recall(rankings, qrels, k=1000),
    }


# ---------------------------------------------------------------------------
# Projection caching — build projection matrices once per r, not per method
# ---------------------------------------------------------------------------


def build_random_projection(r: int, d: int = FP_DIM, seed: int = 42) -> torch.Tensor:
    """Return (r, d) random orthonormal matrix."""
    torch.manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(d, r))
    return Q.T  # (r, d)


def build_pca_projection(
    doc_store: EmbeddingStore, r: int, sample_size: int = 50_000,
    train_doc_ids: list[str] | set[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ((r, d) PCA projection matrix, (d,) PCA mean) fitted on sampled doc embeddings.

    The mean is returned so callers center both query and doc as (x - mean) @ R.T
    (= sklearn pca.transform); applying R without the mean was a fairness bug.

    Args:
        doc_store: Document embedding store.
        r: Target dimension.
        sample_size: Number of token embeddings to sample.
        train_doc_ids: If provided, restrict PCA fit to these document IDs
            to avoid leaking dev/test distribution information.
    """
    from sklearn.decomposition import PCA
    import numpy as np

    if train_doc_ids is not None:
        n_docs = len(train_doc_ids)
        logger.info(f"  Fitting PCA (r={r}) on {min(sample_size, n_docs):,} train-only samples "
                     f"({n_docs:,} train docs)...")
        sample = doc_store.sample_embeddings(sample_size, ids=train_doc_ids)
    else:
        logger.info(f"  Fitting PCA (r={r}) on {min(sample_size, len(doc_store.get_all_ids())):,} samples...")
        sample = doc_store.sample_embeddings(sample_size)
    pca = PCA(n_components=r)
    pca.fit(sample.cpu().numpy())
    # components_ are fit on centered data; the correct projection is
    # (x - mean) @ components_.T (= pca.transform). Return the mean so the
    # query and doc are both centered consistently (see method_pca_proj).
    R = torch.tensor(pca.components_, dtype=torch.float32)      # (r, d)
    mean = torch.tensor(pca.mean_, dtype=torch.float32)         # (d,)
    return R, mean


# ---------------------------------------------------------------------------
# Method builders — return (encode_query, encode_doc, meta) triples
# ---------------------------------------------------------------------------


def method_fp128(device: torch.device):
    def enc(e: torch.Tensor) -> torch.Tensor:
        return e.to(device)
    return enc, enc, {
        "label": "FP128",
        "r": 128,
        "bytes_per_token": FP_DIM * 4,
        "compression": 1,
        "trained": False,
    }


def method_random_proj(r: int, R: torch.Tensor, device: torch.device):
    R = R.to(device)

    def enc_q(e: torch.Tensor) -> torch.Tensor:
        return e.to(device) @ R.T  # float projected

    def enc_d(e: torch.Tensor) -> torch.Tensor:
        proj = e.to(device) @ R.T
        b = torch.sign(proj)
        b[b == 0] = 1.0
        return b

    return enc_q, enc_d, {
        "label": f"random_proj r={r}",
        "r": r,
        "bytes_per_token": r // 8,
        "compression": FP_DIM * 4 // (r // 8),
        "trained": False,
    }


def method_pca_proj(r: int, R: torch.Tensor, mean: torch.Tensor, device: torch.device):
    R = R.to(device)
    mean = mean.to(device)  # center both query and doc (= sklearn pca.transform)

    def enc_q(e: torch.Tensor) -> torch.Tensor:
        return (e.to(device) - mean) @ R.T

    def enc_d(e: torch.Tensor) -> torch.Tensor:
        proj = (e.to(device) - mean) @ R.T
        b = torch.sign(proj)
        b[b == 0] = 1.0
        return b

    return enc_q, enc_d, {
        "label": f"pca_proj r={r}",
        "r": r,
        "bytes_per_token": r // 8,
        "compression": FP_DIM * 4 // (r // 8),
        "trained": False,
    }


def method_identity(r: int, device: torch.device):
    def enc_q(e: torch.Tensor) -> torch.Tensor:
        return e.to(device)[..., :r]  # float truncated

    def enc_d(e: torch.Tensor) -> torch.Tensor:
        t = e.to(device)[..., :r]
        b = torch.sign(t)
        b[b == 0] = 1.0
        return b

    return enc_q, enc_d, {
        "label": f"identity r={r}",
        "r": r,
        "bytes_per_token": r // 8,
        "compression": FP_DIM * 4 // (r // 8),
        "trained": False,
    }


def method_subbit(model: SubBitModel, r: int, device: torch.device):
    model = model.to(device).eval()

    def enc_q(e: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return model.encode_query(e.to(device))

    def enc_d(e: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return model.encode_document(e.to(device))

    return enc_q, enc_d, {
        "label": f"SubBit r={r}",
        "r": r,
        "bytes_per_token": r // 8,
        "compression": FP_DIM * 4 // (r // 8),
        "trained": True,
    }


def method_pq(pq: TorchProductQuantizer, device: torch.device):
    """Product Quantization baseline (reconstruct + MaxSim).

    Uses PQ decode to reconstruct float vectors, then standard MaxSim.
    Quality is identical to ADC scoring; ADC is measured separately in speed benchmarks.
    Storage: n_subquantizers bytes per token (default 8 sub-q × 1 byte = 8 bytes).
    """
    def enc_q(e: torch.Tensor) -> torch.Tensor:
        return e.to(device)

    def enc_d(e: torch.Tensor) -> torch.Tensor:
        e_flat = e.to(device).reshape(-1, e.shape[-1])
        codes = pq.encode(e_flat)
        return pq.decode(codes)

    bytes_per_token = pq.m  # n_subquantizers bytes (1 byte per sub-quantizer)
    return enc_q, enc_d, {
        "label": f"PQ ({pq.m}×{pq.k} centroids)",
        "r": FP_DIM,
        "bytes_per_token": bytes_per_token,
        "compression": FP_DIM * 4 // bytes_per_token,
        "trained": False,
    }


def method_opq(index, m_subq: int, device: torch.device):
    """OPQ baseline (faiss): learned rotation + product quantization.

    Same reconstruct-then-MaxSim path as ``method_pq``, but the faiss
    ``OPQMatrix`` learns an orthogonal rotation (Ge et al. 2013) before PQ,
    so it preempts the "is your PQ baseline optimised?" question at the same
    byte budget. ``index`` is a trained faiss index whose ``sa_encode`` /
    ``sa_decode`` reconstruct in the original 128-d space (the pre-transform
    applies the inverse rotation on decode). Storage: ``m_subq`` bytes/token
    (8 bits per sub-quantizer), matched to the plain-PQ baseline.
    """
    def enc_q(e: torch.Tensor) -> torch.Tensor:
        return e.to(device)

    def enc_d(e: torch.Tensor) -> torch.Tensor:
        e_flat = e.reshape(-1, e.shape[-1]).detach().cpu().numpy().astype("float32")
        recon = index.sa_decode(index.sa_encode(e_flat))
        return torch.from_numpy(recon).to(device)

    return enc_q, enc_d, {
        "label": f"OPQ ({m_subq}×256 centroids)",
        "r": FP_DIM,
        "bytes_per_token": m_subq,
        "compression": FP_DIM * 4 // m_subq,
        "trained": False,
    }


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------


def print_table(all_results: list[dict]) -> None:
    # Sort: FP128 first, then by trained/not-trained, then by r descending
    fp = [x for x in all_results if x["meta"]["r"] == 128 and not x["meta"]["trained"] and x["meta"]["bytes_per_token"] == 512]
    baselines = sorted(
        [x for x in all_results if x not in fp and not x["meta"]["trained"]],
        key=lambda x: (x["meta"]["r"], x["meta"]["label"]),
        reverse=True,
    )
    trained = sorted(
        [x for x in all_results if x["meta"]["trained"]],
        key=lambda x: x["meta"]["r"],
        reverse=True,
    )
    rows = fp + baselines + trained

    # Retention relative to FP128
    fp_mrr = fp[0]["metrics"]["mrr@10"] if fp else None
    fp_ndcg = fp[0]["metrics"]["ndcg@10"] if fp else None

    col_label = 32
    col_r = 5
    col_bytes = 10
    col_compr = 8
    col_mrr = 11
    col_ndcg = 11
    col_recall = 12
    col_ret = 9

    header = (
        f"{'Method':<{col_label}} {'r':>{col_r}} {'Bytes/tok':>{col_bytes}} "
        f"{'Compr':>{col_compr}} {'MRR@10':>{col_mrr}} {'NDCG@10':>{col_ndcg}} "
        f"{'Recall@1k':>{col_recall}} {'Ret%':>{col_ret}}"
    )
    sep = "─" * len(header)

    print()
    print("=" * len(header))
    print("BASELINE COMPARISON: trained SubBit vs unlearned alternatives")
    print("=" * len(header))
    print(header)
    print(sep)

    prev_trained = None
    for row in rows:
        meta = row["meta"]
        m = row["metrics"]
        if prev_trained is not None and meta["trained"] != prev_trained:
            print(sep)
        prev_trained = meta["trained"]

        mrr = m.get("mrr@10", float("nan"))
        ndcg = m.get("ndcg@10", float("nan"))
        recall = m.get("recall@1000", float("nan"))

        ret_pct = ""
        if fp_mrr and meta["bytes_per_token"] < 512:
            ret = mrr / fp_mrr * 100
            ret_pct = f"{ret:.1f}%"

        trained_marker = " *" if meta["trained"] else "  "
        print(
            f"{meta['label'] + trained_marker:<{col_label}} {meta['r']:>{col_r}} "
            f"{meta['bytes_per_token']:>{col_bytes}} {meta['compression']:>{col_compr}}x "
            f"{mrr:>{col_mrr}.4f} {ndcg:>{col_ndcg}.4f} "
            f"{recall:>{col_recall}.4f} {ret_pct:>{col_ret}}"
        )

    print(sep)
    print("  * = trained SubBit model")
    print()

    # Gap summary
    if fp_mrr:
        print("Per-method gain over FP128-normalized baseline at same r:")
        for r in sorted(set(x["meta"]["r"] for x in rows if x["meta"]["r"] < 128), reverse=True):
            best_baseline = max(
                (x for x in rows if x["meta"]["r"] == r and not x["meta"]["trained"]),
                key=lambda x: x["metrics"].get("mrr@10", 0),
                default=None,
            )
            subbit_row = next(
                (x for x in rows if x["meta"]["r"] == r and x["meta"]["trained"]),
                None,
            )
            if best_baseline and subbit_row:
                gap = (
                    subbit_row["metrics"]["mrr@10"] - best_baseline["metrics"]["mrr@10"]
                ) * 100
                print(
                    f"  r={r:>3}: SubBit {subbit_row['metrics']['mrr@10']:.4f}  "
                    f"best-baseline {best_baseline['metrics']['mrr@10']:.4f}  "
                    f"Δ={gap:+.2f}pp  [{best_baseline['meta']['label']}]"
                )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--config", action="append", default=None,
        help="Config file(s). Multiple allowed; later files override earlier ones.",
    )
    p.add_argument("--debug", action="store_true", help="Use debug data (fast smoke test)")
    p.add_argument(
        "--checkpoint", "--checkpoints", nargs="*", default=[], dest="checkpoints",
        help="Paths to trained SubBit checkpoints (one per r value).",
    )
    p.add_argument("--regimes", type=int, nargs="+", default=REGIMES)
    p.add_argument("--max-queries", type=int, default=-1)
    p.add_argument(
        "--query-embeddings", default=None,
        help="Override the query embedding cache path (e.g. an augmented "
             "query_embeddings_aug.pt). Defaults to the slice's query_embeddings.pt.",
    )
    p.add_argument(
        "--latency-repeats", type=int, default=1,
        help=(
            "Repeat the float-path scoring loop N times (corpus encoded once) "
            "to measure run-to-run scoring-latency variance. Per-pass mean ms "
            "is stored under measurement.scoring_latency.repeat_means_ms."
        ),
    )
    p.add_argument("--pca-sample-size", type=int, default=50_000)
    p.add_argument("--output", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--subbit-ablate-scale", action="store_true",
        help=(
            "For each SubBit checkpoint, also score an extra row with "
            "model.use_scale=False (the query-side S(q) head skipped at inference). "
            "Labelled 'SubBit r={r} (no scale)'."
        ),
    )
    p.add_argument(
        "--skip-baselines", metavar="CACHED_JSON", default=None,
        help=(
            "Path to a previously-saved baseline_comparison.json. Skips FP128, "
            "identity, random, and PCA computation and reuses those rows. Only the "
            "SubBit checkpoints are re-scored. The cache must be from the same "
            "corpus/config as this run — the script does not verify compatibility."
        ),
    )
    p.add_argument(
        "--include-plaid", action="store_true",
        help=(
            "Also evaluate a PLAID-style baseline (centroid + per-dim residual "
            "quantization). Fits once on train-split tokens; one row per "
            "--plaid-residual-bits value. Off by default so cached JSONs stay valid."
        ),
    )
    p.add_argument(
        "--plaid-centroids", type=int, default=65536,
        help="PLAID: number of k-means centroids (paper: 2^18). Default 65536.",
    )
    p.add_argument(
        "--plaid-residual-bits", type=int, nargs="+", default=[2],
        help=(
            "PLAID: residual bucket-index width per dim. Accepts multiple values "
            "for a sweep, e.g. `--plaid-residual-bits 1 2 4`. Default: 2."
        ),
    )
    p.add_argument(
        "--plaid-kmeans-sample", type=int, default=1_000_000,
        help="PLAID: cap on tokens used for centroid + bucket fitting.",
    )
    p.add_argument(
        "--include-opq", action="store_true",
        help=(
            "Also evaluate an OPQ baseline (faiss OPQMatrix learned rotation + PQ) "
            "and a plain faiss-PQ control at the same byte budget. Requires faiss-cpu. "
            "Off by default so cached JSONs stay valid."
        ),
    )
    p.add_argument(
        "--opq-subquantizers", type=int, default=8,
        help="OPQ/PQ-control: number of sub-quantizers M (8 bits each → M bytes/token). Default 8.",
    )
    p.add_argument(
        "--opq-train-sample", type=int, default=500_000,
        help="OPQ/PQ-control: cap on train-split tokens used to fit the rotation + codebook.",
    )
    p.add_argument(
        "--eval-mode", choices=["float"], default="float",
        help="Scoring backend. The paper release keeps only the fp32 einsum path.",
    )
    p.add_argument(
        "overrides", nargs="*",
        help="Config overrides in dotlist format (e.g., data.scale=50k run.name=foo).",
    )
    return p.parse_args()


def _load_config(args: argparse.Namespace):
    """Match train.py's config-stacking pattern: multiple --config + dotlist overrides."""
    config_files = list(args.config or [])
    true_overrides = []
    for o in args.overrides:
        if o.endswith((".yaml", ".yml")):
            config_files.append(o)
        else:
            true_overrides.append(o)
    if not config_files:
        config_files = ["configs/base.yaml"]
    cfg = OmegaConf.load(config_files[0])
    for extra in config_files[1:]:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(extra))
    if true_overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(true_overrides))
    return cfg


def main() -> None:
    global LATENCY_REPEATS
    args = parse_args()
    setup_logging()
    seed_everything(args.seed)
    LATENCY_REPEATS = max(1, args.latency_repeats)
    if LATENCY_REPEATS > 1:
        logger.info("Latency variance mode: %d scoring repeats per method (float path)", LATENCY_REPEATS)

    cfg = _load_config(args)
    device = get_device(cfg.hardware.device)

    # ---- Data ---------------------------------------------------------------
    embeddings_dir = Path(cfg.data.embeddings_dir)
    if args.debug:
        embeddings_dir = embeddings_dir / "debug"
        if not (embeddings_dir / "query_embeddings.pt").exists():
            create_debug_data(embeddings_dir)

    # --query-embeddings overrides the default query cache (e.g. to evaluate on
    # standard ColBERT query augmentation via query_embeddings_aug.pt without
    # clobbering the non-augmented file). Defaults to the resolved slice cache.
    query_cache = (Path(args.query_embeddings) if args.query_embeddings
                   else resolve_embedding_cache_path(embeddings_dir, "query"))
    print(f"Query embeddings: {query_cache}")
    query_store = EmbeddingStore(query_cache, mode="dict")
    query_store.load()

    doc_store = EmbeddingStore(resolve_embedding_cache_path(embeddings_dir, "doc"), mode="dict")
    doc_store.load()

    # Infer embedding dimension from loaded data
    global FP_DIM
    FP_DIM = doc_store.dim
    logger.info(f"Embedding dimension: {FP_DIM}")

    qrels_path = embeddings_dir / "qrels.tsv"
    if not qrels_path.exists():
        qrels_path = Path(cfg.data.get("dev_qrels", ""))
    qrels = load_qrels(qrels_path)

    doc_ids = doc_store.get_all_ids()
    max_queries = cfg.debug.num_queries if args.debug else args.max_queries

    logger.info(
        f"Loaded {len(doc_ids):,} docs, {len(qrels):,} queries "
        f"(capped at {max_queries if max_queries > 0 else 'all'})"
    )

    # ---- Load trained checkpoints -------------------------------------------
    subbit_models: dict[int, SubBitModel] = {}
    for ckpt_path in args.checkpoints:
        try:
            model = SubBitModel.load(ckpt_path, device=device)
            subbit_models[model.projected_dim] = model
            logger.info(f"Loaded SubBit r={model.projected_dim} from {ckpt_path}")
        except Exception as exc:
            logger.warning(f"Could not load checkpoint {ckpt_path}: {exc}")

    # ---- Derive train-split doc IDs (avoid data leakage in PCA/PQ) ---------
    triples_path = embeddings_dir / "triples.tsv"
    if not triples_path.exists():
        triples_path = Path(cfg.data.get("train_triples", ""))

    train_doc_ids: set[str] | None = None
    if triples_path.exists():
        train_doc_ids = set()
        with open(triples_path) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    train_doc_ids.add(parts[1])  # pos_pid
                    train_doc_ids.add(parts[2])  # neg_pid
        logger.info(f"PCA/PQ will be restricted to {len(train_doc_ids):,} train-split docs "
                     f"(from {triples_path})")
    else:
        logger.warning("No triples file found — PCA/PQ will fit on all docs (potential data leakage)")

    all_results = []

    if args.skip_baselines:
        # Reuse previously-computed baseline rows (FP128 + all unlearned methods).
        # Only the SubBit rows for checkpoints provided this run are re-scored.
        cache_path = Path(args.skip_baselines)
        if not cache_path.exists():
            raise FileNotFoundError(f"--skip-baselines cache not found: {cache_path}")
        with open(cache_path) as f:
            cached = json.load(f)
        # Accept both schemas: old {list of rows} and new {"run_metadata", "rows": [...]}.
        cached_rows = cached.get("rows", cached) if isinstance(cached, dict) else cached
        reused = [row for row in cached_rows if not row["meta"].get("trained", False)]
        logger.info(
            "Reusing %d cached baseline rows from %s (FP128 + unlearned methods). "
            "Only SubBit checkpoints will be re-scored.",
            len(reused), cache_path,
        )
        all_results.extend(reused)
    else:
        # 1. Full Precision Upper Bound (FP128 has float docs — always use float path)
        logger.info("  Computing FP128 Baseline")
        enc_q, enc_d, meta = method_fp128(device)
        fp_rankings, fp_meas = rank_all(
            query_store, doc_store, qrels, enc_q, enc_d, doc_ids, device, max_queries,
            eval_mode="float", binary_docs=False,
            bytes_per_token=meta["bytes_per_token"],
        )
        all_results.append({
            "meta": meta,
            "metrics": compute_metrics(fp_rankings, qrels),
            "measurement": fp_meas.to_dict(),
        })

    for r in args.regimes:
        if r > FP_DIM:
            continue

        if not args.skip_baselines:
            logger.info(f"\n  Computing r={r} Unlearned Baselines")

            # Identity (binary docs)
            logger.info(f"  -> Identity (r={r})")
            enc_q, enc_d, meta = method_identity(r, device)
            rankings, meas = rank_all(
                query_store, doc_store, qrels, enc_q, enc_d, doc_ids, device, max_queries,
                eval_mode=args.eval_mode, binary_docs=True,
                bytes_per_token=meta["bytes_per_token"],
            )
            all_results.append({"meta": meta, "metrics": compute_metrics(rankings, qrels),
                                 "measurement": meas.to_dict()})

            # Random Orthonormal (binary docs)
            logger.info(f"  -> Random Orthogonal Projection (r={r})")
            R_rand = build_random_projection(r, FP_DIM, seed=args.seed)
            enc_q, enc_d, meta = method_random_proj(r, R_rand, device)
            rankings, meas = rank_all(
                query_store, doc_store, qrels, enc_q, enc_d, doc_ids, device, max_queries,
                eval_mode=args.eval_mode, binary_docs=True,
                bytes_per_token=meta["bytes_per_token"],
            )
            all_results.append({"meta": meta, "metrics": compute_metrics(rankings, qrels),
                                 "measurement": meas.to_dict()})

            # PCA (binary docs)
            logger.info(f"  -> PCA (r={r})")
            R_pca, mean_pca = build_pca_projection(doc_store, r, args.pca_sample_size, train_doc_ids)
            enc_q, enc_d, meta = method_pca_proj(r, R_pca, mean_pca, device)
            rankings, meas = rank_all(
                query_store, doc_store, qrels, enc_q, enc_d, doc_ids, device, max_queries,
                eval_mode=args.eval_mode, binary_docs=True,
                bytes_per_token=meta["bytes_per_token"],
            )
            all_results.append({"meta": meta, "metrics": compute_metrics(rankings, qrels),
                                 "measurement": meas.to_dict()})

        # Trained SubBit (binary docs) — always re-scored.
        if r in subbit_models:
            logger.info(f"  SubBit r={r} (trained)")
            enc_q, enc_d, meta = method_subbit(subbit_models[r], r, device)
            rankings, meas = rank_all(
                query_store, doc_store, qrels, enc_q, enc_d,
                doc_ids, device, max_queries,
                eval_mode=args.eval_mode, binary_docs=True,
                bytes_per_token=meta["bytes_per_token"],
            )
            metrics = compute_metrics(rankings, qrels)
            all_results.append({"meta": meta, "metrics": metrics, "measurement": meas.to_dict()})
            logger.info(
                f"    MRR@10={metrics['mrr@10']:.4f}  NDCG@10={metrics['ndcg@10']:.4f}  "
                f"mean-lat={meas.scoring_latency.mean_ms:.2f}ms  "
                f"peak-mem={meas.memory.device_peak_mb or 0:.0f}MB"
            )

            if args.subbit_ablate_scale and getattr(subbit_models[r], "use_scale", False):
                logger.info(f"  SubBit r={r} (no scale) — ablation")
                original_use_scale = subbit_models[r].use_scale
                subbit_models[r].use_scale = False
                try:
                    enc_q_ns, enc_d_ns, meta_ns = method_subbit(subbit_models[r], r, device)
                    meta_ns = {**meta_ns, "label": f"SubBit r={r} (no scale)", "use_scale": False}
                    rankings_ns, meas_ns = rank_all(
                        query_store, doc_store, qrels, enc_q_ns, enc_d_ns,
                        doc_ids, device, max_queries,
                        eval_mode=args.eval_mode, binary_docs=True,
                        bytes_per_token=meta_ns["bytes_per_token"],
                    )
                    metrics_ns = compute_metrics(rankings_ns, qrels)
                    all_results.append({"meta": meta_ns, "metrics": metrics_ns,
                                          "measurement": meas_ns.to_dict()})
                    logger.info(
                        f"    MRR@10={metrics_ns['mrr@10']:.4f}  "
                        f"NDCG@10={metrics_ns['ndcg@10']:.4f}  "
                        f"Δ MRR@10 = {metrics_ns['mrr@10'] - metrics['mrr@10']:+.4f}"
                    )
                finally:
                    subbit_models[r].use_scale = original_use_scale

    # ---- Optional: PLAID-style baseline -------------------------------------
    # Runs on full 128d with centroid + per-dim residual quantization; fit once
    # on train-split tokens, then one row per residual_bits value.
    if args.include_plaid:
        logger.info("\n  Computing PLAID-style Baseline")
        plaid_sample_ids = train_doc_ids if train_doc_ids is not None else None
        plaid_tokens = doc_store.sample_embeddings(args.plaid_kmeans_sample, ids=plaid_sample_ids)
        plaid_tokens = plaid_tokens.to(device)

        # Centroids only depend on (tokens, seed, C, kmeans_iters) — fit once and
        # share across the residual_bits sweep. Only bucket edges/centers change with b.
        shared_centroids: torch.Tensor | None = None
        for b in args.plaid_residual_bits:
            cfg_plaid = PLAIDConfig(
                n_centroids=args.plaid_centroids,
                residual_bits=b,
                kmeans_sample=args.plaid_kmeans_sample,
                seed=args.seed,
            )
            logger.info(
                "  -> PLAID fit (C=%d, b=%d, %d sampled tokens)%s",
                cfg_plaid.n_centroids, b, plaid_tokens.shape[0],
                " [sharing centroids]" if shared_centroids is not None else "",
            )
            pq_plaid = PLAIDQuantizer(d=FP_DIM, config=cfg_plaid)
            pq_plaid.train(plaid_tokens, centroids=shared_centroids)
            if shared_centroids is None:
                shared_centroids = pq_plaid.centroids

            enc_q, enc_d, meta = method_plaid(pq_plaid, device)
            rankings, meas = rank_all(
                query_store, doc_store, qrels, enc_q, enc_d, doc_ids, device, max_queries,
                eval_mode="float", binary_docs=False,  # float reconstruction path
                bytes_per_token=meta["bytes_per_token"],
            )
            metrics = compute_metrics(rankings, qrels)
            all_results.append({"meta": meta, "metrics": metrics, "measurement": meas.to_dict()})
            logger.info(
                "    PLAID b=%d: MRR@10=%.4f  NDCG@10=%.4f  Recall@1k=%.4f  bytes/tok=%.1f",
                b, metrics["mrr@10"], metrics["ndcg@10"], metrics["recall@1000"],
                meta["bytes_per_token_exact"],
            )

    # ---- Optional: OPQ baseline (faiss) -------------------------------------
    # Optimised PQ: a faiss-learned orthogonal rotation before product
    # quantization, at the same byte budget as plain PQ. Also runs a plain
    # faiss-PQ control that should reproduce the repo Torch-PQ row (sanity that
    # the rotation, not the PQ implementation, is what moves the number).
    if args.include_opq:
        if faiss is None:
            logger.warning("faiss not installed; skipping --include-opq (pip install faiss-cpu)")
        else:
            # faiss-cpu and torch each bundle their own OpenMP runtime; letting
            # faiss spawn a thread team segfaults on macOS. Pin faiss to 1 thread.
            faiss.omp_set_num_threads(1)
            logger.info("\n  Computing OPQ Baseline (faiss, single-threaded)")
            M, NBITS = args.opq_subquantizers, 8
            opq_ids = train_doc_ids if train_doc_ids is not None else None
            opq_train = doc_store.sample_embeddings(args.opq_train_sample, ids=opq_ids)
            opq_train = opq_train.detach().cpu().numpy().astype("float32")

            opq_index = faiss.IndexPreTransform(
                faiss.OPQMatrix(FP_DIM, M), faiss.IndexPQ(FP_DIM, M, NBITS))
            logger.info("  -> OPQ fit (M=%d, %d-bit, %d train tokens)...",
                        M, NBITS, opq_train.shape[0])
            opq_index.train(opq_train)
            enc_q, enc_d, meta = method_opq(opq_index, M, device)
            meta["label"] = f"OPQ {M}×{NBITS}, asym"
            rankings, meas = rank_all(
                query_store, doc_store, qrels, enc_q, enc_d, doc_ids, device, max_queries,
                eval_mode="float", binary_docs=False, bytes_per_token=meta["bytes_per_token"])
            metrics = compute_metrics(rankings, qrels)
            all_results.append({"meta": meta, "metrics": metrics, "measurement": meas.to_dict()})
            logger.info("    OPQ %dx%d: MRR@10=%.4f  NDCG@10=%.4f  Recall@1k=%.4f",
                        M, NBITS, metrics["mrr@10"], metrics["ndcg@10"], metrics["recall@1000"])

            pq_index = faiss.IndexPQ(FP_DIM, M, NBITS)
            logger.info("  -> faiss PQ control fit (M=%d, %d-bit)...", M, NBITS)
            pq_index.train(opq_train)
            enc_q2, enc_d2, meta2 = method_opq(pq_index, M, device)
            meta2["label"] = f"PQ {M}×{NBITS} (faiss control), asym"
            rankings2, meas2 = rank_all(
                query_store, doc_store, qrels, enc_q2, enc_d2, doc_ids, device, max_queries,
                eval_mode="float", binary_docs=False, bytes_per_token=meta2["bytes_per_token"])
            metrics2 = compute_metrics(rankings2, qrels)
            all_results.append({"meta": meta2, "metrics": metrics2, "measurement": meas2.to_dict()})
            logger.info("    PQ(faiss) %dx%d: MRR@10=%.4f  (repo Torch-PQ ref 0.764)",
                        M, NBITS, metrics2["mrr@10"])

    # ---- Print summary table ------------------------------------------------
    print_table(all_results)

    # ---- Save JSON results --------------------------------------------------
    output_path = args.output or str(
        ensure_dir(Path(cfg.paths.results_dir)) / "baseline_comparison.json"
    )
    # Ensure the parent directory exists for user-supplied --output paths too.
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Reviewer-ready payload: rows + run metadata so every result is
    # self-describing (git SHA, torch/cuda versions, eval device, etc).
    payload = {
        "run_metadata": collect_run_metadata(
            config={
                "eval_mode": args.eval_mode,
                "regimes": list(args.regimes),
                "max_queries": args.max_queries,
                "pca_sample_size": args.pca_sample_size,
                "seed": args.seed,
                "checkpoints": args.checkpoints,
                "config_path": args.config,
                "skip_baselines": args.skip_baselines,
                "include_plaid": args.include_plaid,
                "plaid_centroids": args.plaid_centroids,
                "plaid_residual_bits": args.plaid_residual_bits,
                "plaid_kmeans_sample": args.plaid_kmeans_sample,
            }
        ),
        "rows": [
            {"meta": r["meta"], "metrics": r["metrics"], "measurement": r.get("measurement")}
            for r in all_results
        ],
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
