"""Preservation matrix: learned SubBit R vs random / PCA / identity projections.

Answers "is the learning load-bearing for structure preservation, or does any
projection + sign() do the job?" Runs the same Spearman / Kendall / top-k
overlap / Pearson diagnostics on the 100k augmented corpus for:

    * learned R     (from the trained r=64 checkpoint)
    * random R      (orthogonal init, no training)
    * PCA R         (PCA on doc tokens, no training)
    * identity      (keep first r dims, then binarise)
    * [optional]    learned R in symmetric mode, the paper's efficiency mode

Each row of the output is the same metric set used in the main rank-
preservation analysis, so the table is directly comparable to Figure X.

Outputs:
    outputs/aug_eval/preservation_aug_rest.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.subbit.baselines import (
    identity_truncation,
    pca_projection_binary,
    random_projection_binary,
)
from src.subbit.data import EmbeddingStore, resolve_embedding_cache_path
from src.subbit.model import SubBitModel
from src.subbit.rank_preservation import (
    PairwiseScores,
    compute_pairwise_scores,
    compute_rank_preservation,
)
from src.subbit.utils import ensure_dir, get_device, seed_everything, setup_logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Encoder factories — each produces (encode_doc, encode_query) pair that
# matches the SubBitModel interface used by compute_pairwise_scores.
# ---------------------------------------------------------------------------


def _learned_encoders(model: SubBitModel, symmetric: bool):
    def encode_doc(x):  # (n, 128) → (n, r) ±1
        return model.encode_document(x, use_ste=False)

    def encode_query(x):  # (m, 128) → (m, r)  float if asymmetric, ±1 if symmetric
        return model.encode_query(x, symmetric=symmetric, use_ste=False)

    return encode_doc, encode_query


def _random_encoders(r: int, d_in: int, device: torch.device, seed: int = 42):
    # MPS doesn't implement aten::linalg_qr.out — keep QR on CPU and
    # migrate the resulting R to the requested device.
    torch.manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(d_in, r))  # CPU
    R = Q.T.contiguous().to(device)  # (r, d_in)

    def encode_doc(x):
        proj = x @ R.T
        out = torch.sign(proj)
        out[out == 0] = 1.0
        return out

    def encode_query(x):
        return x @ R.T  # fp32, asymmetric

    return encode_doc, encode_query, R


def _pca_encoders(r: int, fit_tokens: torch.Tensor, device: torch.device):
    _, R, _ = pca_projection_binary(fit_tokens, projected_dim=r)  # (result, R, mean)
    R = R.to(device)

    def encode_doc(x):
        proj = x @ R.T
        out = torch.sign(proj)
        out[out == 0] = 1.0
        return out

    def encode_query(x):
        return x @ R.T

    return encode_doc, encode_query, R


def _identity_encoders(r: int):
    def encode_doc(x):
        return identity_truncation(x, projected_dim=r)

    def encode_query(x):
        return x[..., :r]  # no projection, just truncate — query in "fp32 sliced" form

    return encode_doc, encode_query


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _run_one(
    label: str,
    encode_doc,
    encode_query,
    query_store,
    doc_store,
    query_ids,
    doc_ids,
    device,
    k_values,
) -> dict:
    logger.info("=== %s ===", label)
    scores: PairwiseScores = compute_pairwise_scores(
        query_store, doc_store,
        query_ids=query_ids,
        doc_ids=doc_ids,
        encode_query=encode_query,
        encode_document=encode_doc,
        device=device,
        progress=True,
    )
    rep = compute_rank_preservation(scores, k_values=k_values, keep_per_query=True)
    row = {
        "label": label,
        "spearman_mean": rep.spearman_rho["mean"],
        "spearman_std": rep.spearman_rho["std"],
        "kendall_mean": rep.kendall_tau["mean"],
        "pearson_flat": rep.score_correlation_pearson,
        "overlap_at_k": {str(k): rep.topk_overlap[k]["mean"] for k in sorted(rep.topk_overlap.keys())},
        "n_queries": rep.spearman_rho["n"],
        # per-query Spearman ρ (qid -> ρ) for the Δρ-vs-ΔMRR causal test in §8.3.
        "per_query_spearman": {str(q): float(v["spearman_rho"]) for q, v in rep.per_query.items()},
    }
    logger.info(
        "  Spearman ρ=%.4f  Kendall τ=%.4f  Pearson=%.4f  "
        "overlap@{10,100,1000}=%.3f/%.3f/%.3f",
        row["spearman_mean"], row["kendall_mean"], row["pearson_flat"],
        row["overlap_at_k"].get("10", float("nan")),
        row["overlap_at_k"].get("100", float("nan")),
        row["overlap_at_k"].get("1000", float("nan")),
    )
    return row


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--embeddings-dir", type=Path, default=Path("data/embeddings/msmarco/100k_aug"))
    p.add_argument("--checkpoint", type=Path, default=Path("artifacts/checkpoints/50k_topk/best.pt"))
    p.add_argument("--r", type=int, default=64)
    p.add_argument("--output", type=Path, default=Path("outputs/aug_eval/preservation_aug_rest.json"))
    p.add_argument("--k-values", type=int, nargs="+", default=[10, 100, 1000])
    p.add_argument("--max-queries", type=int, default=-1)
    p.add_argument("--max-docs", type=int, default=-1)
    p.add_argument("--pca-fit-tokens", type=int, default=200_000,
                   help="How many doc tokens to fit PCA on (CPU, sklearn).")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-scale", action="store_true",
                   help="Disable the learned query-side scale head (for a "
                        "strictly no-scale comparison against random/identity).")
    p.add_argument(
        "--variants",
        type=str,
        nargs="+",
        default=None,
        help="Restrict to a subset of {learned_asymmetric, learned_symmetric, "
             "random_projection, pca_projection, identity_truncation}. "
             "If omitted, runs all five.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    seed_everything(args.seed)
    device = torch.device(args.device) if args.device else get_device("auto")
    logger.info("device=%s", device)

    query_store = EmbeddingStore(resolve_embedding_cache_path(args.embeddings_dir, "query"), mode="dict")
    query_store.load()
    doc_store = EmbeddingStore(resolve_embedding_cache_path(args.embeddings_dir, "doc"), mode="dict")
    doc_store.load()

    from src.subbit.data import load_qrels
    qrels = load_qrels(args.embeddings_dir / "qrels.tsv")
    known_qids = set(query_store.get_all_ids())
    query_ids = [q for q in qrels.keys() if q in known_qids]
    if 0 < args.max_queries < len(query_ids):
        query_ids = query_ids[: args.max_queries]
    doc_ids = doc_store.get_all_ids()
    if 0 < args.max_docs < len(doc_ids):
        doc_ids = doc_ids[: args.max_docs]
    logger.info("queries=%d  docs=%d", len(query_ids), len(doc_ids))

    # Gather a PCA fit sample from raw doc tokens.
    logger.info("Sampling %d doc tokens for PCA fit …", args.pca_fit_tokens)
    tokens = []
    total = 0
    for did in doc_ids:
        try:
            d = doc_store.get(did)
        except (KeyError, FileNotFoundError):
            continue
        tokens.append(d)
        total += d.shape[0]
        if total >= args.pca_fit_tokens:
            break
    fit_tokens = torch.cat(tokens, dim=0)
    if fit_tokens.shape[0] > args.pca_fit_tokens:
        gen = torch.Generator().manual_seed(0)
        perm = torch.randperm(fit_tokens.shape[0], generator=gen)[: args.pca_fit_tokens]
        fit_tokens = fit_tokens[perm]
    logger.info("PCA fit sample: %s", tuple(fit_tokens.shape))

    # Load the learned model.
    logger.info("Loading learned checkpoint: %s", args.checkpoint)
    model = SubBitModel.load(str(args.checkpoint), device=device)
    model.to(device).eval()
    if args.no_scale and model.use_scale:
        # Strip the query-side scale head so the learned-R row is strictly
        # comparable to the no-scale random/identity/PCA rows (the head is a
        # positive per-token scalar: argmax-invariant, but it reweights the
        # per-query MaxSim sum and hence document rankings).
        model.use_scale = False
        logger.info("--no-scale: disabled query-side scale head for learned R")

    # ---- Run each variant -----------------------------------------------
    # Lazy encoder factories so we only build what we need (PCA fit etc.).
    def make_encoders(name: str):
        if name == "learned_asymmetric":
            ed, eq = _learned_encoders(model, symmetric=False)
        elif name == "learned_symmetric":
            ed, eq = _learned_encoders(model, symmetric=True)
        elif name == "random_projection":
            ed, eq, _ = _random_encoders(args.r, d_in=128, device=device, seed=args.seed)
        elif name == "pca_projection":
            ed, eq, _ = _pca_encoders(args.r, fit_tokens=fit_tokens, device=device)
        elif name == "identity_truncation":
            ed, eq = _identity_encoders(args.r)
        else:
            raise ValueError(f"unknown variant: {name}")
        return ed, eq

    ALL_VARIANTS = [
        "learned_asymmetric",
        "learned_symmetric",
        "random_projection",
        "pca_projection",
        "identity_truncation",
    ]
    variants = args.variants if args.variants else ALL_VARIANTS
    unknown = [v for v in variants if v not in ALL_VARIANTS]
    if unknown:
        raise ValueError(f"unknown variants: {unknown}; valid: {ALL_VARIANTS}")

    # Merge with any existing rows (resume-friendly).
    ensure_dir(args.output.parent)
    rows: list[dict] = []
    if args.output.exists():
        try:
            prior = json.load(open(args.output))
            rows = [r for r in prior.get("rows", []) if r.get("label") not in variants]
            logger.info("Resuming: %d prior rows kept, recomputing %d variants", len(rows), len(variants))
        except Exception as exc:
            logger.warning("Could not parse prior %s (%s); starting fresh", args.output, exc)
            rows = []

    def _persist():
        payload = {
            "config": {
                "embeddings_dir": str(args.embeddings_dir),
                "checkpoint": str(args.checkpoint),
                "r": args.r,
                "k_values": list(args.k_values),
                "max_queries": args.max_queries,
                "max_docs": args.max_docs,
                "pca_fit_tokens": int(fit_tokens.shape[0]),
                "device": str(device),
            },
            "rows": rows,
        }
        with open(args.output, "w") as f:
            json.dump(payload, f, indent=2)

    for name in variants:
        enc_d, enc_q = make_encoders(name)
        row = _run_one(name, enc_d, enc_q,
                       query_store, doc_store, query_ids, doc_ids, device, args.k_values)
        rows.append(row)
        _persist()  # incremental — any crash past this point keeps completed rows.
        logger.info("  [persisted %d/%d rows to %s]", len(rows), len(rows) + (len(variants) - variants.index(name) - 1), args.output)

        # Drop per-variant padded tensors from the MPS allocator between
        # variants — otherwise the 100k padded tensor (~13 GB) accumulates
        # across variants and triggers MPS OOM ~3 variants in.
        if device.type == "mps":
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
        elif device.type == "cuda":
            torch.cuda.empty_cache()

    logger.info("Wrote %s", args.output)

    # Pretty summary.
    logger.info("")
    logger.info("=== Preservation matrix (r=%d) ===", args.r)
    ks = list(args.k_values)
    hdr = ["method", "Spearman ρ", "Kendall τ", "Pearson"] + [f"overlap@{k}" for k in ks]
    logger.info("  %-25s  %s", hdr[0], "  ".join(f"{h:>10s}" for h in hdr[1:]))
    for r in rows:
        ovs = [f"{r['overlap_at_k'].get(str(k), float('nan')):>10.4f}" for k in ks]
        logger.info(
            "  %-25s  %10.4f  %10.4f  %10.4f  %s",
            r["label"], r["spearman_mean"], r["kendall_mean"], r["pearson_flat"],
            "  ".join(ovs),
        )


if __name__ == "__main__":
    main()
