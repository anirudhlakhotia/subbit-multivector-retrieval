#!/usr/bin/env python3
"""Interleaved, repeated latency benchmark — trustworthy Table 7 numbers.

Why this script exists
----------------------
The paper's e2e latency table (`outputs/latency/cpu_latency_full_c32768.json`)
reports per-projection latencies for four r=64 rows that perform IDENTICAL
compute (same projected-query shape, same sign-code shape, same float einsum):

    trained 70.7 ms | pca 83.3 ms | identity 91.0 ms | random 107.7 ms

A 52% spread across identical-compute rows is a measurement artifact, not an
effect. Repeated smoke runs found the cause: per-pass MEANS swing ~60% pass-to-pass (a handful of
stalls dominate), while per-query MEDIANS are stable within ~2% of the
canonical anchor. The canonical bench also runs methods as sequential blocks,
so cold caches hit the first rows and thermal throttle hits the last.

Fixes implemented here
----------------------
1. INTERLEAVED SCHEDULE: every method's corpus is built once, then timed
   passes run in rounds; within each round the method order is shuffled
   (seeded). Thermal drift and background load hit every method equally.
2. MEDIAN-PRIMARY: the per-pass statistic is the per-query MEDIAN (stall
   robust); the headline per method is the median of per-round medians with
   IQR across rounds. Means are recorded but not headline.
3. WARM-UP ROUND: one untimed round before any timing (page cache, kernel
   compile, allocator steady state).
4. ENVIRONMENT CAPTURE: power source, thread count, device, git metadata
   recorded; refuses to start on battery power unless --allow-battery.
5. AUG-CONVENTION QUERIES by default (`query_embeddings_aug.pt`, 32 tokens),
   matching the paper's augmented re-baseline. Latency scales with query
   tokens, so the old ~8-token numbers do not carry over.
6. VERDICT BLOCK: for the projections group, reports whether the max pairwise
   gap between projection medians exceeds the within-method across-round IQR.
   If it does not, the four rows are statistically indistinguishable and the
   paper should collapse them into one "sign-coded r=64" row.

Memory plan (why two groups)
----------------------------
Interleaving requires all corpora of a group resident at once. Groups bound
peak RSS on the 48 GB M4 Pro:
  --group projections : four r=64 float corpora (trained / random / identity /
                        pca) ~10-12 GB total. Answers "are the four rows equal?"
  --group table       : sign-coded random r=64 + PLAID b in {1,2,4} (C=32768,
                        reconstructed 128-dim floats) + FP128 ~20-24 GB.
                        Produces the actual Table 7 replacement numbers.
Run them back-to-back, not simultaneously, on an otherwise idle box.

Usage
-----
    # Idle box, plugged in. caffeinate blocks sleep/low-power throttling.
    caffeinate -i python latency/bench_latency_interleaved.py \
        --group projections --rounds 5 --max-queries 500 \
        2>&1 | tee /tmp/latency_interleaved_projections.log

    caffeinate -i python latency/bench_latency_interleaved.py \
        --group table --rounds 5 --max-queries 500 \
        2>&1 | tee /tmp/latency_interleaved_table.log

    # Smoke (validates plumbing, ~minutes):
    python latency/bench_latency_interleaved.py --group projections \
        --rounds 2 --max-queries 50 --output /tmp/lat_smoke.json

Timing boundary is identical to the canonical bench and W4: the query tensor
is fetched and moved per query OUTSIDE the measured block; the measured block
is encode_query + float einsum MaxSim over the prebuilt corpus + top-k.
Scoring helpers mirror evaluation/run_baseline_comparison.py's float branch,
so numbers trace to the same provenance chain as the latency tables.
"""
from __future__ import annotations

import argparse
import json
import logging
import random as pyrandom
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.subbit.baselines_plaid import PLAIDConfig, PLAIDQuantizer, method_plaid
from src.subbit.data import EmbeddingStore, load_qrels, resolve_embedding_cache_path
from src.subbit.measurement import collect_run_metadata
from src.subbit.model import SubBitModel
from src.subbit.utils import ensure_dir, seed_everything, setup_logging

logger = logging.getLogger(__name__)

DOC_BATCH_SIZE = 25_000  # matches run_baseline_comparison's float-path chunking
_TTY = sys.stdout.isatty()


# ---------------------------------------------------------------------------
# Corpus builder + scorer — copied from bench_latency_variance.py (which copied
# run_baseline_comparison.rank_all's float branch) so timed work is canonical.
# ---------------------------------------------------------------------------

def build_float_corpus(doc_store, doc_ids, encode_doc, device):
    encoded: dict[str, torch.Tensor] = {}
    for pid in tqdm(doc_ids, desc="  encode docs", leave=False, disable=not _TTY):
        try:
            d = doc_store.get(pid).to(device)
        except (KeyError, FileNotFoundError):
            continue
        with torch.no_grad():
            encoded[pid] = encode_doc(d)
    valid = sorted(encoded.keys(), key=lambda pid: encoded[pid].shape[0])
    batched_vecs, batched_masks = [], []
    for i in range(0, len(valid), DOC_BATCH_SIZE):
        ids = valid[i:i + DOC_BATCH_SIZE]
        max_len = max(encoded[p].shape[0] for p in ids)
        dim = encoded[ids[0]].shape[1]
        vecs = torch.zeros(len(ids), max_len, dim, device=device)
        mask = torch.zeros(len(ids), max_len, dtype=torch.bool, device=device)
        for j, pid in enumerate(ids):
            d = encoded[pid]
            vecs[j, :d.shape[0], :] = d
            mask[j, :d.shape[0]] = True
        batched_vecs.append(vecs)
        batched_masks.append(mask)
    del encoded
    return batched_vecs, batched_masks


def make_float_scorer(encode_query, batched_vecs, batched_masks, top_k):
    def score_one(q_embs: torch.Tensor, qid=None) -> None:
        with torch.no_grad():
            q_enc = encode_query(q_embs)
            all_scores = []
            for doc_vecs, doc_mask in zip(batched_vecs, batched_masks):
                sim = torch.einsum("md,cnd->cmn", q_enc, doc_vecs)
                sim = sim.masked_fill(~doc_mask.unsqueeze(1), float("-inf"))
                all_scores.append(sim.max(dim=-1).values.sum(dim=-1))
            scores = torch.cat(all_scores, dim=0)
            torch.topk(scores, k=min(top_k, scores.shape[0]))
    return score_one


# ---------------------------------------------------------------------------
# Projection encode closures (untrained rows: latency depends on shapes only)
# ---------------------------------------------------------------------------

def make_proj_closures(R: torch.Tensor, device: torch.device):
    """Sign-coded asymmetric closures for a fixed projection R (r x 128)."""
    Rt = R.to(device).T.contiguous()  # (128, r)

    def enc_q(e: torch.Tensor) -> torch.Tensor:
        return e.to(device) @ Rt

    def enc_d(e: torch.Tensor) -> torch.Tensor:
        return torch.sign(e.to(device) @ Rt)

    return enc_q, enc_d


def random_orthogonal_R(r: int, dim: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(dim, dim, generator=g)
    q, _ = torch.linalg.qr(a)
    return q[:r, :]


def pca_R(doc_store, r: int, sample: int, seed: int) -> torch.Tensor:
    toks = doc_store.sample_embeddings(sample).float()
    toks = toks - toks.mean(dim=0, keepdim=True)
    # values are latency-irrelevant (only shapes time); SVD keeps the row honest
    _, _, vt = torch.linalg.svd(toks, full_matrices=False)
    return vt[:r, :].contiguous()


# ---------------------------------------------------------------------------
# Interleaved timing driver
# ---------------------------------------------------------------------------

def timed_pass(score_one, query_store, qids, device) -> dict:
    """One full pass over qids; returns per-query latency stats for the pass."""
    lat_ms = np.empty(len(qids), dtype=np.float64)
    for i, qid in enumerate(qids):
        q = query_store.get(qid).to(device)   # outside the timed region (canonical boundary)
        t0 = time.perf_counter_ns()
        score_one(q, qid)
        lat_ms[i] = (time.perf_counter_ns() - t0) / 1e6
    return {
        "median_ms": float(np.median(lat_ms)),
        "mean_ms": float(lat_ms.mean()),
        "p95_ms": float(np.percentile(lat_ms, 95)),
        "max_ms": float(lat_ms.max()),
    }


def environment_snapshot() -> dict:
    def sh(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout.strip()
        except Exception:
            return "unavailable"
    return {
        "power": sh(["pmset", "-g", "batt"]).splitlines()[0] if sh(["pmset", "-g", "batt"]) else "unavailable",
        "thermal": sh(["pmset", "-g", "therm"]),
        "torch_threads": torch.get_num_threads(),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--group", choices=["projections", "table", "rerank"], required=True)
    p.add_argument("--rerank-cutoffs", type=int, nargs="+", default=[100, 256, 1024],
                   help="Stage-2 fp32 rerank depths K (rerank group only).")
    p.add_argument("--rounds", type=int, default=5, help="Timed rounds (after 1 warm-up round).")
    p.add_argument("--max-queries", type=int, default=500,
                   help="Seeded query subsample shared by all methods/rounds (-1 = all 6,980).")
    p.add_argument("--query-embeddings", default="data/embeddings/msmarco/100k/query_embeddings_aug.pt",
                   help="Aug 32-token artifact by default (paper convention). Pass the non-aug "
                        "query_embeddings.pt for a quick projection-equality diagnosis.")
    p.add_argument("--embeddings-dir", default="data/embeddings/msmarco/100k")
    p.add_argument("--checkpoint", default="artifacts/checkpoints/50k_topk/best.pt")
    p.add_argument("--r", type=int, default=64)
    p.add_argument("--top-k", type=int, default=1000)
    p.add_argument("--threads", type=int, default=8, help="torch CPU threads (paper convention: 8).")
    p.add_argument("--device", default="cpu", help="Paper Table 7 convention is cpu.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pca-sample", type=int, default=50_000)
    p.add_argument("--plaid-centroids", type=int, default=32_768, help="Paper Table 7 value.")
    p.add_argument("--plaid-residual-bits", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--plaid-kmeans-sample", type=int, default=1_000_000)
    p.add_argument("--allow-battery", action="store_true")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    setup_logging()
    seed_everything(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)

    env = environment_snapshot()
    logger.info("Environment: %s", env)
    if "AC Power" not in env.get("power", "") and not args.allow_battery:
        logger.error("Not on AC power (%s). Plug in or pass --allow-battery.", env.get("power"))
        sys.exit(2)

    out_path = Path(args.output or f"outputs/latency/latency_interleaved_{args.group}.json")

    # ---- data ----------------------------------------------------------------
    emb_dir = Path(args.embeddings_dir)
    doc_store = EmbeddingStore(resolve_embedding_cache_path(emb_dir, "doc"), mode="dict")
    doc_store.load()
    query_store = EmbeddingStore(Path(args.query_embeddings), mode="dict")
    query_store.load()
    qrels = load_qrels(emb_dir / "qrels.tsv")
    doc_ids = doc_store.get_all_ids()
    fp_dim = doc_store.dim

    all_qids = sorted(qrels.keys())
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(all_qids))
    qids = [all_qids[i] for i in perm]
    if 0 < args.max_queries < len(qids):
        qids = qids[:args.max_queries]
    qids = [q for q in qids if _has(query_store, q)]
    n_qtok = int(query_store.get(qids[0]).shape[0])
    logger.info("Group=%s | %d docs | %d queries (%d tokens each) | device=%s threads=%d",
                args.group, len(doc_ids), len(qids), n_qtok, device, args.threads)

    # ---- build all corpora for the group (ONCE) -------------------------------
    methods: dict[str, Callable] = {}
    meta: dict[str, dict] = {}

    if args.group == "projections":
        model = SubBitModel.load(args.checkpoint, device=device).to(device).eval()

        def t_enc_q(e):
            with torch.no_grad():
                return model.encode_query(e.to(device))

        def t_enc_d(e):
            with torch.no_grad():
                return model.encode_document(e.to(device))

        specs = [
            ("trained r=64", t_enc_q, t_enc_d,
             {"projection": "trained", "checkpoint": args.checkpoint}),
        ]
        R_rand = random_orthogonal_R(args.r, fp_dim, args.seed)
        q1, d1 = make_proj_closures(R_rand, device)
        specs.append(("random r=64", q1, d1, {"projection": "random_orthogonal", "seed": args.seed}))
        R_id = torch.zeros(args.r, fp_dim); R_id[:, :args.r] = torch.eye(args.r)
        q2, d2 = make_proj_closures(R_id, device)
        specs.append(("identity r=64", q2, d2, {"projection": "identity_truncation"}))
        R_p = pca_R(doc_store, args.r, args.pca_sample, args.seed)
        q3, d3 = make_proj_closures(R_p, device)
        specs.append(("pca r=64", q3, d3, {"projection": "pca", "sample": args.pca_sample}))

        for label, eq, ed, m in specs:
            logger.info("Building corpus: %s", label)
            vecs, masks = build_float_corpus(doc_store, doc_ids, ed, device)
            methods[label] = make_float_scorer(eq, vecs, masks, args.top_k)
            meta[label] = {**m, "r": args.r, "eval_mode": "float"}

    elif args.group == "table":
        R_rand = random_orthogonal_R(args.r, fp_dim, args.seed)
        eq, ed = make_proj_closures(R_rand, device)
        logger.info("Building corpus: sign-coded r=64 (random R; projection-independent timing)")
        vecs, masks = build_float_corpus(doc_store, doc_ids, ed, device)
        methods["sign-coded r=64"] = make_float_scorer(eq, vecs, masks, args.top_k)
        meta["sign-coded r=64"] = {"projection": "random_orthogonal", "r": args.r,
                                   "eval_mode": "float", "bytes_per_token": args.r // 8}

        plaid_tokens = doc_store.sample_embeddings(args.plaid_kmeans_sample).to(device)
        shared_centroids = None
        for b in args.plaid_residual_bits:
            cfgp = PLAIDConfig(n_centroids=args.plaid_centroids, residual_bits=b,
                               kmeans_sample=args.plaid_kmeans_sample, seed=args.seed)
            logger.info("PLAID fit (C=%d, b=%d)%s", args.plaid_centroids, b,
                        " [shared centroids]" if shared_centroids is not None else "")
            pq = PLAIDQuantizer(d=fp_dim, config=cfgp)
            pq.train(plaid_tokens, centroids=shared_centroids)
            if shared_centroids is None:
                shared_centroids = pq.centroids
            enq, end, m = method_plaid(pq, device)
            vecs, masks = build_float_corpus(doc_store, doc_ids, end, device)
            del pq
            label = f"PLAID (C={args.plaid_centroids}, b={b})"
            methods[label] = make_float_scorer(enq, vecs, masks, args.top_k)
            meta[label] = {**m, "eval_mode": "float"}
        del plaid_tokens, shared_centroids

        logger.info("Building corpus: FP128")
        ident = lambda e: e.to(device)
        vecs, masks = build_float_corpus(doc_store, doc_ids, ident, device)
        methods["FP128"] = make_float_scorer(ident, vecs, masks, args.top_k)
        meta["FP128"] = {"r": fp_dim, "eval_mode": "float", "bytes_per_token": fp_dim * 4}

    else:  # rerank — two-stage pipeline, per-stage latency (float, no kernels)
        # stage 1 (candidate generation): sign-coded random-R full scan -> top-maxK
        # stage 2 (rerank, TIMED per K): fp32 MaxSim over the top-K candidates.
        # The sign-coded and FP128 full-scan rows are timed too, so the stage-1
        # cost and the fp32-exhaustive reference come from the same interleaved run.
        R_rand = random_orthogonal_R(args.r, fp_dim, args.seed)
        Rt = R_rand.to(device).T.contiguous()
        enc_q_sign, _ = make_proj_closures(R_rand, device)
        ident = lambda e: e.to(device)

        logger.info("Caching fp32 docs (for stage-2 gather)...")
        fp_by_pid: dict[str, torch.Tensor] = {}
        for pid in tqdm(doc_ids, desc="  cache fp32", leave=False, disable=not _TTY):
            try:
                fp_by_pid[pid] = doc_store.get(pid).to(device)
            except (KeyError, FileNotFoundError):
                continue
        # length-sorted pid order shared by the batched corpora and the index->pid map
        pid_list = sorted(fp_by_pid.keys(), key=lambda p: fp_by_pid[p].shape[0])

        def build_batched(encode, dim):
            V, M = [], []
            for i in range(0, len(pid_list), DOC_BATCH_SIZE):
                ids = pid_list[i:i + DOC_BATCH_SIZE]
                mxl = max(fp_by_pid[p].shape[0] for p in ids)
                v = torch.zeros(len(ids), mxl, dim, device=device)
                m = torch.zeros(len(ids), mxl, dtype=torch.bool, device=device)
                for j, p in enumerate(ids):
                    with torch.no_grad():
                        e = encode(fp_by_pid[p])
                    v[j, :e.shape[0]] = e
                    m[j, :e.shape[0]] = True
                V.append(v); M.append(m)
            return V, M

        logger.info("Building corpus: sign-coded r=64 (stage 1 / candidate generation)")
        sgn_v, sgn_m = build_batched(lambda e: torch.sign(e @ Rt), args.r)
        methods["stage1 sign-coded full scan"] = make_float_scorer(enc_q_sign, sgn_v, sgn_m, args.top_k)
        meta["stage1 sign-coded full scan"] = {"projection": "random_orthogonal", "r": args.r,
                                               "eval_mode": "float", "bytes_per_token": args.r // 8,
                                               "stage": "candidate_generation"}

        logger.info("Building corpus: FP128 (fp32 exhaustive reference)")
        fp_v, fp_m = build_batched(ident, fp_dim)
        methods["fp32 full scan (reference)"] = make_float_scorer(ident, fp_v, fp_m, args.top_k)
        meta["fp32 full scan (reference)"] = {"r": fp_dim, "eval_mode": "float",
                                              "bytes_per_token": fp_dim * 4, "stage": "full_exhaustive"}

        # stage-1 candidate precompute (UNTIMED): sign-coded top-maxK pids per query
        maxK = max(args.rerank_cutoffs)
        logger.info("Precomputing stage-1 top-%d candidates per query (untimed)...", maxK)
        cand_pids: dict[str, list] = {}
        for qid in tqdm(qids, desc="  stage1 cands", leave=False, disable=not _TTY):
            q = query_store.get(qid).to(device)
            with torch.no_grad():
                qe = enc_q_sign(q)
                scs = []
                for dv, dm in zip(sgn_v, sgn_m):
                    sim = torch.einsum("md,cnd->cmn", qe, dv)
                    sim = sim.masked_fill(~dm.unsqueeze(1), float("-inf"))
                    scs.append(sim.max(dim=-1).values.sum(dim=-1))
                sc = torch.cat(scs)
                top = torch.topk(sc, min(maxK, sc.shape[0])).indices.tolist()
            cand_pids[qid] = [pid_list[g] for g in top]

        def make_rerank_scorer(K):
            def score_one(q, qid=None):
                with torch.no_grad():
                    docs = [fp_by_pid[p] for p in cand_pids[qid][:K]]
                    mxl = max(d.shape[0] for d in docs)
                    v = torch.zeros(len(docs), mxl, fp_dim, device=device)
                    m = torch.zeros(len(docs), mxl, dtype=torch.bool, device=device)
                    for j, d in enumerate(docs):
                        v[j, :d.shape[0]] = d
                        m[j, :d.shape[0]] = True
                    sim = torch.einsum("md,cnd->cmn", q.to(device), v)
                    sim = sim.masked_fill(~m.unsqueeze(1), float("-inf"))
                    sc = sim.max(dim=-1).values.sum(dim=-1)
                    torch.topk(sc, min(args.top_k, sc.shape[0]))
            return score_one

        for K in args.rerank_cutoffs:
            methods[f"stage2 fp32 rerank K={K}"] = make_rerank_scorer(K)
            meta[f"stage2 fp32 rerank K={K}"] = {"stage": "rerank", "K": K, "eval_mode": "float"}

    labels = list(methods.keys())

    # ---- warm-up round (untimed) ----------------------------------------------
    logger.info("Warm-up round (untimed)...")
    for label in labels:
        for qid in qids[:max(50, len(qids) // 10)]:
            methods[label](query_store.get(qid).to(device), qid)

    # ---- interleaved timed rounds ----------------------------------------------
    rounds: list[dict] = []
    order_rng = pyrandom.Random(args.seed)
    for rnd in range(args.rounds):
        order = labels[:]
        order_rng.shuffle(order)
        logger.info("Round %d/%d  order=%s", rnd + 1, args.rounds, order)
        entry = {"round": rnd + 1, "order": order, "results": {}, "t_start": time.time()}
        for label in order:
            stats = timed_pass(methods[label], query_store, qids, device)
            entry["results"][label] = stats
            logger.info("  %-28s median=%.2f ms  mean=%.2f  p95=%.2f",
                        label, stats["median_ms"], stats["mean_ms"], stats["p95_ms"])
        rounds.append(entry)

    # ---- summarize ---------------------------------------------------------------
    summary = {}
    for label in labels:
        med = sorted(r["results"][label]["median_ms"] for r in rounds)
        n = len(med)
        q25, q75 = med[max(0, int(0.25 * (n - 1)))], med[min(n - 1, int(round(0.75 * (n - 1))))]
        summary[label] = {
            "median_of_round_medians_ms": med[n // 2] if n % 2 else 0.5 * (med[n//2 - 1] + med[n//2]),
            "round_medians_ms": med,
            "iqr_ms": q75 - q25,
            "min_ms": med[0], "max_ms": med[-1], "spread_ms": med[-1] - med[0],
            "mean_of_round_means_ms": float(np.mean([r["results"][label]["mean_ms"] for r in rounds])),
        }

    verdict = None
    if args.group == "projections":
        meds = {l: summary[l]["median_of_round_medians_ms"] for l in labels}
        gap = max(meds.values()) - min(meds.values())
        max_iqr = max(summary[l]["iqr_ms"] for l in labels)
        verdict = {
            "max_pairwise_median_gap_ms": gap,
            "max_within_method_iqr_ms": max_iqr,
            "projections_indistinguishable": bool(gap <= max(max_iqr, 1e-9)),
            "recommendation": ("collapse the four projection rows into one sign-coded row"
                               if gap <= max_iqr else
                               "gap exceeds within-method IQR; investigate before collapsing"),
        }
        logger.info("VERDICT: %s", verdict)

    payload = {
        "run_metadata": collect_run_metadata(config={
            "benchmark": "latency_interleaved",
            "group": args.group, "rounds": args.rounds,
            "num_queries": len(qids), "query_tokens": n_qtok,
            "query_embeddings": args.query_embeddings,
            "num_docs": len(doc_ids), "top_k": args.top_k, "seed": args.seed,
            "device": str(device), "threads": args.threads,
            "plaid_centroids": args.plaid_centroids,
            "plaid_residual_bits": args.plaid_residual_bits,
            "checkpoint": args.checkpoint if args.group == "projections" else None,
            "design": "1 warmup round + N timed rounds, per-round shuffled method order; "
                      "per-pass statistic = per-query MEDIAN (stall-robust, per W4 finding)",
        }),
        "environment": env,
        "method_meta": meta,
        "rounds": rounds,
        "summary": summary,
        "verdict": verdict,
    }
    ensure_dir(out_path.parent)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Saved %s", out_path)

    print("\n" + "=" * 80)
    print(f"INTERLEAVED LATENCY  group={args.group}  rounds={args.rounds}  "
          f"queries={len(qids)} ({n_qtok} tokens)  device={device} threads={args.threads}")
    print("=" * 80)
    print(f"{'Method':<30} {'median':>9} {'IQR':>8} {'spread':>8} {'mean':>9}")
    print("-" * 80)
    for label in labels:
        s = summary[label]
        print(f"{label:<30} {s['median_of_round_medians_ms']:>8.2f}m {s['iqr_ms']:>7.2f}m "
              f"{s['spread_ms']:>7.2f}m {s['mean_of_round_means_ms']:>8.2f}m")
    if verdict:
        print("-" * 80)
        print(f"VERDICT: {verdict['recommendation']}")
        print(f"  max pairwise median gap = {verdict['max_pairwise_median_gap_ms']:.2f} ms; "
              f"max within-method IQR = {verdict['max_within_method_iqr_ms']:.2f} ms")


def _has(store, qid) -> bool:
    try:
        store.get(qid)
        return True
    except (KeyError, FileNotFoundError):
        return False


if __name__ == "__main__":
    main()
