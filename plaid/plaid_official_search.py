#!/usr/bin/env python3
"""Quality + engine latency for the OFFICIAL PLAID engine (colbert-ai).

Two phases:
  A) QUALITY: all 6,980 judged dev queries, k=1000 -> MRR@10 / R@100 / R@1000
     against the repo's qrels (results mapped back via pid_map.tsv).
     PLAID encodes queries with its own ColBERTv2 checkpoint (standard
     32-token augmented convention, matching the paper's re-baseline).
  B) ENGINE LATENCY: the same seeded 500-query subsample as
     bench_latency_interleaved.py (rng(42).permutation over sorted qrels qids),
     1 warm-up round + N timed rounds. Primary timed region = dense_search()
     (the engine: candidate generation + pruning + decompression + scoring),
     EXCLUDING BERT query encoding, to match the bench's timing boundary
     (our rows time projection + scoring, not the upstream query encoder).
     Full search() time (encode included) is recorded as secondary.

Run (after the index finishes, on an otherwise idle box):
    PYTHONUNBUFFERED=1 .venv-plaid/bin/python plaid/plaid_official_search.py \
        --index msmarco100k.nbits2 \
        2>&1 | tee /tmp/plaid_official_search.log

Smoke (plumbing check against the 1k smoke index; quality will be meaningless):
    .venv-plaid/bin/python plaid/plaid_official_search.py \
        --index smoke1k.nbits2 --collection data/plaid_official/collection_smoke1k.tsv \
        --max-quality-queries 20 --rounds 1 --latency-queries 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Same env fixes as the indexer: ninja on PATH for JIT extensions; the faiss
# libomp is symlinked to torch's (single OpenMP runtime), KMP guard harmless.
os.environ["PATH"] = f"{Path(sys.executable).parent}:{os.environ.get('PATH', '')}"
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--index", required=True)
    p.add_argument("--root", default="data/plaid_official/experiments")
    p.add_argument("--collection", default="data/plaid_official/collection_100k.tsv")
    p.add_argument("--queries", default="data/plaid_official/queries_dev.tsv")
    p.add_argument("--pid-map", default="data/plaid_official/pid_map.tsv")
    p.add_argument("--qrels", default="data/embeddings/msmarco/100k/qrels.tsv")
    p.add_argument("--k", type=int, default=1000)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--latency-queries", type=int, default=500)
    p.add_argument("--max-quality-queries", type=int, default=-1)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="outputs/latency/plaid_official_engine.json")
    args = p.parse_args()

    import torch
    torch.set_num_threads(args.threads)

    from colbert import Searcher
    from colbert.infra import Run, RunConfig

    # ---- inputs ---------------------------------------------------------------
    new2pid: dict[int, str] = {}
    with open(args.pid_map) as f:
        for line in f:
            nid, pid = line.rstrip("\n").split("\t")
            new2pid[int(nid)] = pid

    queries: dict[str, str] = {}
    with open(args.queries) as f:
        for line in f:
            qid, _, text = line.partition("\t")
            queries[qid] = text.rstrip("\n")

    qrels: dict[str, set] = {}
    with open(args.qrels) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            qid, pid = parts[0], parts[2] if len(parts) >= 4 else parts[1]
            qrels.setdefault(qid, set()).add(pid)

    qids_all = sorted(qrels.keys())
    qids_all = [q for q in qids_all if q in queries]
    print(f"{len(qids_all):,} judged queries with text", flush=True)

    with Run().context(RunConfig(nranks=1, experiment="plaid_official", root=args.root)):
        searcher = Searcher(index=args.index, collection=args.collection)

        # engine parameters: colbert-ai resolves ncells/threshold/ndocs lazily
        # at search time; record config values with the documented k-dependent
        # defaults as fallback (Searcher._search_all_Q logic).
        cfg = searcher.config
        if args.k <= 10:
            dflt = (1, 0.5, 256)
        elif args.k <= 100:
            dflt = (2, 0.45, 1024)
        else:
            dflt = (4, 0.4, max(args.k * 4, 4096))
        engine_params = {
            "k": args.k,
            "ncells": cfg.ncells or dflt[0],
            "centroid_score_threshold": cfg.centroid_score_threshold or dflt[1],
            "ndocs": cfg.ndocs or dflt[2],
            "nbits": cfg.nbits, "doc_maxlen": cfg.doc_maxlen,
        }
        print("engine params:", engine_params, flush=True)

        # ---- Phase A: quality ------------------------------------------------
        qq = qids_all if args.max_quality_queries <= 0 else qids_all[:args.max_quality_queries]
        print(f"[A] quality pass: {len(qq):,} queries, k={args.k}", flush=True)
        rr_sum = 0.0; r100 = 0; r1000 = 0
        t0 = time.time()
        for i, qid in enumerate(qq):
            ranked = searcher.search(queries[qid], k=args.k)
            pids = [new2pid[d] for d in ranked[0]]
            rel = qrels[qid]
            rr = 0.0
            for rank, pid in enumerate(pids[:10], 1):
                if pid in rel:
                    rr = 1.0 / rank
                    break
            rr_sum += rr
            if rel & set(pids[:100]):
                r100 += 1
            if rel & set(pids[:1000]):
                r1000 += 1
            if (i + 1) % 500 == 0:
                print(f"    {i+1}/{len(qq)}  mrr@10 so far {rr_sum/(i+1):.4f} "
                      f"({(time.time()-t0)/ (i+1)*1000:.0f} ms/q wall)", flush=True)
        quality = {
            "n_queries": len(qq),
            "mrr@10": rr_sum / max(1, len(qq)),
            "recall@100": r100 / max(1, len(qq)),
            "recall@1000": r1000 / max(1, len(qq)),
        }
        print("[A] quality:", quality, flush=True)

        # ---- Phase B: engine latency (dense_search, excludes query BERT) ------
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(len(qids_all))
        lat_qids = [qids_all[i] for i in perm][: args.latency_queries]
        print(f"[B] latency: {len(lat_qids)} queries x (1 warmup + {args.rounds} rounds)", flush=True)

        # pre-encode all latency queries (matches the bench boundary: query
        # encoding happens outside the timed region)
        enc = {qid: searcher.encode([queries[qid]]) for qid in lat_qids}

        def timed_round() -> dict:
            lat = np.empty(len(lat_qids))
            full = np.empty(len(lat_qids))
            for i, qid in enumerate(lat_qids):
                Q = enc[qid]
                t0 = time.perf_counter_ns()
                searcher.dense_search(Q, k=args.k)
                lat[i] = (time.perf_counter_ns() - t0) / 1e6
                t1 = time.perf_counter_ns()
                searcher.search(queries[qid], k=args.k)
                full[i] = (time.perf_counter_ns() - t1) / 1e6
            return {"engine_median_ms": float(np.median(lat)),
                    "engine_mean_ms": float(lat.mean()),
                    "engine_p95_ms": float(np.percentile(lat, 95)),
                    "full_search_median_ms": float(np.median(full))}

        print("    warmup...", flush=True)
        for qid in lat_qids[: max(50, len(lat_qids) // 10)]:
            searcher.dense_search(enc[qid], k=args.k)

        rounds = []
        for r in range(args.rounds):
            stats = timed_round()
            rounds.append(stats)
            print(f"    round {r+1}/{args.rounds}: engine median={stats['engine_median_ms']:.2f} ms  "
                  f"(full search median={stats['full_search_median_ms']:.2f} ms)", flush=True)

    med = sorted(s["engine_median_ms"] for s in rounds)
    n = len(med)
    summary = {
        "engine_median_of_round_medians_ms": med[n // 2] if n % 2 else 0.5 * (med[n//2-1] + med[n//2]),
        "engine_round_medians_ms": med,
        "engine_iqr_ms": med[min(n-1, int(round(0.75*(n-1))))] - med[max(0, int(0.25*(n-1)))],
        "full_search_median_ms": float(np.median([s["full_search_median_ms"] for s in rounds])),
    }
    payload = {
        "benchmark": "plaid_official_engine",
        "implementation": "colbert-ai 0.2.22 (official PLAID), CPU",
        "index": args.index, "engine_params": engine_params,
        "threads": args.threads, "seed": args.seed,
        "latency_queries": len(lat_qids), "rounds": args.rounds,
        "timing_note": "engine_* times dense_search() only (candidate generation + pruning + "
                       "decompression + scoring), excluding BERT query encoding, to match the "
                       "interleaved bench's boundary; full_search_* includes query encoding.",
        "quality": quality,
        "rounds_detail": rounds,
        "summary": summary,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nsaved {out}")
    print(f"ENGINE latency (median of round medians): {summary['engine_median_of_round_medians_ms']:.2f} ms "
          f"| IQR {summary['engine_iqr_ms']:.2f} | full-search median {summary['full_search_median_ms']:.2f} ms")
    print(f"QUALITY: {quality}")


if __name__ == "__main__":
    main()
