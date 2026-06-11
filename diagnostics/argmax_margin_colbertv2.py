"""ColBERTv2 median fp32 argmax margin on a matched-500-query subsample
of MS MARCO Passage dev.

The ConstBERT 500-query argmax margin probe (``argmax_margin_constbert.py``)
is run on a 500-query subsample while the ColBERTv2 margin numbers cited
alongside it in ``\\sec{results}`` are from the full 6,980-query setup.
A strict reviewer (iter_5 c5) asks whether the order-of-magnitude tail
gap (1.08% vs 0.12% at margin >= 0.2) is robust on matched sample sizes.

This script runs the SAME 500-query subsample (same seed, same judged
qids) against ColBERTv2 100k embeddings and reports the high-margin
tail fraction alongside the full margin distribution. The matching
mechanism is:
  rng = np.random.default_rng(SEED)
  judged_qids = sorted([q for q in qry_emb if q in qrels and any(s > 0 for s in qrels[q].values())])
  sample_qids = rng.choice(judged_qids, size=500, replace=False)
which is identical to ``argmax_margin_constbert.py``. judged_qids is
encoder-independent (it is determined by the MS MARCO qrels file, not
the embedding model), so seed + size yield the same 500 query IDs.

Output:
  outputs/argmax_margin_colbertv2_500q.json
    - margin distribution (P25, median, P75, mean, std)
    - high-margin tail fraction at margin >= 0.2 (the Exp 5 verdict
      number; ConstBERT's was 0.12%, the 6,980-query ColBERTv2 ref is
      1.08%)
    - sigma_r predictions for r in {16, 32, 64}
    - smallest r in R_DIMS with median(margin) > sigma_r
    - the 500 query IDs used (for provenance reproducibility)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(".").resolve()))
from src.subbit.data import load_qrels

DEFAULT_EMB_DIR = Path("data/embeddings/msmarco/100k")
DEFAULT_OUT = Path("outputs/argmax_margin_colbertv2_500q.json")

N_SAMPLED_QUERIES = 500
TOP_K_CANDIDATES = 100
SEED = 42                 # matched to argmax_margin_constbert.py
HIGH_MARGIN_THRESHOLD = 0.2  # the runbook's Exp 5 reference threshold
R_DIMS = [16, 32, 64]


def sigma_r_simhash(theta: float, r: int) -> float:
    """Charikar SimHash collision-prob std on cosine after r-bit sign.
    sigma_r = sqrt(theta * (pi - theta) / (r * pi^2))."""
    return math.sqrt(theta * (math.pi - theta) / (r * math.pi**2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dir", type=Path, default=DEFAULT_EMB_DIR)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--query-embeddings", type=Path, default=None,
                    help="Override the query cache (e.g. the augmented "
                         "query_embeddings_aug.pt). Defaults to emb-dir/query_embeddings.pt.")
    ap.add_argument("--device", default="mps", choices=["auto", "mps", "cuda", "cpu"])
    ap.add_argument("--n-queries", type=int, default=N_SAMPLED_QUERIES,
                    help="Number of queries to sample. Keep at 500 for matched comparison with argmax_margin_constbert.py.")
    ap.add_argument("--top-k", type=int, default=TOP_K_CANDIDATES)
    ap.add_argument("--doc-chunk", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=SEED,
                    help="RNG seed for the matched-subsample. Default 42 mirrors argmax_margin_constbert.py.")
    ap.add_argument("--high-margin-threshold", type=float, default=HIGH_MARGIN_THRESHOLD,
                    help="Margin threshold for the tail-fraction summary statistic.")
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"device={device}", flush=True)

    print(f"loading ColBERTv2 embeddings from {args.emb_dir}...", flush=True)
    doc_emb = torch.load(args.emb_dir / "doc_embeddings.pt", map_location="cpu", weights_only=False)
    qpath = args.query_embeddings or (args.emb_dir / "query_embeddings.pt")
    print(f"  query cache: {qpath}", flush=True)
    qry_emb = torch.load(qpath, map_location="cpu", weights_only=False)
    qrels = load_qrels(args.emb_dir / "qrels.tsv")

    # ColBERTv2 docs have variable token counts, unlike ConstBERT's fixed K=32.
    # Detect and pad if needed.
    pids = list(doc_emb.keys())
    N = len(pids)
    sample_tensor = doc_emb[pids[0]]
    if sample_tensor.ndim == 2:
        token_counts = [doc_emb[p].shape[0] for p in pids]
        K_doc = max(token_counts)
        print(f"  {N} docs, variable K (max={K_doc}, mean={sum(token_counts)/N:.1f}), "
              f"{len(qry_emb)} queries", flush=True)
        # Pad to K_doc for batched einsum; record mask.
        D = torch.zeros(N, K_doc, 128, dtype=torch.float32)
        D_lengths = torch.tensor(token_counts, dtype=torch.long)
        for i, pid in enumerate(pids):
            ln = int(D_lengths[i].item())
            D[i, :ln] = doc_emb[pid]
        variable_length = True
    else:
        K_doc = sample_tensor.shape[0]
        print(f"  {N} docs, fixed K={K_doc}, {len(qry_emb)} queries", flush=True)
        D = torch.empty(N, K_doc, 128, dtype=torch.float32)
        for i, pid in enumerate(pids):
            D[i] = doc_emb[pid]
        D_lengths = torch.full((N,), K_doc, dtype=torch.long)
        variable_length = False
    D = D.to(device)
    D_lengths = D_lengths.to(device)
    if variable_length:
        col = torch.arange(K_doc, device=device)
        valid = col[None, :] < D_lengths[:, None]    # (N, K_doc) bool
    else:
        valid = None  # no padding to mask

    rng = np.random.default_rng(args.seed)
    judged_qids = sorted([q for q in qry_emb if q in qrels and any(s > 0 for s in qrels[q].values())])
    sample_qids = rng.choice(judged_qids, size=min(args.n_queries, len(judged_qids)), replace=False)
    print(f"  sampled {len(sample_qids)} judged queries with seed={args.seed} "
          f"(matched to argmax_margin_constbert.py at seed=42)", flush=True)

    margins: list[float] = []
    top_sims: list[float] = []
    t0 = time.perf_counter()

    for qi, qid in enumerate(sample_qids):
        q = qry_emb[qid].to(device).float()  # (m_q, 128)
        m_q = q.shape[0]

        # Brute-force fp32 MaxSim across full 100k corpus to identify top-K.
        best_scores = torch.empty(N, dtype=torch.float32, device=device)
        for cs in range(0, N, args.doc_chunk):
            ce = min(cs + args.doc_chunk, N)
            sim = torch.einsum("md,nkd->nmk", q, D[cs:ce])  # (chunk, m_q, K_doc)
            if variable_length:
                sim.masked_fill_(~valid[cs:ce][:, None, :], float("-inf"))
            best_scores[cs:ce] = sim.max(dim=-1).values.sum(dim=-1)
        topk_idx = torch.topk(best_scores, args.top_k).indices

        cand_docs = D[topk_idx]  # (TOP_K, K_doc, 128)
        if variable_length:
            cand_valid = valid[topk_idx]  # (TOP_K, K_doc)
        else:
            cand_valid = None

        with torch.no_grad():
            sim_cand = torch.einsum("md,knd->mkn", q, cand_docs)  # (m_q, TOP_K, K_doc)
            if variable_length:
                # Mask padded positions so they cannot win the argmax.
                sim_cand.masked_fill_(~cand_valid[None, :, :], float("-inf"))
            sim_flat = sim_cand.reshape(m_q, args.top_k * K_doc)
            top_val, top_idx = sim_flat.max(dim=-1)
            sim_clone = sim_flat.clone()
            sim_clone.scatter_(1, top_idx.unsqueeze(-1), float("-inf"))
            runner_val = sim_clone.max(dim=-1).values
            margin = (top_val - runner_val).detach().cpu().numpy()
            top_val_np = top_val.detach().cpu().numpy()

        margins.extend(margin.tolist())
        top_sims.extend(top_val_np.tolist())

        if (qi + 1) % 50 == 0 or qi + 1 == len(sample_qids):
            elapsed = time.perf_counter() - t0
            eta = elapsed / (qi + 1) * (len(sample_qids) - qi - 1)
            print(
                f"  [{qi+1}/{len(sample_qids)}] elapsed={elapsed:.1f}s eta={eta:.1f}s",
                flush=True,
            )

    margins_arr = np.array(margins)
    top_sims_arr = np.array(top_sims)
    n_qtokens = int(len(margins_arr))
    n_high_margin = int((margins_arr >= args.high_margin_threshold).sum())
    tail_fraction = float(n_high_margin / max(n_qtokens, 1))

    print(f"\nTotal q-tokens: {n_qtokens}", flush=True)
    print(f"Margin distribution:", flush=True)
    print(f"  P25     = {np.percentile(margins_arr, 25):.4f}", flush=True)
    print(f"  median  = {np.median(margins_arr):.4f}", flush=True)
    print(f"  P75     = {np.percentile(margins_arr, 75):.4f}", flush=True)
    print(f"  mean    = {margins_arr.mean():.4f}", flush=True)
    print(f"  std     = {margins_arr.std():.4f}", flush=True)
    print(f"\nHigh-margin tail (margin >= {args.high_margin_threshold}):", flush=True)
    print(f"  count           = {n_high_margin}", flush=True)
    print(f"  tail fraction   = {tail_fraction*100:.2f}%", flush=True)
    print(f"  (runbook ref: 1.08% at 6,980 queries; ConstBERT 0.12%)", flush=True)

    print(f"\nMean top-1 similarity: {top_sims_arr.mean():.4f}", flush=True)
    s_bar = float(np.clip(top_sims_arr.mean(), -1.0, 1.0))
    theta_bar = math.acos(s_bar)
    print(f"theta_bar (radians) = {theta_bar:.4f}", flush=True)

    sigma_by_r = {}
    print(f"\nsigma_r predictions (Charikar SimHash):", flush=True)
    for r in R_DIMS:
        sr = sigma_r_simhash(theta_bar, r)
        sigma_by_r[r] = sr
        print(f"  r={r:>3}: sigma_r = {sr:.4f}", flush=True)

    median_margin = float(np.median(margins_arr))
    predicted_r = None
    for r in R_DIMS:
        if median_margin > sigma_by_r[r]:
            predicted_r = r
            break
    print(f"\nMedian margin = {median_margin:.4f}", flush=True)
    if predicted_r is not None:
        print(f"Smallest r in {R_DIMS} with median_margin > sigma_r: r = {predicted_r}", flush=True)
    else:
        print(f"Median margin does not exceed sigma_r at any tested r; predicted r > {R_DIMS[-1]}", flush=True)

    summary = {
        "encoder": "colbert-ir/colbertv2.0",
        "corpus": "MS MARCO Passage 100k",
        "n_queries": int(len(sample_qids)),
        "top_k_candidates": int(args.top_k),
        "n_qtokens_total": n_qtokens,
        "seed": int(args.seed),
        "high_margin_threshold": float(args.high_margin_threshold),
        "matched_subsample": True,
        "matched_to_script": "argmax_margin_constbert.py",
        "margin_distribution": {
            "p25": float(np.percentile(margins_arr, 25)),
            "median": float(np.median(margins_arr)),
            "p75": float(np.percentile(margins_arr, 75)),
            "mean": float(margins_arr.mean()),
            "std": float(margins_arr.std()),
        },
        "high_margin_tail": {
            "threshold": float(args.high_margin_threshold),
            "count": n_high_margin,
            "fraction": tail_fraction,
            "fraction_percent": tail_fraction * 100.0,
        },
        "mean_top1_similarity": float(top_sims_arr.mean()),
        "theta_bar_radians": float(theta_bar),
        "sigma_r_simhash": {str(r): float(sigma_by_r[r]) for r in R_DIMS},
        "predicted_min_r_for_invariance": predicted_r,
        # Provenance: which 500 queries were used. Lets a teammate
        # verify the matched-subsample claim against argmax_margin_constbert.py.
        "sample_qids_first_10": [str(q) for q in sample_qids[:10]],
        "sample_qids_last_10": [str(q) for q in sample_qids[-10:]],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
