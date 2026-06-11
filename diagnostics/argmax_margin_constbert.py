"""ConstBERT median fp32 argmax margin — quantitative test of paper §6's
predictive claim that the configuration-invariance regime tightens for
encoders with smaller median argmax margins.

For each of N_SAMPLED_QUERIES judged dev queries:
  1. Brute-force fp32 ConstBERT MaxSim against the full 100k corpus.
  2. Take the top-TOP_K_CANDIDATES (=100) candidate docs.
  3. For each query token, find the per-q-token argmax (cand_idx, doc_token_idx)
     and its similarity. Find the runner-up similarity over the same flattened
     (cand_idx, doc_token_idx) space.
  4. Margin = top1_sim - top2_sim.

Output:
  outputs/argmax_margin_constbert.json
    - median, mean, std, P25, P75 of margin distribution
    - sigma_r(theta_bar) for r in {16, 32, 64} via Charikar SimHash
    - predicted minimum r for which median(margin) > sigma_r

ConstBERT vs ColBERTv2 reference:
  ColBERTv2's median argmax margin on the same matched setup is ~0.20. If
  ConstBERT's median is ~0.10-0.15,
  the prediction "tighter-margin encoders need higher r for configuration
  invariance" is empirically confirmed.
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

DEFAULT_EMB_DIR = Path("data/embeddings/constbert/100k")
DEFAULT_OUT = Path("outputs/argmax_margin_constbert.json")

N_SAMPLED_QUERIES = 500
TOP_K_CANDIDATES = 100
SEED = 42
R_DIMS = [16, 32, 64]


def sigma_r_simhash(theta: float, r: int) -> float:
    """Charikar SimHash collision-prob standard deviation on cosine similarity
    after r-bit sign code projection. sigma_r = sqrt(theta*(pi-theta) / (r*pi^2)).
    theta is the angle (radians) between the two unit vectors, recoverable from
    similarity s as theta = arccos(s).
    """
    return math.sqrt(theta * (math.pi - theta) / (r * math.pi**2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dir", type=Path, default=DEFAULT_EMB_DIR)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--device", default="mps", choices=["auto", "mps", "cuda", "cpu"])
    ap.add_argument("--n-queries", type=int, default=N_SAMPLED_QUERIES)
    ap.add_argument("--top-k", type=int, default=TOP_K_CANDIDATES)
    ap.add_argument("--doc-chunk", type=int, default=8192)
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"device={device}", flush=True)

    print(f"loading ConstBERT embeddings from {args.emb_dir}...", flush=True)
    doc_emb = torch.load(args.emb_dir / "doc_embeddings.pt", map_location="cpu", weights_only=False)
    qry_emb = torch.load(args.emb_dir / "query_embeddings.pt", map_location="cpu", weights_only=False)
    qrels = load_qrels(args.emb_dir / "qrels.tsv")
    pids = list(doc_emb.keys())
    N = len(pids)
    K_doc = doc_emb[pids[0]].shape[0]
    print(f"  {N} docs, K={K_doc}, {len(qry_emb)} queries", flush=True)

    print("stacking docs...", flush=True)
    D = torch.empty(N, K_doc, 128, dtype=torch.float32)
    for i, pid in enumerate(pids):
        D[i] = doc_emb[pid]
    D = D.to(device)

    rng = np.random.default_rng(SEED)
    judged_qids = sorted([q for q in qry_emb if q in qrels and any(s > 0 for s in qrels[q].values())])
    sample_qids = rng.choice(judged_qids, size=min(args.n_queries, len(judged_qids)), replace=False)
    print(f"  sampled {len(sample_qids)} judged queries", flush=True)

    margins: list[float] = []
    top_sims: list[float] = []  # for computing mean theta_bar
    t0 = time.perf_counter()

    for qi, qid in enumerate(sample_qids):
        q = qry_emb[qid].to(device).float()  # (m_q, 128) — for ConstBERT m_q is 32
        m_q = q.shape[0]

        # Brute-force fp32 MaxSim over full corpus to identify top-K candidates.
        best_scores = torch.empty(N, dtype=torch.float32, device=device)
        for cs in range(0, N, args.doc_chunk):
            ce = min(cs + args.doc_chunk, N)
            sim = torch.einsum("md,nkd->nmk", q, D[cs:ce])  # (chunk, m_q, K_doc)
            best_scores[cs:ce] = sim.max(dim=-1).values.sum(dim=-1)
        topk_idx = torch.topk(best_scores, args.top_k).indices

        cand_docs = D[topk_idx]  # (TOP_K, K_doc, 128)

        # Per-q-token argmax over flattened (TOP_K, K_doc) space.
        with torch.no_grad():
            sim_cand = torch.einsum("md,knd->mkn", q, cand_docs)  # (m_q, TOP_K, K_doc)
            sim_flat = sim_cand.reshape(m_q, args.top_k * K_doc)
            top_val, top_idx = sim_flat.max(dim=-1)  # (m_q,)
            sim_clone = sim_flat.clone()
            sim_clone.scatter_(1, top_idx.unsqueeze(-1), float("-inf"))
            runner_val = sim_clone.max(dim=-1).values  # (m_q,)
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
    print(f"\nTotal q-tokens: {len(margins_arr)}", flush=True)
    print(f"Margin distribution:", flush=True)
    print(f"  P25     = {np.percentile(margins_arr, 25):.4f}", flush=True)
    print(f"  median  = {np.median(margins_arr):.4f}", flush=True)
    print(f"  P75     = {np.percentile(margins_arr, 75):.4f}", flush=True)
    print(f"  mean    = {margins_arr.mean():.4f}", flush=True)
    print(f"  std     = {margins_arr.std():.4f}", flush=True)
    print(f"\nMean top-1 similarity: {top_sims_arr.mean():.4f}", flush=True)

    # sigma_r at theta_bar = arccos(mean top1 sim).
    # Clip to valid arccos domain.
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

    thresholds = [0.05, 0.1, 0.2]
    n_total = int(len(margins_arr))
    high_margin = {}
    for t in thresholds:
        n_ge = int((margins_arr >= t).sum())
        high_margin[f"n_ge_{t}"] = n_ge
        high_margin[f"fraction_margin_ge_{t}"] = n_ge / n_total if n_total else 0.0

    summary = {
        "encoder": "pinecone/ConstBERT",
        "corpus": "MS MARCO Passage 100k",
        "n_queries": int(len(sample_qids)),
        "top_k_candidates": int(args.top_k),
        "n_qtokens_total": n_total,
        "seed": SEED,
        "margin_distribution": {
            "p25": float(np.percentile(margins_arr, 25)),
            "median": float(np.median(margins_arr)),
            "p75": float(np.percentile(margins_arr, 75)),
            "mean": float(margins_arr.mean()),
            "std": float(margins_arr.std()),
        },
        "high_margin_tail": high_margin,
        "mean_top1_similarity": float(top_sims_arr.mean()),
        "theta_bar_radians": float(theta_bar),
        "sigma_r_simhash": {str(r): float(sigma_by_r[r]) for r in R_DIMS},
        "predicted_min_r_for_invariance": predicted_r,
    }
    print(f"\nHigh-margin tail counts:", flush=True)
    for t in thresholds:
        n_ge = high_margin[f"n_ge_{t}"]
        frac = high_margin[f"fraction_margin_ge_{t}"]
        print(f"  margin >= {t}: {n_ge} / {n_total} ({frac*100:.4f}%)", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
