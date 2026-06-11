"""Paired bootstrap 95% CI for Delta MRR@10 between identity sign(d) and
random orthogonal R at r=128 on the full 8.8M MS MARCO dev corpus.

Loads the two per_query.pt files, aligns by qids, paired-bootstraps the
per-query rr@10 delta (sign_d - rand_r128) at n_boot=10000, seed=42.
Writes results to outputs/full_msmarco/bootstrap_r128_8m.json and prints
the headline number.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
RAND_PATH = ROOT / "outputs/full_msmarco/rand_r128_full_msmarco.per_query.pt"
SIGN_PATH = ROOT / "outputs/full_msmarco/sign_d_full_msmarco.per_query.pt"
OUT_PATH = ROOT / "outputs/full_msmarco/bootstrap_r128_8m.json"

N_BOOT = 10_000
SEED = 42
METRICS = ("rr@10", "recall@100", "recall@1000", "ndcg@10")


def load(path: Path) -> dict:
    d = torch.load(path, weights_only=False, map_location="cpu")
    return d


def align(d_rand: dict, d_sign: dict) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    qids_rand = list(d_rand["qids"])
    qids_sign = list(d_sign["qids"])
    if qids_rand == qids_sign:
        order_rand = np.arange(len(qids_rand))
        order_sign = np.arange(len(qids_sign))
        qids = np.array(qids_rand)
    else:
        # Build index by qid
        idx_rand = {q: i for i, q in enumerate(qids_rand)}
        idx_sign = {q: i for i, q in enumerate(qids_sign)}
        common = sorted(set(qids_rand) & set(qids_sign))
        order_rand = np.array([idx_rand[q] for q in common])
        order_sign = np.array([idx_sign[q] for q in common])
        qids = np.array(common)

    rand_metrics = {}
    sign_metrics = {}
    for m in METRICS:
        rand_metrics[m] = np.asarray(d_rand[m])[order_rand]
        sign_metrics[m] = np.asarray(d_sign[m])[order_sign]
    return qids, rand_metrics, sign_metrics


def paired_bootstrap_ci(
    a: np.ndarray, b: np.ndarray, n_boot: int, seed: int
) -> dict:
    """Paired bootstrap 95% CI on mean(b - a)."""
    assert a.shape == b.shape
    n = a.shape[0]
    rng = np.random.default_rng(seed)
    delta = b - a
    point = float(delta.mean())
    # Resample query indices with replacement
    boot_means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = delta[idx].mean()
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return {
        "point": point,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "n_queries": int(n),
        "n_boot": int(n_boot),
        "seed": int(seed),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=OUT_PATH)
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Loading {RAND_PATH.name} ...")
    d_rand = load(RAND_PATH)
    print(f"Loading {SIGN_PATH.name} ...")
    d_sign = load(SIGN_PATH)

    qids, rand_m, sign_m = align(d_rand, d_sign)
    print(f"Aligned n={len(qids)} queries (rand: {len(d_rand['qids'])}, sign: {len(d_sign['qids'])})")

    results = {
        "method_a": d_rand.get("method", "rand_r128"),
        "method_b": d_sign.get("method", "sign_d"),
        "comparison": "sign_d - rand_r128 (paired bootstrap)",
        "n_queries": int(len(qids)),
        "n_boot": args.n_boot,
        "seed": args.seed,
        "metrics": {},
    }

    for m in METRICS:
        out = paired_bootstrap_ci(rand_m[m], sign_m[m], args.n_boot, args.seed)
        results["metrics"][m] = out
        print(
            f"  {m}: rand={out['mean_a']:.4f}, sign={out['mean_b']:.4f}, "
            f"Δ={out['point']:+.4f} [{out['ci_lo']:+.4f}, {out['ci_hi']:+.4f}]"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.output}")

    mrr = results["metrics"]["rr@10"]
    print(
        f"\nHEADLINE: r=128 8.8M, sign(d) - random orthogonal R, "
        f"ΔMRR@10 = {mrr['point']:+.4f} "
        f"[{mrr['ci_lo']:+.4f}, {mrr['ci_hi']:+.4f}] "
        f"(paired bootstrap, n={mrr['n_queries']}, n_boot={mrr['n_boot']})"
    )


if __name__ == "__main__":
    main()
