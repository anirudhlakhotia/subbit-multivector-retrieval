"""Paired bootstrap 95% CI for Delta MRR@10 between trained R and random
orthogonal R + scale at r=64 on the full 8.8M MS MARCO dev corpus.

Closes the headline-falsification CI gap (the Δ=-0.0003 number that has
been a point estimate across iter-1/2/3 peer reviews). Mirrors
bootstrap_8m_r128.py but compares the two r=64 conditions instead.

Prerequisites:
  - outputs/full_msmarco/subbit_full_msmarco.per_query.pt
  - outputs/full_msmarco/rand_plus_scale_full_msmarco.per_query.pt
Both are produced by full_scale/eval_subbit_full.py and
full_scale/eval_rand_plus_scale_full.py with the per-query sidecar block.

Loads the two per_query.pt files, aligns by qids, paired-bootstraps the
per-query rr@10 delta (subbit_trained - rand_plus_scale) at n_boot=10000,
seed=42. Writes results to outputs/full_msmarco/bootstrap_r64_8m.json
and prints the headline number.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
RAND_PATH = ROOT / "outputs/full_msmarco/rand_plus_scale_full_msmarco.per_query.pt"
TRAINED_PATH = ROOT / "outputs/full_msmarco/subbit_full_msmarco.per_query.pt"
OUT_PATH = ROOT / "outputs/full_msmarco/bootstrap_r64_8m.json"

N_BOOT = 10_000
SEED = 42
METRICS = ("rr@10", "recall@100", "recall@1000")


def load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run the corresponding full_scale/eval_*_full.py first."
        )
    return torch.load(path, weights_only=False, map_location="cpu")


def align(
    d_rand: dict, d_trained: dict
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    qids_rand = list(d_rand["qids"])
    qids_trained = list(d_trained["qids"])
    if qids_rand == qids_trained:
        order_rand = np.arange(len(qids_rand))
        order_trained = np.arange(len(qids_trained))
        qids = np.array(qids_rand)
    else:
        idx_rand = {q: i for i, q in enumerate(qids_rand)}
        idx_trained = {q: i for i, q in enumerate(qids_trained)}
        common = sorted(set(qids_rand) & set(qids_trained))
        order_rand = np.array([idx_rand[q] for q in common])
        order_trained = np.array([idx_trained[q] for q in common])
        qids = np.array(common)

    # Drop queries without qrels on either side.
    has_rand = np.asarray(d_rand["has_qrels"], dtype=bool)[order_rand]
    has_trained = np.asarray(d_trained["has_qrels"], dtype=bool)[order_trained]
    keep = has_rand & has_trained
    order_rand = order_rand[keep]
    order_trained = order_trained[keep]
    qids = qids[keep]

    rand_metrics: dict[str, np.ndarray] = {}
    trained_metrics: dict[str, np.ndarray] = {}
    for m in METRICS:
        rand_metrics[m] = np.asarray(d_rand[m], dtype=np.float64)[order_rand]
        trained_metrics[m] = np.asarray(d_trained[m], dtype=np.float64)[order_trained]
    return qids, rand_metrics, trained_metrics


def paired_bootstrap_ci(
    a: np.ndarray, b: np.ndarray, n_boot: int, seed: int
) -> dict:
    """Paired bootstrap 95% CI on mean(b - a)."""
    assert a.shape == b.shape, (a.shape, b.shape)
    n = a.shape[0]
    rng = np.random.default_rng(seed)
    delta = b - a
    point = float(delta.mean())
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
    print(f"Loading {TRAINED_PATH.name} ...")
    d_trained = load(TRAINED_PATH)

    qids, rand_m, trained_m = align(d_rand, d_trained)
    print(
        f"Aligned n={len(qids)} queries with qrels on both sides "
        f"(rand sidecar: {len(d_rand['qids'])}, "
        f"trained sidecar: {len(d_trained['qids'])})"
    )

    results = {
        "method_a": d_rand.get("method", "rand_plus_scale"),
        "method_b": d_trained.get("method", "subbit_trained"),
        "comparison": "subbit_trained - rand_plus_scale (paired bootstrap)",
        "r": 64,
        "scale": "8.8M MS MARCO Passage dev",
        "n_queries": int(len(qids)),
        "n_boot": args.n_boot,
        "seed": args.seed,
        "metrics": {},
    }

    for m in METRICS:
        out = paired_bootstrap_ci(rand_m[m], trained_m[m], args.n_boot, args.seed)
        results["metrics"][m] = out
        print(
            f"  {m}: rand+scale={out['mean_a']:.4f}, "
            f"trained={out['mean_b']:.4f}, "
            f"Δ={out['point']:+.4f} [{out['ci_lo']:+.4f}, {out['ci_hi']:+.4f}]"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.output}")

    mrr = results["metrics"]["rr@10"]
    print(
        f"\nHEADLINE: r=64 8.8M, trained R - random orthogonal R + scale, "
        f"ΔMRR@10 = {mrr['point']:+.4f} "
        f"[{mrr['ci_lo']:+.4f}, {mrr['ci_hi']:+.4f}] "
        f"(paired bootstrap, n={mrr['n_queries']}, n_boot={mrr['n_boot']})"
    )


if __name__ == "__main__":
    main()
