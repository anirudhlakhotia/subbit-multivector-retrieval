#!/usr/bin/env python3
"""Per-corpus paired-bootstrap CIs + multiple-comparison correction for the
BEIR rotation-invariance panel (sign(d) vs random orthogonal R, r=64, NDCG@10).

Reads the per-query NDCG@10 arrays released as .per_query.pt sidecars (the same
arrays the paper's Table 2 caption says CIs are "computable from"), and reports,
per corpus: Delta NDCG@10, the 95% percentile-bootstrap CI on the paired mean
difference, a two-sided paired test p-value, and Holm + Bonferroni adjusted
p-values across the nine corpora.

Usage:
    python statistics/compute_beir_ci.py \
        --indir outputs/beir_ci/r64 \
        --out outputs/beir_ci/beir9_r64_ci.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

SIGN_KEY = "sign_d_r64"
RAND_KEY = "random_R_r64_sign"
METRIC = "ndcg@10"
N_BOOT = 10000
SEED = 0


def find_sidecars(indir: Path):
    files = sorted(indir.rglob("beir_*_sign_vs_random_r64.per_query.pt"))
    out = {}
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        out[d["dataset"]] = d
    return out


def extract_arms(d):
    """Return (sign_ndcg, rand_ndcg, has_qrels) handling both sidecar schemas."""
    if "per_method" in d:  # canonical schema
        pm = d["per_method"]
        sign = np.asarray(pm[SIGN_KEY][METRIC], dtype=float)
        rand = np.asarray(pm[RAND_KEY][METRIC], dtype=float)
        qrel = np.asarray(pm[SIGN_KEY]["has_qrels"], dtype=bool)
    else:  # older nq schema: d['sign_d']['pq'][metric], d['random_R']['pq'][metric]
        sign = np.asarray(d["sign_d"]["pq"][METRIC], dtype=float)
        rand = np.asarray(d["random_R"]["pq"][METRIC], dtype=float)
        qrel = np.ones(len(sign), dtype=bool)
    return sign, rand, qrel


def percentile_ci(diffs: np.ndarray, rng: np.random.Generator, n_boot: int):
    n = len(diffs)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        means[b] = diffs[idx].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    # two-sided bootstrap p: 2 * min(P(mean<=0), P(mean>=0))
    p_lo = (means <= 0).mean()
    p_hi = (means >= 0).mean()
    p_boot = min(1.0, 2.0 * min(p_lo, p_hi))
    return float(lo), float(hi), float(p_boot)


def holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="outputs/beir_ci/r64")
    ap.add_argument("--out", default="outputs/beir_ci/beir9_r64_ci.json")
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    data = find_sidecars(Path(args.indir))
    assert len(data) == 9, f"expected 9 corpora, found {sorted(data)}"

    rows = []
    for ds, d in data.items():
        sign, rand, qrel = extract_arms(d)
        sign, rand = sign[qrel], rand[qrel]
        diffs = sign - rand
        lo, hi, p_boot = percentile_ci(diffs, rng, N_BOOT)
        rows.append({
            "corpus": ds,
            "n_q": int(qrel.sum()),
            "sign_ndcg": round(float(sign.mean()), 4),
            "rand_ndcg": round(float(rand.mean()), 4),
            "delta": round(float(diffs.mean()), 4),
            "ci_lo": round(lo, 4),
            "ci_hi": round(hi, 4),
            "p_boot": p_boot,
            "ci_excludes_0": bool(lo > 0 or hi < 0),
        })

    pvals = np.array([r["p_boot"] for r in rows])
    p_holm = holm(pvals)
    p_bonf = np.minimum(1.0, pvals * len(pvals))
    for r, ph, pb in zip(rows, p_holm, p_bonf):
        r["p_holm"] = round(float(ph), 4)
        r["p_bonf"] = round(float(pb), 4)
        r["sig_holm_05"] = bool(ph < 0.05)

    rows.sort(key=lambda r: -abs(r["delta"]))
    summary = {
        "metric": METRIC,
        "comparison": "sign(d) - random orthogonal R, r=64",
        "n_boot": N_BOOT,
        "seed": SEED,
        "n_corpora": len(rows),
        "n_sig_holm_05": int(sum(r["sig_holm_05"] for r in rows)),
        "n_ci_excludes_0": int(sum(r["ci_excludes_0"] for r in rows)),
        "median_abs_delta": round(float(np.median([abs(r["delta"]) for r in rows])), 4),
        "rows": rows,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)

    hdr = f'{"corpus":18s}{"n_q":>6s}{"sign":>8s}{"rand":>8s}{"Δ":>9s}{"95% CI":>20s}{"p_boot":>9s}{"p_holm":>9s}'
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        ci = f'[{r["ci_lo"]:+.3f},{r["ci_hi"]:+.3f}]'
        star = " *" if r["sig_holm_05"] else ""
        print(f'{r["corpus"]:18s}{r["n_q"]:>6d}{r["sign_ndcg"]:>8.3f}{r["rand_ndcg"]:>8.3f}'
              f'{r["delta"]:>+9.3f}{ci:>20s}{r["p_boot"]:>9.3f}{r["p_holm"]:>9.3f}{star}')
    print("-" * len(hdr))
    print(f'corpora with CI excluding 0: {summary["n_ci_excludes_0"]}/9; '
          f'significant after Holm(.05): {summary["n_sig_holm_05"]}/9; '
          f'median |Δ| = {summary["median_abs_delta"]}')
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
