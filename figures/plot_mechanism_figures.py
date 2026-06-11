"""Regenerate the two Mechanism (§7) paper figures from the AUGMENTED c6 substrate.

Both figures are reconstructed offline (no new run) from
``outputs/c6_flip_residuals_aug100k_r64.npz`` (aug-100k, r=64, random orthogonal
R, seed 42, 6,980 dev queries, 223,360 q-tokens). This replaces the stale
pre-aug renders whose panel annotations (n=39,401 / 606 / 600/606 / 0.30 / etc.)
contradicted the augmented §7 caption/prose.

Outputs (overwrites the stale figures in place):
  paper/figures/fig_argmax_margin.pdf      (+ .png)
  paper/figures/fig_low_margin_residual.pdf (+ .png)

Annotations are computed from the npz so figure == caption == artifact:
  fig_argmax_margin     : 93% below 0.05; tail >=0.2 = 1,116 (0.50%);
                          preserved >=0.2 = 1,090/1,116 = 97.7%; overall 0.29.
  fig_low_margin_residual: 158,452 flipped low-margin; median delta -0.0353;
                          62.0% within 0.05; per-query median loss 4.2%;
                          98.6% lose <= 10%.

Run:
    python figures/plot_mechanism_figures.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NPZ = Path("outputs/c6_flip_residuals_aug100k_r64.npz")
FIGDIR = Path("paper/figures")
HIGH = 0.2
EPS = 0.003  # log-axis display floor for the [0, .) bucket

_RC = {
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7.5,
    "axes.linewidth": 0.6, "lines.linewidth": 1.2, "lines.markersize": 4,
    "figure.dpi": 150,
}
_BOX = dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.88, edgecolor="none")
_BLUE, _DBLUE, _RED = "#3878a8", "#1f4e7f", "#aa1f1f"


def _save(fig, stem: str) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    out = FIGDIR / f"{stem}.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", pad_inches=0.04, dpi=300)
    plt.close(fig)
    print(f"wrote {out}")


def fig_argmax_margin(margin, preserved) -> None:
    """Panel (a): margin distribution with high-margin tail shaded.
       Panel (b): argmax-preservation rate vs fp32 margin at r=64."""
    total = len(margin)
    hi = margin >= HIGH
    n_high = int(hi.sum())
    pres_high = float(preserved[hi].mean())
    frac_high = n_high / total
    below_005 = int((margin < 0.05).sum())
    overall = float(preserved.mean())

    # log-spaced margin buckets (shared by both panels)
    edges = np.concatenate([[0.0], np.logspace(np.log10(0.01), np.log10(1.2), 24)])
    ns, _ = np.histogram(margin, bins=edges)
    rate = np.array([
        preserved[(margin >= edges[i]) & (margin < edges[i + 1])].mean()
        if ns[i] > 0 else np.nan
        for i in range(len(edges) - 1)
    ])
    disp_edges = edges.copy()
    disp_edges[0] = EPS  # show the [0, .01) bucket on the log axis
    centers = np.sqrt(disp_edges[:-1] * disp_edges[1:])

    plt.rcParams.update(_RC)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.7), dpi=150)

    # ---- (a) margin distribution ----
    ax1.stairs(ns, disp_edges, fill=True, color=_BLUE, alpha=0.55,
               edgecolor=_DBLUE, linewidth=0.8)
    ax1.axvspan(HIGH, disp_edges[-1], color=_RED, alpha=0.10, linewidth=0)
    ax1.axvline(HIGH, color=_RED, linestyle="--", alpha=0.6, linewidth=0.8)
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlim(EPS, 1.2)
    ax1.set_xlabel(r"fp32 argmax margin (top-1 $-$ top-2)")
    ax1.set_ylabel("q-tokens (count)")
    ax1.set_title("(a) margins concentrate at the low end")
    ax1.text(0.055, ns.max() * 0.50,
             f"{100 * below_005 / total:.0f}% of q-tokens\nbelow margin 0.05",
             fontsize=7, color=_DBLUE, va="center", ha="center", bbox=_BOX)
    tail_h = ns[np.searchsorted(edges, HIGH, side="right") - 1]
    ax1.annotate(f"high-margin tail\n$\\geq 0.2$: {n_high:,} ({100 * frac_high:.2f}%)",
                 xy=(0.30, max(tail_h, 1)), xytext=(0.42, ns.max() * 0.30),
                 fontsize=7, color=_RED, ha="center", va="center", bbox=_BOX,
                 arrowprops=dict(arrowstyle="-", color=_RED, lw=0.6))
    ax1.grid(alpha=0.25, linewidth=0.4, which="major")

    # ---- (b) preservation vs margin ----
    ok = ~np.isnan(rate)
    ax2.plot(centers[ok], rate[ok], "o-", color=_DBLUE, markersize=3.5, linewidth=1.2)
    ax2.axvline(HIGH, color=_RED, linestyle="--", alpha=0.6, linewidth=0.8)
    ax2.axhline(overall, color="#888888", linestyle=":", alpha=0.8, linewidth=0.8)
    ax2.text(0.21, 0.06, r"margin $= 0.2$", rotation=90, fontsize=7,
             color=_RED, va="bottom", ha="left")
    ax2.text(0.0045, overall + 0.04, f"overall rate {overall:.2f}",
             fontsize=7, color="#666666", va="bottom", ha="left", bbox=_BOX)
    ax2.annotate(f"$\\geq 0.2$: {int(round(pres_high * n_high)):,}/{n_high:,} "
                 f"$=$ {100 * pres_high:.1f}%",
                 xy=(0.42, pres_high), xytext=(0.055, 0.82),
                 fontsize=7, color=_DBLUE, ha="left", va="center", bbox=_BOX,
                 arrowprops=dict(arrowstyle="-", color=_DBLUE, lw=0.6))
    ax2.set_xscale("log"); ax2.set_xlim(EPS, 1.2); ax2.set_ylim(-0.04, 1.06)
    ax2.set_xlabel(r"fp32 argmax margin (top-1 $-$ top-2)")
    ax2.set_ylabel("argmax preservation rate ($r{=}64$)")
    ax2.set_title("(b) high-margin argmaxes survive at $r{=}64$")
    ax2.grid(alpha=0.25, linewidth=0.4, which="major")

    fig.tight_layout(pad=0.4)
    _save(fig, "fig_argmax_margin")
    print(f"  total={total:,} below0.05={below_005:,} ({100*below_005/total:.1f}%) "
          f"high>=0.2={n_high:,} ({100*frac_high:.2f}%) "
          f"pres>=0.2={pres_high*100:.1f}% overall={overall:.4f}")


def fig_low_margin_residual(margin, preserved, delta, qid, qsum) -> None:
    """Panel (left): residual delta for flipped low-margin q-tokens.
       Panel (right): per-query aggregate relative loss of fp32 MaxSim sum."""
    flipped = (margin < HIGH) & (~preserved)
    d = delta[flipped]
    med = float(np.median(d))
    within = float((d >= -0.05).mean())
    n_fl = int(flipped.sum())

    # per-query aggregate: sum(delta) / fp32 MaxSim sum
    order = np.argsort(qid, kind="stable")
    qs = qid[order]; ds = delta[order]; sums = qsum[order]
    uq, idx = np.unique(qs, return_index=True)
    sumdelta = np.add.reduceat(ds, idx)
    qfp32 = sums[idx]
    rel = sumdelta / qfp32  # negative = loss
    rel_med = float(np.median(rel))
    rel_within10 = float((rel >= -0.10).mean())

    plt.rcParams.update(_RC)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.7), dpi=150)

    # ---- (left) per-q-token residual ----
    lo_clip = max(-0.5, d.min())
    bins = np.linspace(lo_clip, 0.0, 60)
    ax1.hist(np.clip(d, lo_clip, 0.0), bins=bins, color=_BLUE, alpha=0.6,
             edgecolor=_DBLUE, linewidth=0.4)
    ax1.axvspan(-0.05, 0.0, color="#2ca02c", alpha=0.10, linewidth=0)
    ax1.axvline(med, color=_RED, linestyle="--", alpha=0.8, linewidth=0.9)
    ax1.set_xlim(lo_clip, 0.01)
    ax1.set_xlabel(r"residual $\delta_i$ (fp32 score: sign winner $-$ fp32 winner)")
    ax1.set_ylabel("flipped low-margin q-tokens")
    ax1.set_title("(a) per-token substitution cost is small")
    ax1.text(0.04, 0.94,
             f"{n_fl:,} flipped low-margin q-tokens\nmedian $\\delta = {med:.4f}$\n"
             f"{100*within:.1f}% within $0.05$ of zero",
             transform=ax1.transAxes, fontsize=7, color=_DBLUE, va="top", ha="left", bbox=_BOX)
    ax1.grid(alpha=0.25, linewidth=0.4, which="major")

    # ---- (right) per-query aggregate loss ----
    lo_clip2 = max(-0.25, rel.min())
    bins2 = np.linspace(lo_clip2, 0.0, 50)
    ax2.hist(np.clip(rel, lo_clip2, 0.0) * 100, bins=bins2 * 100, color=_BLUE, alpha=0.6,
             edgecolor=_DBLUE, linewidth=0.4)
    ax2.axvspan(-10, 0, color="#2ca02c", alpha=0.10, linewidth=0)
    ax2.axvline(rel_med * 100, color=_RED, linestyle="--", alpha=0.8, linewidth=0.9)
    ax2.set_xlim(lo_clip2 * 100, 1.0)
    ax2.set_xlabel("per-query loss of fp32 MaxSim sum (%)")
    ax2.set_ylabel("queries")
    ax2.set_title("(b) summed per query, the loss stays small")
    ax2.text(0.04, 0.94,
             f"median loss ${100*abs(rel_med):.1f}\\%$\n"
             f"{100*rel_within10:.1f}% of queries lose $\\leq 10\\%$",
             transform=ax2.transAxes, fontsize=7, color=_DBLUE, va="top", ha="left", bbox=_BOX)
    ax2.grid(alpha=0.25, linewidth=0.4, which="major")

    fig.tight_layout(pad=0.4)
    _save(fig, "fig_low_margin_residual")
    print(f"  flipped={n_fl:,} median_delta={med:.4f} within0.05={100*within:.1f}% "
          f"perq_median={100*rel_med:.2f}% within10%={100*rel_within10:.1f}% nq={len(uq)}")


def main() -> None:
    z = np.load(NPZ, allow_pickle=True)
    margin = z["margin"].astype(np.float64)
    preserved = z["preserved"].astype(bool)
    delta = z["delta_fp32"].astype(np.float64)
    qid = z["qid"]
    qsum = z["query_fp32_sum"].astype(np.float64)
    print(f"loaded {NPZ}  ({len(margin):,} q-tokens)")
    fig_argmax_margin(margin, preserved)
    fig_low_margin_residual(margin, preserved, delta, qid, qsum)


if __name__ == "__main__":
    main()
