"""
Figure for §8 — "What training actually buys"
4-panel summary: (A) training improves fidelity, (B) more data does not,
(C) fidelity gain ≠ retrieval gain (scatter), (D) gain vanishes at scale.

Usage:
    python figures/plot_training_anatomy.py
Outputs:
    paper/figures/training_anatomy.pdf
    paper/figures/training_anatomy.png
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
DATA_PATH = "outputs/aug_eval/sec8_figure_data.json"
OUT_DIR   = "paper/figures"
os.makedirs(OUT_DIR, exist_ok=True)

with open(DATA_PATH) as fh:
    data = json.load(fh)

scatter      = data["scatter"]
fid_bars     = data["fidelity_bars"]
budget       = data["budget"]
scale_gain   = data["scale_gain"]

drho  = np.array(scatter["drho"])
dmrr  = np.array(scatter["dmrr"])
pearson_r = scatter["pearson_r"]
pearson_p = scatter["pearson_p"]

# ---------------------------------------------------------------------------
# RC params
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.size":           8,
    "axes.titlesize":      8.5,
    "axes.labelsize":      8,
    "xtick.labelsize":     7,
    "ytick.labelsize":     7,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "figure.dpi":          150,
    "font.family":         "sans-serif",
    "axes.linewidth":      0.8,
    "xtick.major.width":   0.8,
    "ytick.major.width":   0.8,
})

# Colour palette
C_RANDOM  = "#9E9E9E"   # medium grey
C_TRAINED = "#2166AC"   # strong blue
C_50K     = "#9E9E9E"
C_2M      = "#4DAC26"   # muted green
C_100K    = "#2166AC"
C_88M     = "#D73027"   # red

# ---------------------------------------------------------------------------
# Layout: 2×2 grid, Panel C gets more vertical space via height_ratios
# We give C a larger cell by using a custom grid.
# Row 0: A, B  — narrow
# Row 1: C, D  — taller (C is the centrepiece)
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(6.8, 5.0))
gs = gridspec.GridSpec(
    2, 2,
    figure=fig,
    hspace=0.55,
    wspace=0.38,
    height_ratios=[1, 1.35],
    left=0.09, right=0.97,
    top=0.94, bottom=0.10,
)

ax_A = fig.add_subplot(gs[0, 0])
ax_B = fig.add_subplot(gs[0, 1])
ax_C = fig.add_subplot(gs[1, 0])
ax_D = fig.add_subplot(gs[1, 1])

# ---------------------------------------------------------------------------
# Panel A — Training improves fidelity
# Grouped bars: random vs trained, for ρ and MRR@10
# ---------------------------------------------------------------------------
rho_random   = fid_bars["metric_rho"]["random"]    # 0.645
rho_trained  = fid_bars["metric_rho"]["trained"]   # 0.740
mrr_random   = fid_bars["metric_mrr"]["random"]    # 0.8393
mrr_trained  = fid_bars["metric_mrr"]["trained"]   # 0.8496

x       = np.array([0, 1])
width   = 0.32
x_rho   = x[0]
x_mrr   = x[1]

bars_r = ax_A.bar([x_rho - width/2, x_mrr - width/2],
                  [rho_random, mrr_random],
                  width, label="Random", color=C_RANDOM, zorder=3)
bars_t = ax_A.bar([x_rho + width/2, x_mrr + width/2],
                  [rho_trained, mrr_trained],
                  width, label="Trained", color=C_TRAINED, zorder=3)

# Value annotations above bars — ρ at 3 dp, MRR@10 at 4 dp (matches the paper text)
for bar, val, fmt in [(bars_r[0], rho_random, "{:.3f}"), (bars_t[0], rho_trained, "{:.3f}"),
                      (bars_r[1], mrr_random, "{:.4f}"), (bars_t[1], mrr_trained, "{:.4f}")]:
    ax_A.text(bar.get_x() + bar.get_width() / 2,
              bar.get_height() + 0.005,
              fmt.format(val), ha="center", va="bottom", fontsize=6.5)

ax_A.set_xticks(x)
ax_A.set_xticklabels(["Spearman ρ", "MRR@10"])
ax_A.set_ylabel("Score")
ax_A.set_ylim(0.0, 1.05)
ax_A.yaxis.grid(True, linewidth=0.5, alpha=0.5, zorder=0)
ax_A.set_title("(A) Training improves fidelity", pad=4)
ax_A.legend(fontsize=6.5, loc="upper left",
            framealpha=0.8, borderpad=0.4, labelspacing=0.3)

# ---------------------------------------------------------------------------
# Panel B — More data does not
# Grouped bars: 50k vs 2M triples for ρ and MRR@10
# ---------------------------------------------------------------------------
pts = budget["points"]
rho_50k  = pts[0]["rho"];  mrr_50k  = pts[0]["mrr"]   # 0.7401, 0.8496
rho_2m   = pts[1]["rho"];  mrr_2m   = pts[1]["mrr"]   # 0.7445, 0.8492

bars_50k = ax_B.bar([x_rho - width/2, x_mrr - width/2],
                    [rho_50k, mrr_50k],
                    width, label="50k triples", color=C_50K, zorder=3)
bars_2m  = ax_B.bar([x_rho + width/2, x_mrr + width/2],
                    [rho_2m, mrr_2m],
                    width, label="2M triples", color=C_2M, zorder=3)

# For the ρ group, the bars are nearly identical height — stagger labels
# to avoid collision: 50k label slightly left+lower, 2M label slightly right+higher
rho_pairs = [(bars_50k[0], rho_50k, -0.01, 0.005),   # (bar, val, x_offset, y_extra)
             (bars_2m[0],  rho_2m,  +0.01, 0.030)]
mrr_pairs = [(bars_50k[1], mrr_50k, 0, 0.005),
             (bars_2m[1],  mrr_2m,  0, 0.005)]

for bar, val, xoff, yoff in rho_pairs:
    ax_B.text(bar.get_x() + bar.get_width() / 2 + xoff,
              bar.get_height() + yoff,
              f"{val:.4f}", ha="center", va="bottom", fontsize=6.5)

for bar, val, xoff, yoff in mrr_pairs:
    ax_B.text(bar.get_x() + bar.get_width() / 2 + xoff,
              bar.get_height() + yoff,
              f"{val:.4f}", ha="center", va="bottom", fontsize=6.5)

ax_B.set_xticks(x)
ax_B.set_xticklabels(["Spearman ρ", "MRR@10"])
ax_B.set_ylabel("Score")
ax_B.set_ylim(0.0, 1.05)
ax_B.yaxis.grid(True, linewidth=0.5, alpha=0.5, zorder=0)
ax_B.set_title("(B) More data does not", pad=4)
ax_B.legend(fontsize=6.5, loc="upper left",
            framealpha=0.8, borderpad=0.4, labelspacing=0.3)

# ---------------------------------------------------------------------------
# Panel C — Fidelity gain ≠ retrieval gain (CENTREPIECE)
# Scatter: Δρ (x) vs ΔMRR@10 (y), 1500 points
# ---------------------------------------------------------------------------
# Faint jittered scatter (background): a tiny vertical jitter spreads the discrete
# ΔMRR@10 bands so the cloud reads as a blob rather than stripes.
_rng = np.random.default_rng(0)
_yj = dmrr + _rng.uniform(-0.03, 0.03, size=len(dmrr))
ax_C.scatter(drho, _yj, s=3, alpha=0.12, color=C_TRAINED, linewidths=0, zorder=2)

# Reference line at y = 0
ax_C.axhline(0, color="#666666", linewidth=0.8, linestyle="--", zorder=1)

# Binned conditional mean of ΔMRR vs Δρ (± SE): flat across Δρ => no relationship.
_nb = 7
_q = np.quantile(drho, np.linspace(0, 1, _nb + 1)); _q[-1] += 1e-9
_bi = np.clip(np.digitize(drho, _q) - 1, 0, _nb - 1)
_xb, _yb, _eb = [], [], []
for _k in range(_nb):
    _m = _bi == _k
    if _m.sum() >= 5:
        _xb.append(float(drho[_m].mean()))
        _yb.append(float(dmrr[_m].mean()))
        _eb.append(float(dmrr[_m].std() / np.sqrt(_m.sum())))
ax_C.errorbar(_xb, _yb, yerr=_eb, fmt="o-", color="#c44e52", ms=3.2, lw=1.3,
              capsize=2, zorder=4, label="binned mean $\\pm$ SE")
ax_C.legend(loc="lower left", fontsize=6.3, frameon=False, handletextpad=0.4)

# Pearson annotation — place it in the upper-right, away from data mass
ax_C.text(0.97, 0.96,
          f"Pearson $r = {pearson_r:.3f}$ ($p = {pearson_p:.2f}$)",
          transform=ax_C.transAxes,
          ha="right", va="top", fontsize=7,
          bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                    edgecolor="#cccccc", linewidth=0.6, alpha=0.9))

ax_C.set_xlabel("Per-query $\\Delta\\rho$ (fidelity gain)", labelpad=3)
ax_C.set_ylabel("Per-query $\\Delta$MRR@10\n(retrieval gain)", labelpad=3)
ax_C.set_title("(C) Fidelity gain $\\neq$ retrieval gain", pad=4)
ax_C.yaxis.grid(True, linewidth=0.4, alpha=0.4, zorder=0)
ax_C.xaxis.grid(True, linewidth=0.4, alpha=0.4, zorder=0)

# ---------------------------------------------------------------------------
# Panel D — Gain vanishes at scale
# Two bars with CI error bars; horizontal line at 0
# ---------------------------------------------------------------------------
sg = scale_gain   # list of two dicts
labels_d = ["100k", "8.8M"]
dmrrs_d  = [sg[0]["dmrr"], sg[1]["dmrr"]]         # [+0.01, -0.0001]
ci_lo    = [sg[0]["ci"][0], sg[1]["ci"][0]]
ci_hi    = [sg[0]["ci"][1], sg[1]["ci"][1]]
yerr_lo  = [d - lo for d, lo in zip(dmrrs_d, ci_lo)]
yerr_hi  = [hi - d for d, hi in zip(dmrrs_d, ci_hi)]

x_d = np.array([0, 1])
bar_colors = [C_100K, C_88M]

for i, (xi, dm, elo, ehi, col, lbl) in enumerate(
        zip(x_d, dmrrs_d, yerr_lo, yerr_hi, bar_colors, labels_d)):
    ax_D.bar(xi, dm, width=0.5,
             color=col, alpha=0.85, zorder=3)
    ax_D.errorbar(xi, dm,
                  yerr=[[elo], [ehi]],
                  fmt="none", color="#333333",
                  capsize=4, capthick=1.0, linewidth=1.0, zorder=4)

ax_D.axhline(0, color="#555555", linewidth=0.9, linestyle="-", zorder=2)

# Annotate the actual values
for xi, dm in zip(x_d, dmrrs_d):
    va  = "bottom" if dm >= 0 else "top"
    off = 0.0007 if dm >= 0 else -0.0007
    ax_D.text(xi, dm + off, f"{dm:+.4f}",
              ha="center", va=va, fontsize=6.8)

ax_D.set_xticks(x_d)
ax_D.set_xticklabels(["100k\ncorpus", "8.8M\ncorpus"])
ax_D.set_ylabel("$\\Delta$MRR@10 (trained $-$ random)", labelpad=3)
ax_D.set_title("(D) Gain vanishes at scale", pad=4)
ax_D.yaxis.grid(True, linewidth=0.5, alpha=0.5, zorder=0)

# Ensure zero line is visible — set sensible ylim
ymax_d = max(ci_hi) * 1.6
ymin_d = min(ci_lo) * 1.6
ax_D.set_ylim(min(ymin_d, -0.005), max(ymax_d, 0.020))

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_pdf = os.path.join(OUT_DIR, "training_anatomy.pdf")
out_png = os.path.join(OUT_DIR, "training_anatomy.png")

fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_png, bbox_inches="tight", dpi=150)
plt.close(fig)

print(f"Saved: {out_pdf}")
print(f"Saved: {out_png}")
