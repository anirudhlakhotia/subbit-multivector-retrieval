"""fig:pareto — storage--quality Pareto, all baselines, augmented 100k numbers.

Replaces the trained-vs-random Δ-CI plot in §5 with the all-baselines
storage-quality frontier (the "old good" pareto plot, regenerated on the
augmented re-baseline with a clean palette).

x = document storage (bytes/token, log), y = MRR@10 on the 100k diagnostic slice.
The hero is the training-free recipe: random orthogonal R at 8 B/tok + an fp32
top-100 rerank, which reaches fp32 quality (0.8641) at 64x less storage and
dominates the PLAID residual-code family.

All numbers are the augmented values from Table~\ref{tab:pareto} / outputs/aug_eval:
  fp32 ColBERTv2        512 B/tok  0.8642   (128 dims x 4 B; baseline_100k_aug_r64.json FP128)
  PLAID b=4 / b=2 / b=1  66/34/18  0.8627/0.8593/0.8498  (table_plaid_noscale_aug.json)
  RaBitQ                 24 B/tok  0.8582   (rabitq_100k_aug.json)
  8 B/tok single-stage:  trained 0.8496, random 0.8393, identity 0.8367,
                         PQ 0.8330, ITQ 0.8319, OPQ 0.8048
  recipe (random+rerank) 8 B/tok  0.8641   (rerank_aug_fullfp32_randomR.json K=100)

Output: paper/figures/pareto_quality_vs_storage.pdf (+ .png)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("paper/figures")
OUT.mkdir(parents=True, exist_ok=True)

# clean palette
C_FP32 = "#444444"      # fp32 reference
C_PLAID = "#ff7f0e"     # PLAID residual-code family
C_RABITQ = "#9467bd"    # RaBitQ
C_BASE = "#9aa0a6"      # 8 B/tok single-stage baselines
C_HERO = "#1f77b4"      # the recipe (sign-coded + rerank)
C_FRONTIER = "#1f77b4"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
})

fig, ax = plt.subplots(figsize=(3.5, 2.7))

# PLAID family (connected line)
plaid = [(18, 0.8498, "$b{=}1$"), (34, 0.8593, "$b{=}2$"), (66, 0.8627, "$b{=}4$")]
ax.plot([p[0] for p in plaid], [p[1] for p in plaid], "-o", color=C_PLAID,
        ms=3.5, lw=1.3, zorder=3, label="PLAID ($b{=}1,2,4$)")
for bx, by, bl in plaid:
    ax.annotate(bl, (bx, by), xytext=(0, 6), textcoords="offset points",
                ha="center", fontsize=7, color=C_PLAID)

# fp32 reference
ax.plot(512, 0.8642, "D", color=C_FP32, ms=3.5, zorder=3, label="fp32 ColBERTv2")
ax.annotate("fp32", (512, 0.8642), xytext=(-6, 4), textcoords="offset points",
            ha="right", fontsize=7.5, color=C_FP32)

# RaBitQ
ax.plot(24, 0.8582, "s", color=C_RABITQ, ms=4, zorder=3, label="RaBitQ ($24$ B)")

# 8 B/tok single-stage codes (grey cluster): trained R and random R are the
# section's two arms; the rest are matched-byte single-vector baselines.
# This is a single-stage storage-quality plot -- the fp32 rerank is a separate
# two-stage step (deferred to the PLAID/rerank section) and is NOT shown here,
# since its quality needs the full-precision store, not the 8 B/tok index.
base8 = [("trained $R$", 0.8496), ("random $R$", 0.8393), ("identity", 0.8367),
         ("PQ, ITQ", 0.8330), ("", 0.8319), ("OPQ", 0.8048)]
ax.scatter([8] * len(base8), [b[1] for b in base8], s=22, color=C_BASE, zorder=3,
           label="single-stage $8$ B/tok")
for lab, y in base8:
    if lab:
        ax.annotate(lab, (8, y), xytext=(7, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=6.6, color="#666666")

ax.set_xscale("log")
ax.set_xticks([8, 18, 24, 34, 66, 512])
ax.set_xticklabels(["8", "18", "24", "34", "66", "512"])
ax.set_xlim(6.5, 700)
ax.set_ylim(0.795, 0.872)
ax.set_xlabel("document storage (bytes / token)")
ax.set_ylabel("MRR@10  ($100$k diagnostic)")
ax.legend(loc="lower right", frameon=False, fontsize=6.8,
          handletextpad=0.3, labelspacing=0.2, borderpad=0.2)
fig.tight_layout()
fig.savefig(OUT / "pareto_quality_vs_storage.pdf", bbox_inches="tight")
fig.savefig(OUT / "pareto_quality_vs_storage.png", bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT}/pareto_quality_vs_storage.pdf (+ .png)")
