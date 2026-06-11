"""fig for §6 — the shape of preservation across depth, at three levels.

The point is not a tidy 23 -> 49 -> 99.7 gradient (overlap and recall are
different metric families). The point is the contrast in shape: token-geometry
agreement sits low and falls as the neighbourhood grows, while relevant-document
recall sits at the ceiling and rises. Document rankings land in between. The same
r=64 sign code scrambles local geometry yet keeps the relevant document.

Curves (augmented 100k diagnostic slice, r=64):
  Token neighbourhoods (random orth R) -- overlap with fp32 top-k:
    k=1: 0.458, k=10: 0.482, k=100: 0.308, k=1000: 0.231
    (outputs/geometry_fidelity_vs_argmax.json, build_random_R seed=42)
  Document rankings (random orth R) -- overlap with fp32 top-K:
    K=10: 0.612, K=100: 0.504, K=1000: 0.486
    (outputs/aug_eval/preservation_aug_rest.json, Random orth r=64 row)
  Relevant-document retrieval (trained R) -- Rec@K:
    K=10: 0.9668, K=100: 0.9910, K=1000: 0.9972
    (Rec@100/1000 outputs/aug_eval/rerank_aug_fullfp32.json;
     Rec@10 outputs/aug_eval/rec_at_10_trained_aug.json)

Output: paper/figures/three_levels.pdf (+ .png)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("paper/figures")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
})

# (label, colour, marker, [(k, value), ...])
curves = [
    ("Token neighbourhoods (overlap)", "#d62728", "o",
     [(1, 0.458), (10, 0.482), (100, 0.308), (1000, 0.231)]),
    ("Document rankings (overlap)", "#ff7f0e", "s",
     [(10, 0.612), (100, 0.504), (1000, 0.486)]),
    ("Relevant-document retrieval (Rec)", "#2ca02c", "^",
     [(10, 0.9668), (100, 0.9910), (1000, 0.9972)]),
]

fig, ax = plt.subplots(figsize=(4.6, 2.9))
for label, colour, marker, pts in curves:
    ks = [k for k, _ in pts]
    vs = [v * 100 for _, v in pts]
    ax.plot(ks, vs, color=colour, marker=marker, markersize=5,
            linewidth=1.6, label=label, zorder=3)
    # annotate the deepest point (the headline number per level)
    k_end, v_end = pts[-1]
    ax.annotate(f"{v_end*100:.1f}%", (k_end, v_end * 100),
                textcoords="offset points", xytext=(7, -2),
                fontsize=8, color=colour, ha="left", va="center")

ax.set_xscale("log")
ax.set_xticks([1, 10, 100, 1000])
ax.set_xticklabels(["1", "10", "100", "1000"])
ax.set_xlabel("depth $K$ (neighbours / candidates)")
ax.set_ylim(0, 105)
ax.set_yticks([0, 25, 50, 75, 100])
ax.set_ylabel("agreement with fp32 (%)")
ax.set_xlim(0.8, 2200)
ax.legend(loc="lower left", fontsize=7.2, frameon=False,
          bbox_to_anchor=(0.02, 0.02))
fig.tight_layout()
fig.savefig(OUT / "three_levels.pdf", bbox_inches="tight")
fig.savefig(OUT / "three_levels.png", bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT}/three_levels.pdf (+ .png)")
for label, _, _, pts in curves:
    pretty = "  ".join(f"@{k}={v*100:.1f}%" for k, v in pts)
    print(f"  {label:38s} {pretty}")
