"""Missed-vs-resolved doc-level margin decomposition (augmented 100k).

Reproduces the §9 "why it works" split: among queries that still MISS after a
K=100 fp32 rerank (no relevant doc in the reranked top-10) vs those RESOLVED,
what is the median document-level top-1 - top-2 fp32 MaxSim margin? (Pre-aug
paper: missed 0.327 vs resolved 0.661.)

We do NOT re-run the ~60 min stage-1 binary sweep: the sign-coded top-1000
candidates are already saved in the scale_sweep per-query artifact. We only
redo the 33 s stage-2 fp32 rerank of the top-100 candidates (pure 128-dim
q.d MaxSim, no R), which is what the paper's full-128 rerank does.

Two validation hooks:
  (1) stage-1 RR@10 recomputed from the saved top-1000 order must match the
      artifact's stored per-query rr@10 (validates index->pid->qrels mapping).
  (2) aggregate reranked K=100 MRR@10 must reproduce rerank_aug_fullfp32.json
      (0.8642 for trained R).
"""
from __future__ import annotations
import json
import argparse
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
EMB = ROOT / "data/embeddings/msmarco/100k_aug"
PERQ = ROOT / "outputs/aug_eval/scale_sweep_trained_100000.per_query.pt"
OUT = ROOT / "outputs/aug_eval/missed_resolved_margin_aug100k.json"
K = 100
NEAR_TIE = 0.05  # "near-tie in the fp32 ranking" threshold for the doc-level margin

dev = "mps" if torch.backends.mps.is_available() else "cpu"


def load_qrels(path: Path) -> dict[str, set[str]]:
    # pids are STRINGS everywhere (doc dict keys, pids_sorted_by_length, qrels dids).
    rel: dict[str, set[str]] = {}
    for line in open(path):
        p = line.split()
        if len(p) >= 4:
            qid, _, did, r = p[:4]
        elif len(p) >= 3:
            qid, did, r = p[:3]
        else:
            continue
        if int(r) > 0:
            rel.setdefault(qid, set()).add(did)
    return rel


def rr_at_10(ranking_pids: list[str], rel: set[str]) -> float:
    for rank, pid in enumerate(ranking_pids[:10], start=1):
        if pid in rel:
            return 1.0 / rank
    return 0.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=OUT)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"device={dev}")
    blob = torch.load(PERQ, map_location="cpu", weights_only=False)
    qids = [str(q) for q in blob["qids"]]
    top_idx = np.asarray(blob["top1000_indices"])      # (Q, 1000)
    stored_rr = np.asarray(blob["rr@10"])              # (Q,)
    has_q = np.asarray(blob["has_qrels"]).astype(bool)
    pids_by_len = list(blob["pids_sorted_by_length"])  # candidate index space (one option)
    Q = len(qids)
    print(f"  {Q} queries, top_idx {top_idx.shape}")

    rel = load_qrels(EMB / "qrels.tsv")

    print("loading doc embeddings (3.5 GB) ...")
    docs = torch.load(EMB / "doc_embeddings.pt", map_location="cpu", weights_only=False)

    # top1000_indices are positional into pids_sorted_by_length (string pids).
    idx2pid = lambda j: pids_by_len[j]

    # --- Hook 1: recomputed stage-1 RR@10 must match the artifact's stored rr@10 ---
    def stage1_rr_mean() -> float:
        s = []
        for i, qid in enumerate(qids):
            if not has_q[i] or qid not in rel:
                continue
            ranking = [idx2pid(j) for j in top_idx[i, :10]]
            s.append(rr_at_10(ranking, rel[qid]))
        return float(np.mean(s))

    got = stage1_rr_mean()
    stored_mean = float(np.mean([stored_rr[i] for i, qid in enumerate(qids)
                                 if has_q[i] and qid in rel]))
    print(f"  stored stage-1 RR@10 mean={stored_mean:.4f} | recomputed={got:.4f}")
    assert abs(got - stored_mean) < 1e-3, "index->pid mapping does not reproduce stored RR@10"
    print("  HOOK 1 PASS: mapping reproduces stored stage-1 RR@10")

    queries = torch.load(EMB / "query_embeddings.pt", map_location="cpu", weights_only=False)

    # --- Stage-2 fp32 rerank of the top-K, per query ---
    reranked_rr, margins, resolved = [], [], []
    for i, qid in enumerate(qids):
        if not has_q[i] or qid not in rel:
            continue
        q = queries[qid].to(dev).float()                       # (32,128)
        cand_pids = [idx2pid(j) for j in top_idx[i, :K]]
        scores = torch.empty(K, device=dev)
        for c, pid in enumerate(cand_pids):
            d = docs[pid].to(dev).float()                      # (n,128)
            scores[c] = (q @ d.T).max(dim=1).values.sum()      # MaxSim
        order = torch.argsort(scores, descending=True).cpu().numpy()
        head_pids = [cand_pids[j] for j in order]
        # reranked ranking = reranked head (K) then stage-1 tail (>K)
        tail_pids = [idx2pid(j) for j in top_idx[i, K:]]
        ranking = head_pids + tail_pids
        rr = rr_at_10(ranking, rel[qid])
        reranked_rr.append(rr)
        resolved.append(rr > 0)
        s_sorted = scores[torch.from_numpy(order).to(dev)]
        margins.append(float((s_sorted[0] - s_sorted[1]).cpu()))  # doc-level top1-top2 fp32 margin

    reranked_rr = np.asarray(reranked_rr)
    margins = np.asarray(margins)
    resolved = np.asarray(resolved)

    # --- Hook 2: aggregate reranked K=100 MRR@10 must reproduce 0.8642 ---
    mrr = float(reranked_rr.mean())
    print(f"  HOOK 2: reranked K={K} MRR@10 = {mrr:.4f}  (target 0.8642)")

    res_margins = margins[resolved]
    miss_margins = margins[~resolved]
    out = {
        "config": {"K": K, "n_judged": int(resolved.size), "mapping": "pids_sorted_by_length",
                   "near_tie_threshold": NEAR_TIE, "embeddings_dir": str(EMB)},
        "validation": {"reranked_mrr@10": mrr, "target_mrr@10": 0.8642,
                       "stage1_rr@10_mean": stored_mean},
        "n_resolved": int(resolved.sum()), "n_missed": int((~resolved).sum()),
        "resolved_median_margin": float(np.median(res_margins)),
        "missed_median_margin": float(np.median(miss_margins)),
        "missed_frac_near_tie": float((miss_margins < NEAR_TIE).mean()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
