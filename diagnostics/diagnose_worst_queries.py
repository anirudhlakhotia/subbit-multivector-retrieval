"""Worst-5% query decomposition (augmented 100k) for paper §6.

Of the worst 5% of queries by RR@10 drop (fp32 - sign-coded), how many still
hold the relevant doc inside the SubBit top-10 (vs drop out entirely)?
(Pre-aug paper: 349 worst, mean shift -0.580, 325 (93.1%) still in top-10, 24 out.)

rr_sb (sign-coded stage-1 RR@10) comes from the saved per-query artifact (already
validated to reproduce the stored mean 0.8505). rr_fp (full fp32 RR@10) needs the
oracle: full-corpus 128-dim fp32 MaxSim per query -> rank of the relevant doc.

Validation hooks: mean rr_fp must reproduce the fp32 oracle MRR@10 (0.8641 from
rerank_aug_fullfp32.json); mean rr_sb = 0.8505.
"""
from __future__ import annotations
import json, sys, time
import argparse
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from evaluation.evaluate_two_stage_rerank import score_batch_vs_corpus  # identical MaxSim as the artifact

EMB = ROOT / "data/embeddings/msmarco/100k_aug"
PERQ = ROOT / "outputs/aug_eval/scale_sweep_trained_100000.per_query.pt"
OUT = ROOT / "outputs/aug_eval/worst5_fp32_aug100k.json"
MAXT = 180          # matches evaluate_two_stage_rerank max_doc_tokens (so 0.8641 reproduces)
QBATCH = 32
DOC_CHUNK = 4096
WORST_FRAC = 0.05
# CPU: no MPS INT_MAX NDArray limit, so the full 100k-doc fp32 corpus tensor
# (2.28B elements) is fine. Slower than the A100 oracle (~27 min) but faithful.
dev = "cpu"


def load_qrels(path: Path) -> dict[str, set[str]]:
    rel: dict[str, set[str]] = {}
    for ln in open(path):
        p = ln.split()
        if len(p) >= 4: qid, _, did, r = p[:4]
        elif len(p) >= 3: qid, did, r = p[:3]
        else: continue
        if int(r) > 0: rel.setdefault(qid, set()).add(did)
    return rel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=OUT)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"device={dev}")
    blob = torch.load(PERQ, map_location="cpu", weights_only=False)
    qids = [str(q) for q in blob["qids"]]
    rr_sb_all = np.asarray(blob["rr@10"], dtype=np.float64)   # sign-coded stage-1 RR@10
    rel = load_qrels(EMB / "qrels.tsv")

    print("loading doc embeddings (3.5 GB) ...")
    docs = torch.load(EMB / "doc_embeddings.pt", map_location="cpu", weights_only=False)
    doc_pids = list(docs.keys())
    pid2i = {p: i for i, p in enumerate(doc_pids)}
    N = len(doc_pids)

    print(f"building padded fp32 doc tensor (N={N}, MAXT={MAXT}) ...")
    t0 = time.time()
    lengths = [min(docs[p].shape[0], MAXT) for p in doc_pids]
    T = max(lengths)
    doc_raw = torch.zeros(N, T, 128, dtype=torch.float32)
    for i, p in enumerate(doc_pids):
        d = docs[p][:MAXT]
        doc_raw[i, :d.shape[0]] = d
    doc_mask = torch.zeros(N, T, dtype=torch.bool)
    for i, n in enumerate(lengths):
        doc_mask[i, :n] = True
    del docs
    doc_raw = doc_raw.to(dev); doc_mask = doc_mask.to(dev)
    print(f"  doc_raw {tuple(doc_raw.shape)} ~{doc_raw.numel()*4/1e9:.1f} GB in {time.time()-t0:.0f}s")

    queries = torch.load(EMB / "query_embeddings.pt", map_location="cpu", weights_only=False)

    # judged queries only (have >=1 relevant doc present in the 100k corpus)
    judged = [(i, q) for i, q in enumerate(qids)
              if q in rel and any(d in pid2i for d in rel[q])]
    print(f"  {len(judged)} judged queries")

    rr_fp = {}
    t0 = time.time()
    with torch.no_grad():
        for b in range(0, len(judged), QBATCH):
            batch = judged[b:b + QBATCH]
            qmax = max(queries[q].shape[0] for _, q in batch)
            qb = torch.zeros(len(batch), qmax, 128)
            for k, (_, q) in enumerate(batch):
                e = queries[q]; qb[k, :e.shape[0]] = e
            scores = score_batch_vs_corpus(qb.to(dev), doc_raw, doc_mask, DOC_CHUNK)  # (B,N) fp32
            for k, (qi, q) in enumerate(batch):
                s = scores[k]
                rel_idx = [pid2i[d] for d in rel[q] if d in pid2i]
                best = min(int((s > s[j]).sum().item()) + 1 for j in rel_idx)
                rr_fp[qi] = 1.0 / best if best <= 10 else 0.0
            if b % (QBATCH * 20) == 0:
                print(f"    {b}/{len(judged)}  ({(time.time()-t0)/60:.1f} min)")
    print(f"  oracle done in {(time.time()-t0)/60:.1f} min")

    qi_list = [qi for qi, _ in judged]
    fp = np.array([rr_fp[qi] for qi in qi_list])
    sb = np.array([rr_sb_all[qi] for qi in qi_list])

    mean_fp, mean_sb = float(fp.mean()), float(sb.mean())
    print(f"  HOOK: mean fp32 RR@10 = {mean_fp:.4f} (target 0.8641) | "
          f"mean subbit RR@10 = {mean_sb:.4f} (target 0.8505)")

    drop = fp - sb                                  # positive => fp32 better (subbit lost rank)
    n_worst = max(1, round(WORST_FRAC * len(drop)))
    worst = np.argsort(-drop)[:n_worst]             # largest drops
    sb_worst = sb[worst]
    still_in = int((sb_worst > 0).sum())            # relevant still in subbit top-10
    dropped = int((sb_worst == 0).sum())            # relevant fell out of top-10
    mean_shift = float((sb_worst - fp[worst]).mean())  # subbit - fp32 (negative)

    out = {
        "config": {"worst_frac": WORST_FRAC, "n_judged": len(drop), "maxt": MAXT,
                   "embeddings_dir": str(EMB)},
        "validation": {"mean_fp32_rr@10": mean_fp, "target_fp32": 0.8641,
                       "mean_subbit_rr@10": mean_sb, "target_subbit": 0.8505},
        "n_worst": int(n_worst),
        "worst_mean_shift_subbit_minus_fp32": mean_shift,
        "worst_still_in_top10": still_in,
        "worst_still_in_top10_pct": round(100.0 * still_in / n_worst, 1),
        "worst_dropped_out": dropped,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
