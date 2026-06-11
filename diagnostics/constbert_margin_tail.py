"""Quick re-run of ConstBERT argmax-margin computation, dumping the raw
margins array so we can report P(margin >= 0.2) and other tail stats.

Mirrors diagnostics/argmax_margin_constbert.py but writes the full margin
array to outputs/argmax_margin_constbert_raw.npz.
"""
from __future__ import annotations

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

EMB_DIR = Path("data/embeddings/constbert/100k")
OUT_NPZ = Path("outputs/argmax_margin_constbert_raw.npz")
OUT_JSON = Path("outputs/argmax_margin_constbert_tail.json")

N_SAMPLED_QUERIES = 500
TOP_K_CANDIDATES = 100
SEED = 42
DOC_CHUNK = 8192


def main() -> None:
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"device={device}", flush=True)

    doc_emb = torch.load(EMB_DIR / "doc_embeddings.pt", map_location="cpu", weights_only=False)
    qry_emb = torch.load(EMB_DIR / "query_embeddings.pt", map_location="cpu", weights_only=False)
    qrels = load_qrels(EMB_DIR / "qrels.tsv")
    pids = list(doc_emb.keys())
    N = len(pids)
    K_doc = doc_emb[pids[0]].shape[0]
    print(f"  {N} docs, K={K_doc}, {len(qry_emb)} queries", flush=True)

    D = torch.empty(N, K_doc, 128, dtype=torch.float32)
    for i, pid in enumerate(pids):
        D[i] = doc_emb[pid]
    D = D.to(device)

    rng = np.random.default_rng(SEED)
    judged_qids = sorted([q for q in qry_emb if q in qrels and any(s > 0 for s in qrels[q].values())])
    sample_qids = rng.choice(judged_qids, size=min(N_SAMPLED_QUERIES, len(judged_qids)), replace=False)
    print(f"  sampled {len(sample_qids)} judged queries", flush=True)

    margins: list[float] = []
    t0 = time.perf_counter()

    for qi, qid in enumerate(sample_qids):
        q = qry_emb[qid].to(device).float()
        m_q = q.shape[0]
        best_scores = torch.empty(N, dtype=torch.float32, device=device)
        for cs in range(0, N, DOC_CHUNK):
            ce = min(cs + DOC_CHUNK, N)
            sim = torch.einsum("md,nkd->nmk", q, D[cs:ce])
            best_scores[cs:ce] = sim.max(dim=-1).values.sum(dim=-1)
        topk_idx = torch.topk(best_scores, TOP_K_CANDIDATES).indices
        cand_docs = D[topk_idx]
        with torch.no_grad():
            sim_cand = torch.einsum("md,knd->mkn", q, cand_docs)
            sim_flat = sim_cand.reshape(m_q, TOP_K_CANDIDATES * K_doc)
            top_val, top_idx = sim_flat.max(dim=-1)
            sim_clone = sim_flat.clone()
            sim_clone.scatter_(1, top_idx.unsqueeze(-1), float("-inf"))
            runner_val = sim_clone.max(dim=-1).values
            margin = (top_val - runner_val).detach().cpu().numpy()
        margins.extend(margin.tolist())

        if (qi + 1) % 50 == 0 or qi + 1 == len(sample_qids):
            elapsed = time.perf_counter() - t0
            eta = elapsed / (qi + 1) * (len(sample_qids) - qi - 1)
            print(f"  [{qi+1}/{len(sample_qids)}] elapsed={elapsed:.1f}s eta={eta:.1f}s", flush=True)

    margins_arr = np.array(margins, dtype=np.float64)
    n = len(margins_arr)
    n_ge_02 = int((margins_arr >= 0.2).sum())
    n_ge_01 = int((margins_arr >= 0.1).sum())
    n_ge_005 = int((margins_arr >= 0.05).sum())
    summary = {
        "encoder": "pinecone/ConstBERT",
        "corpus": "MS MARCO Passage 100k",
        "n_queries": int(len(sample_qids)),
        "n_qtokens_total": n,
        "fraction_margin_ge_0.2": n_ge_02 / n,
        "fraction_margin_ge_0.1": n_ge_01 / n,
        "fraction_margin_ge_0.05": n_ge_005 / n,
        "n_ge_0.2": n_ge_02,
        "n_ge_0.1": n_ge_01,
        "n_ge_0.05": n_ge_005,
        "median": float(np.median(margins_arr)),
        "mean": float(margins_arr.mean()),
        "p95": float(np.percentile(margins_arr, 95)),
        "p99": float(np.percentile(margins_arr, 99)),
        "max": float(margins_arr.max()),
    }
    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_NPZ, margins=margins_arr)
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {OUT_NPZ}, {OUT_JSON}")


if __name__ == "__main__":
    main()
