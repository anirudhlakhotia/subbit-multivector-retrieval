"""Minimal MPS-einsum eval of sign(d) at r=128 on ConstBERT 100k MS MARCO.

Pre-computes sign(d) once on CPU (100k * 32 * 128 = 12.8M bits, ~51 MB as
fp32 +-1 on MPS). For each judged query, runs a batched (n_q, 128) x
(b, 32, 128) einsum on MPS, amax over the K=32 doc-vector dim, sum over
q-token dim, argsort. Asymmetric: fp32 query against sign(d) doc.

Output: outputs/sign_only_canonical_constbert.json (matches canonical schema
so it interoperates with the ColBERTv2 result).
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(".").resolve()))
from src.subbit.data import load_qrels


def stack_corpus_fixed(doc_embs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, list[str]]:
    pids = sorted(doc_embs.keys())
    return torch.stack([doc_embs[p] for p in pids], dim=0), pids


def eval_maxsim(
    docs_dev: torch.Tensor,  # (N, K, 128) on device, fp16 +-1 for sign mode or fp32 for baseline
    pids: list[str],
    qry_embs: dict[str, torch.Tensor],
    qrels: dict[str, dict[str, int]],
    device: torch.device,
    doc_chunk: int = 8192,
    label: str = "eval",
) -> tuple[float, float, float]:
    N, K, dim = docs_dev.shape
    judged = sorted(set(qrels.keys()) & set(qry_embs.keys()))
    mrr10 = r100 = r1000 = 0.0
    n = 0
    pbar = tqdm(judged, desc=label, unit="q", smoothing=0.1, mininterval=0.5)
    for qid in pbar:
        q = qry_embs[qid].to(device, dtype=docs_dev.dtype)        # (n_q, 128)
        # Score all N docs in chunks
        scores = torch.empty(N, device=device, dtype=torch.float32)
        for s in range(0, N, doc_chunk):
            e = min(s + doc_chunk, N)
            chunk = docs_dev[s:e]                                 # (b, K, 128)
            sim = torch.einsum("td,bkd->tbk", q, chunk)           # (n_q, b, K)
            scores[s:e] = sim.amax(dim=-1).sum(dim=0).to(torch.float32)
            del sim
        order = torch.argsort(scores, descending=True)
        top1000 = [pids[i] for i in order[: min(1000, N)].cpu().tolist()]
        rel = {p for p, v in qrels[qid].items() if v > 0}
        if not rel:
            continue
        n += 1
        for rank, p in enumerate(top1000[:10], start=1):
            if p in rel:
                mrr10 += 1.0 / rank
                break
        r100 += sum(1 for p in top1000[:100] if p in rel) / len(rel)
        r1000 += sum(1 for p in top1000[:1000] if p in rel) / len(rel)
        if n % 100 == 0:
            pbar.set_postfix(MRR10=f"{mrr10/n:.4f}", R100=f"{r100/n:.4f}", R1k=f"{r1000/n:.4f}")
    return mrr10 / n, r100 / n, r1000 / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dir", type=Path, default=Path("data/embeddings/constbert/100k"))
    ap.add_argument("--output", type=Path, default=Path("outputs/sign_only_canonical_constbert.json"))
    ap.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"device: {device}", flush=True)

    t0 = time.perf_counter()
    print(f"loading {args.emb_dir}...", flush=True)
    docs_raw = torch.load(args.emb_dir / "doc_embeddings.pt", weights_only=False, map_location="cpu")
    qry_embs = torch.load(args.emb_dir / "query_embeddings.pt", weights_only=False, map_location="cpu")
    qrels = load_qrels(args.emb_dir / "qrels.tsv")
    docs_fp32, pids = stack_corpus_fixed(docs_raw)
    N, K, dim = docs_fp32.shape
    print(f"  N={N}, K={K}, dim={dim}, qrels={len(qrels)} ({time.perf_counter()-t0:.1f}s)", flush=True)

    # === fp32 baseline ===
    print("\n--- fp32 baseline (FP128) ---", flush=True)
    docs_fp32_dev = docs_fp32.to(device)
    print(f"  docs on {device}: {docs_fp32_dev.numel()*4/1e9:.2f} GB fp32", flush=True)
    mrr10_fp, r100_fp, r1000_fp = eval_maxsim(
        docs_fp32_dev, pids, qry_embs, qrels, device, label="FP128",
    )
    print(f"  FP128: MRR@10={mrr10_fp:.4f}  R@100={r100_fp:.4f}  R@1000={r1000_fp:.4f}", flush=True)

    del docs_fp32_dev
    torch.mps.empty_cache() if device.type == "mps" else None

    # === sign(d) r=128 ===
    print("\n--- identity r=128 (sign(d) direct) ---", flush=True)
    docs_sign = torch.sign(docs_fp32).to(torch.float16)
    # Replace zeros with +1 (rare; ColBERTv2-style unit-norm vectors have no zero coords)
    docs_sign = torch.where(docs_sign == 0, torch.ones_like(docs_sign), docs_sign)
    docs_sign_dev = docs_sign.to(device)
    print(f"  docs on {device}: {docs_sign_dev.numel()*2/1e9:.2f} GB fp16 sign", flush=True)
    mrr10_s, r100_s, r1000_s = eval_maxsim(
        docs_sign_dev, pids, qry_embs, qrels, device, label="sign(d) r=128",
    )
    print(f"  sign(d) r=128: MRR@10={mrr10_s:.4f}  R@100={r100_s:.4f}  R@1000={r1000_s:.4f}", flush=True)
    retention = mrr10_s / mrr10_fp * 100
    print(f"  retention: {retention:.1f}% (delta {mrr10_s - mrr10_fp:+.4f})", flush=True)

    out = {
        "run_metadata": {
            "encoder": "pinecone/ConstBERT",
            "corpus": "MS MARCO Passage Dev 100k",
            "n_docs": N, "K": K, "dim": dim, "n_queries": len(qrels),
            "eval_path": "MPS einsum, fp16 sign codes (asymmetric: fp32 q vs sign(d))",
        },
        "rows": [
            {
                "meta": {"label": "FP128", "r": 128, "bytes_per_token": 4 * dim, "compression": 1, "trained": False},
                "metrics": {"mrr@10": mrr10_fp, "recall@100": r100_fp, "recall@1000": r1000_fp},
            },
            {
                "meta": {"label": "identity r=128", "r": 128, "bytes_per_token": dim // 8, "compression": 32, "trained": False},
                "metrics": {"mrr@10": mrr10_s, "recall@100": r100_s, "recall@1000": r1000_s},
                "retention_pct": retention,
            },
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
