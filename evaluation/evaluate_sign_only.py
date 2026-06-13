"""Evaluate sign-of-fp32-vector MaxSim retrieval at r=128 (no projection).

A direct test of the argmax-floor claim: each document token
d is stored as sign(d) in {-1,+1}^128 (16 B/tok, 8x over INT8 ColBERTv2);
each query token stays fp32; per-q-token MaxSim sums over doc tokens.

Runs on both ColBERTv2 and ConstBERT 100k MS MARCO Passage Dev embeddings.
Compares against fp32 baselines.

Outputs: outputs/sign_only_eval.json
"""
from __future__ import annotations
import argparse, json, pickle, sys, time
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(".").resolve()))
from src.subbit.data import load_qrels


def stack_corpus_variable(doc_embs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, list[str], torch.Tensor]:
    """ColBERTv2-style: variable K. Pad to common K with zeros, track lengths."""
    pids = sorted(doc_embs.keys())
    Ks = [doc_embs[p].shape[0] for p in pids]
    Kmax = max(Ks)
    dim = doc_embs[pids[0]].shape[1]
    out = torch.zeros(len(pids), Kmax, dim, dtype=torch.float32)
    for i, p in enumerate(pids):
        e = doc_embs[p]
        out[i, : e.shape[0]] = e
    return out, pids, torch.tensor(Ks, dtype=torch.int32)


def stack_corpus_fixed(doc_embs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, list[str]]:
    pids = sorted(doc_embs.keys())
    out = torch.stack([doc_embs[p] for p in pids], dim=0)
    return out, pids


def eval_maxsim_signed(
    docs: torch.Tensor,             # (N, K, dim)
    doc_lens: torch.Tensor | None,  # (N,) or None for fixed K
    pids: list[str],
    qry_embs: dict[str, torch.Tensor],
    qrels: dict[str, dict[str, int]],
    device: torch.device,
    doc_chunk: int = 4096,
) -> tuple[float, float, float, dict[str, int]]:
    """Return (MRR@10, R@100, R@1000, ranks[qid]).

    docs is fp32 (we sign() on the fly per chunk so the source tensor stays
    small in CPU memory). Queries are fp32. Score is asymmetric: q dot sign(d).
    """
    N, K, dim = docs.shape
    judged = sorted(set(qrels.keys()) & set(qry_embs.keys()))
    ranks: dict[str, int] = {}
    mrr10 = 0.0
    r100 = 0.0
    r1000 = 0.0
    n_with_rel = 0
    pbar = tqdm(judged, desc="sign(d) r=128", unit="q", smoothing=0.1, mininterval=0.5)
    for qid in pbar:
        q = qry_embs[qid].to(device)
        scores = torch.empty(N, device=device, dtype=torch.float32)
        for s in range(0, N, doc_chunk):
            e = min(s + doc_chunk, N)
            chunk_fp32 = docs[s:e].to(device, non_blocking=True)
            chunk_sign = torch.sign(chunk_fp32)                  # (b, K, dim)
            sim = torch.einsum("td,bkd->tbk", q, chunk_sign)     # (n_q, b, K)
            if doc_lens is not None:
                lens = doc_lens[s:e].to(device)
                mask = torch.arange(K, device=device)[None, :] < lens[:, None]  # (b, K)
                sim = sim.masked_fill(~mask[None, :, :], float("-inf"))
            scores[s:e] = sim.amax(dim=-1).sum(dim=0)
            del sim, chunk_fp32, chunk_sign
        order = torch.argsort(scores, descending=True)
        top1000_idx = order[: min(1000, N)].cpu().tolist()
        top1000_pids = [pids[i] for i in top1000_idx]
        rel = {p for p, v in qrels[qid].items() if v > 0}
        if not rel:
            continue
        n_with_rel += 1
        for rank, p in enumerate(top1000_pids[:10], start=1):
            if p in rel:
                mrr10 += 1.0 / rank
                break
        r100 += sum(1 for p in top1000_pids[:100] if p in rel) / len(rel)
        r1000 += sum(1 for p in top1000_pids[:1000] if p in rel) / len(rel)
        best = N + 1
        for rank, p in enumerate(top1000_pids, start=1):
            if p in rel:
                best = rank; break
        ranks[qid] = best
        if n_with_rel % 50 == 0:
            pbar.set_postfix(
                MRR10=f"{mrr10/n_with_rel:.4f}",
                R100=f"{r100/n_with_rel:.4f}",
                R1k=f"{r1000/n_with_rel:.4f}",
            )
    return mrr10 / n_with_rel, r100 / n_with_rel, r1000 / n_with_rel, ranks


def run_encoder(name: str, emb_dir: Path, fixed_K: bool, device: torch.device) -> dict:
    print(f"\n=== {name} ({emb_dir}) ===", flush=True)
    t0 = time.perf_counter()
    docs_raw = torch.load(emb_dir / "doc_embeddings.pt", weights_only=False)
    qs = torch.load(emb_dir / "query_embeddings.pt", weights_only=False)
    qrels = load_qrels(emb_dir / "qrels.tsv")
    if fixed_K:
        docs, pids = stack_corpus_fixed(docs_raw)
        doc_lens = None
    else:
        docs, pids, doc_lens = stack_corpus_variable(docs_raw)
    N, K, dim = docs.shape
    print(f"  docs: N={N} K={K} dim={dim} (loaded in {time.perf_counter()-t0:.1f}s)", flush=True)
    print(f"  fp32 baseline (sign-off comparison reference)...", flush=True)
    # Same eval with sign() turned off, i.e. plain dot-product MaxSim.
    t_fp = time.perf_counter()
    # Computed via _eval_fp32_baseline, which mirrors the eval protocol with
    # sign() removed, so the baseline matches the signed eval exactly.
    mrr10_fp, r100_fp, r1000_fp, _ = _eval_fp32_baseline(
        docs, doc_lens, pids, qs, qrels, device,
    )
    print(f"  fp32 done in {time.perf_counter()-t_fp:.1f}s: MRR@10={mrr10_fp:.4f}  R@100={r100_fp:.4f}  R@1000={r1000_fp:.4f}", flush=True)

    print(f"  sign(d) eval ...", flush=True)
    t_sign = time.perf_counter()
    mrr10, r100, r1000, ranks = eval_maxsim_signed(docs, doc_lens, pids, qs, qrels, device)
    print(f"  sign(d) done in {time.perf_counter()-t_sign:.1f}s: MRR@10={mrr10:.4f}  R@100={r100:.4f}  R@1000={r1000:.4f}", flush=True)
    retention = mrr10 / mrr10_fp * 100
    print(f"  retention vs fp32: {retention:.1f}%   (delta {mrr10 - mrr10_fp:+.4f})", flush=True)

    return {
        "encoder": name,
        "n_docs": N,
        "K": K,
        "dim": dim,
        "n_queries_evaluated": len(ranks),
        "fp32": {"mrr@10": mrr10_fp, "recall@100": r100_fp, "recall@1000": r1000_fp},
        "sign_d_r128": {"mrr@10": mrr10, "recall@100": r100, "recall@1000": r1000},
        "retention_pct": retention,
        "bytes_per_token": dim // 8,
    }


def _eval_fp32_baseline(docs, doc_lens, pids, qry_embs, qrels, device, doc_chunk=4096):
    """Plain fp32 MaxSim baseline; matches eval_maxsim_signed protocol exactly
    except no sign() applied."""
    N, K, dim = docs.shape
    judged = sorted(set(qrels.keys()) & set(qry_embs.keys()))
    mrr10 = r100 = r1000 = 0.0
    n = 0
    pbar = tqdm(judged, desc="fp32 baseline", unit="q", smoothing=0.1, mininterval=0.5)
    for qid in pbar:
        q = qry_embs[qid].to(device)
        scores = torch.empty(N, device=device, dtype=torch.float32)
        for s in range(0, N, doc_chunk):
            e = min(s + doc_chunk, N)
            chunk = docs[s:e].to(device, non_blocking=True)
            sim = torch.einsum("td,bkd->tbk", q, chunk)
            if doc_lens is not None:
                lens = doc_lens[s:e].to(device)
                mask = torch.arange(K, device=device)[None, :] < lens[:, None]
                sim = sim.masked_fill(~mask[None, :, :], float("-inf"))
            scores[s:e] = sim.amax(dim=-1).sum(dim=0)
            del sim, chunk
        order = torch.argsort(scores, descending=True)
        top1000 = [pids[i] for i in order[: min(1000, N)].cpu().tolist()]
        rel = {p for p, v in qrels[qid].items() if v > 0}
        if not rel: continue
        n += 1
        for rank, p in enumerate(top1000[:10], start=1):
            if p in rel:
                mrr10 += 1.0 / rank; break
        r100 += sum(1 for p in top1000[:100] if p in rel) / len(rel)
        r1000 += sum(1 for p in top1000[:1000] if p in rel) / len(rel)
        if n % 50 == 0:
            pbar.set_postfix(
                MRR10=f"{mrr10/n:.4f}",
                R100=f"{r100/n:.4f}",
                R1k=f"{r1000/n:.4f}",
            )
    return mrr10/n, r100/n, r1000/n, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    ap.add_argument("--output", type=Path, default=Path("outputs/sign_only_eval.json"))
    args = ap.parse_args()
    device = torch.device(args.device)
    print(f"device: {device}", flush=True)

    results = {}
    results["colbertv2"] = run_encoder(
        "ColBERTv2",
        Path("data/embeddings/msmarco/100k"),
        fixed_K=False,
        device=device,
    )
    constbert_dir = Path("data/embeddings/constbert/100k")
    if constbert_dir.exists():
        results["constbert"] = run_encoder(
            "ConstBERT",
            constbert_dir,
            fixed_K=True,
            device=device,
        )
    else:
        print(
            f"  [skip] ConstBERT cache not found at {constbert_dir}; "
            "skipping ConstBERT (not bundled in this release).",
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"\n=== SUMMARY ===")
    for k, v in results.items():
        rt = v["retention_pct"]
        print(f"{k:12s} fp32 MRR@10={v['fp32']['mrr@10']:.4f}  sign(d) r=128 MRR@10={v['sign_d_r128']['mrr@10']:.4f}  retention={rt:.1f}%")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
