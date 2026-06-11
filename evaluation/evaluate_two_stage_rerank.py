"""Two-stage retrieval at low r — BPR/JPQ template adapted for SubBit.

Stage 1: binary MaxSim ``q_proj · sign(R·d)`` over the FULL corpus → top-K
         candidates. Storage: r bits/token.
Stage 2: FULL-PRECISION MaxSim ``q · d`` on the ORIGINAL 128-dim ColBERTv2
         embeddings over the K candidates only → rerank. This is the paper's
         "rescore at full precision" (NOT the r-dim projection R·q·R·d, which
         only recovers the r=64 projected-float ceiling, ~0.01 MRR below fp32).
         Storage: a 128-dim fp32 store (512 B/token) for the stage-2 candidates.

Compared against:

  (a) Binary-only (Stage 1 ranking, current low-r baseline)
  (b) fp32 oracle: full 128-dim ``q · d`` MaxSim over the FULL corpus
      (= the FP128 ColBERTv2 reference; should match tab:pareto's FP128 row).

Implementation notes
--------------------
- Pre-encoding is one batched matmul per chunk (NOT a per-doc Python loop).
  100K docs × 180 tokens × 128 dim → ~10 sec on Mac MPS instead of ~2 min.
- Per-query scoring sweeps doc chunks while batching queries together.
  Each chunk: einsum ``(Q, m, r) × (C, T, r) → (Q, C, m, T)`` then max over
  doc-tokens, sum over query-tokens. fp16 in the einsum keeps the
  intermediate <3 GB.
- Stage-2 rerank only touches top-K candidate docs per query (e.g. top-1000),
  so it costs ~100× less than a full fp32 sweep.

Usage::

    python evaluation/evaluate_two_stage_rerank.py \
        --checkpoint artifacts/checkpoints/50k_topk/best.pt \
        --embeddings-dir data/embeddings/msmarco/100k \
        --max-queries 1000

Output JSON saved to ``<checkpoint>/../results/two_stage.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.subbit.data import EmbeddingStore
from src.subbit.evaluation import compute_mrr, compute_ndcg, compute_recall
from src.subbit.model import SubBitModel
from src.subbit.utils import get_device, setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=str)
    p.add_argument("--embeddings-dir", default="data/embeddings/msmarco/100k", type=str)
    p.add_argument("--top-k", nargs="+", type=int, default=[100, 500, 1000, 2000],
                   help="Stage-1 candidate cutoffs to evaluate")
    p.add_argument("--max-queries", type=int, default=-1)
    p.add_argument("--device", default="auto", type=str)
    p.add_argument("--output", default=None, type=str)
    p.add_argument("--max-doc-tokens", type=int, default=180)
    p.add_argument("--encode-chunk", type=int, default=2048,
                   help="Doc chunk size during pre-encoding (memory ↔ speed).")
    p.add_argument("--score-doc-chunk", type=int, default=4096,
                   help="Doc chunk size during scoring sweep.")
    p.add_argument("--query-batch", type=int, default=32,
                   help="Queries per scoring batch.")
    p.add_argument("--skip-oracle", action="store_true",
                   help="Skip the fp32 oracle pass (saves ~half the runtime).")
    return p.parse_args()


def load_qrels_tsv(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                qid, _, did, rel = parts[:4]
                qrels.setdefault(qid, {})[did] = int(rel)
            elif len(parts) >= 3:
                qid, did, rel = parts[:3]
                qrels.setdefault(qid, {})[did] = int(rel)
    return qrels


def encode_corpus_batched(
    model: SubBitModel,
    doc_store: EmbeddingStore,
    doc_ids: list[str],
    device: torch.device,
    max_tokens: int,
    encode_chunk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pre-encode the entire corpus into binary {-1,+1} (fp16) and fp32 R·d.

    All docs are first padded to a single ``T_max``, then projected through
    ``model.R`` in chunks (matmul, no Python loop). Sign() is taken on GPU.

    Returns
    -------
    doc_bin   : (N, T_max, r) fp16   — sign(R·d), zero-padded (stage-1 retrieval)
    doc_raw   : (N, T_max, input_dim) fp32 — the ORIGINAL 128-dim embeddings,
                zero-padded (full-precision stage-2 rerank + oracle; NOT R·d)
    doc_mask  : (N, T_max) bool       — True at real-token positions
    """
    N = len(doc_ids)
    r = model.projected_dim

    print(f"  collecting raw fp32 corpus ({N:,} docs, max_tokens={max_tokens}) ...")
    t0 = time.time()
    raw_chunks: list[torch.Tensor] = []
    lengths: list[int] = []
    for did in doc_ids:
        d = doc_store.get(did)[:max_tokens]
        raw_chunks.append(d)
        lengths.append(int(d.shape[0]))
    T_max = max(lengths)
    raw_padded = torch.zeros((N, T_max, model.input_dim), dtype=torch.float32)
    for i, d in enumerate(raw_chunks):
        raw_padded[i, :d.shape[0]] = d
    print(f"  raw padded: {tuple(raw_padded.shape)}  ~{raw_padded.numel()*4/1e9:.2f} GB"
          f"  in {time.time()-t0:.1f}s")

    # Project through R in chunks. R is (r, input_dim).
    print(f"  applying R in {encode_chunk}-doc chunks ...")
    t0 = time.time()
    R = model.R.weight.detach().to(device)  # (r, input_dim)
    doc_fp = torch.zeros((N, T_max, r), dtype=torch.float32)
    for i in range(0, N, encode_chunk):
        j = min(i + encode_chunk, N)
        with torch.no_grad():
            chunk = raw_padded[i:j].to(device)            # (C, T, d_in)
            proj = chunk @ R.T                             # (C, T, r)
        doc_fp[i:j] = proj.cpu()
    print(f"  R applied in {time.time()-t0:.1f}s")

    # Build mask + binary
    doc_mask = torch.zeros((N, T_max), dtype=torch.bool)
    for i, n in enumerate(lengths):
        doc_mask[i, :n] = True

    doc_bin = torch.sign(doc_fp)
    doc_bin[doc_bin == 0] = 1.0
    # Zero out pad positions so they don't contribute to MaxSim sums.
    doc_bin = doc_bin * doc_mask.unsqueeze(-1).to(doc_bin.dtype)
    # doc_raw = ORIGINAL 128-dim fp32 embeddings (the paper's full-precision
    # rerank/oracle target). Mask in place to avoid a second ~9 GB copy.
    raw_padded *= doc_mask.unsqueeze(-1).to(raw_padded.dtype)
    del doc_fp
    return doc_bin.to(torch.float16), raw_padded, doc_mask


def score_batch_vs_corpus(
    q_batch: torch.Tensor,        # (Q, m, r) fp32
    doc_enc: torch.Tensor,        # (N, T, r) fp16 or fp32 — already on device
    doc_mask: torch.Tensor,       # (N, T) bool — already on device
    score_doc_chunk: int,
) -> torch.Tensor:
    """Compute MaxSim score for a batch of queries against the full corpus.

    Returns (Q, N) tensor of scores in fp32.
    """
    Q, m, r = q_batch.shape
    N = doc_enc.shape[0]
    out = torch.empty(Q, N, dtype=torch.float32, device=q_batch.device)
    q_typed = q_batch.to(doc_enc.dtype)
    for da in range(0, N, score_doc_chunk):
        db = min(da + score_doc_chunk, N)
        chunk_doc = doc_enc[da:db]                                     # (C, T, r)
        chunk_mask = doc_mask[da:db]                                   # (C, T)
        # einsum: (Q, m, r) × (C, T, r) -> (Q, C, m, T)
        sim = torch.einsum("qmr,cnr->qcmn", q_typed, chunk_doc)
        sim = sim.masked_fill(~chunk_mask[None, :, None, :], float("-inf"))
        # max over doc-tokens (n), sum over query-tokens (m)
        scores = sim.max(dim=-1).values.sum(dim=-1).float()            # (Q, C)
        out[:, da:db] = scores
        del sim, scores
    return out


def score_query_vs_candidates(
    q: torch.Tensor,              # (m, r) fp32
    cand_doc: torch.Tensor,       # (K, T, r)
    cand_mask: torch.Tensor,      # (K, T)
) -> torch.Tensor:
    """Score ONE query against K candidate docs."""
    sim = torch.einsum("mr,knr->kmn", q.to(cand_doc.dtype), cand_doc)
    sim = sim.masked_fill(~cand_mask.unsqueeze(1), float("-inf"))
    return sim.max(dim=-1).values.sum(dim=-1).float()                  # (K,)


def main() -> None:
    args = parse_args()
    setup_logging()
    device = get_device(args.device)
    print(f"Device: {device}")

    print(f"Loading model from {args.checkpoint}...")
    model = SubBitModel.load(args.checkpoint, device=device).to(device).eval()
    r = model.projected_dim
    print(f"  R: {tuple(model.R.weight.shape)} | use_scale={model.use_scale} | "
          f"r={r}")

    emb_dir = Path(args.embeddings_dir)
    print(f"Loading data from {emb_dir}...")
    doc_store = EmbeddingStore(emb_dir / "doc_embeddings.pt", mode="dict")
    doc_store.load()
    query_store = EmbeddingStore(emb_dir / "query_embeddings.pt", mode="dict")
    query_store.load()
    qrels = load_qrels_tsv(emb_dir / "qrels.tsv")
    print(f"  {len(qrels):,} dev queries with qrels")

    doc_ids = doc_store.get_all_ids()
    qids_all = [q for q in qrels.keys() if q in query_store._data]
    qids = qids_all if args.max_queries < 0 else qids_all[: args.max_queries]
    print(f"  evaluating {len(qids):,} queries × {len(doc_ids):,} docs")

    print("\n[Pre-encode corpus] (batched matmul) ...")
    doc_bin, doc_raw, doc_mask = encode_corpus_batched(
        model, doc_store, doc_ids, device, args.max_doc_tokens, args.encode_chunk
    )
    doc_bin = doc_bin.to(device)
    doc_mask = doc_mask.to(device)
    # doc_raw (original 128-dim) drives BOTH the stage-2 rerank and the oracle.
    doc_raw_dev = doc_raw.to(device)
    print(f"  doc_bin: {tuple(doc_bin.shape)} {doc_bin.dtype} ~{doc_bin.numel()*2/1e9:.2f} GB")
    print(f"  doc_raw: {tuple(doc_raw.shape)} {doc_raw.dtype} ~{doc_raw.numel()*4/1e9:.2f} GB (128-dim)")

    # Pre-encode queries — single canonical query rep used for all paths so
    # stage1/stage2/oracle scoring rules differ ONLY in the doc representation.
    # We use model.encode_query (R·q with the trained scale head) because that's
    # the query side the model was trained on. Both binary stage and fp32 rerank
    # will use this same q; the doc switches between sign(R·d) and R·d.
    print(f"\n[Pre-encode queries] ({len(qids):,} queries) ...")
    t0 = time.time()
    q_list: list[torch.Tensor] = []      # R·q·scale (r=64) — stage-1 binary path
    qraw_list: list[torch.Tensor] = []   # original 128-dim — full-precision stage-2/oracle
    q_lens: list[int] = []
    with torch.no_grad():
        for qid in qids:
            q = query_store.get(qid).to(device)
            q_list.append(model.encode_query(q).float().cpu())
            qraw_list.append(q.float().cpu())
            q_lens.append(int(q.shape[0]))
    Q = len(qids)
    Mq = max(q_lens)
    din = model.input_dim
    q_enc = torch.zeros(Q, Mq, r, dtype=torch.float32)
    q_raw = torch.zeros(Q, Mq, din, dtype=torch.float32)
    q_mask = torch.zeros(Q, Mq, dtype=torch.bool)
    for i, (qb, qr, n) in enumerate(zip(q_list, qraw_list, q_lens)):
        q_enc[i, :n] = qb
        q_raw[i, :n] = qr
        q_mask[i, :n] = True
    print(f"  encoded in {time.time()-t0:.1f}s; q_enc {tuple(q_enc.shape)} (r=64), "
          f"q_raw {tuple(q_raw.shape)} (128-dim)")

    # Score Stage-1 (binary) for ALL queries against the full corpus
    print(f"\n[Stage 1] binary scoring ({Q:,} queries × {len(doc_ids):,} docs, "
          f"chunks {args.score_doc_chunk}, query batch {args.query_batch}) ...")
    t0 = time.time()
    DEPTH = max(1000, max(args.top_k))
    stage1_top_indices = torch.empty(Q, DEPTH, dtype=torch.int64)
    with torch.no_grad():
        for qa in range(0, Q, args.query_batch):
            qb = min(qa + args.query_batch, Q)
            q_batch = q_enc[qa:qb].to(device)                           # (Qb, m, r)
            scores = score_batch_vs_corpus(
                q_batch, doc_bin, doc_mask, args.score_doc_chunk
            )                                                           # (Qb, N)
            top = torch.topk(scores, DEPTH, dim=-1).indices.cpu()       # (Qb, DEPTH)
            stage1_top_indices[qa:qb] = top
    elapsed_s1 = time.time() - t0
    print(f"  stage 1 done in {elapsed_s1/60:.2f} min")

    # Stage-2 rerank — same query rep, fp32 doc (R·d) instead of sign(R·d)
    print(f"\n[Stage 2] FULL 128-dim fp32 rerank on top-{DEPTH} candidates ...")
    t0 = time.time()
    stage2_scores = torch.empty(Q, DEPTH, dtype=torch.float32)
    with torch.no_grad():
        for i in range(Q):
            cand_idx = stage1_top_indices[i].to(device)                 # (DEPTH,)
            cand_doc = doc_raw_dev[cand_idx]                            # (DEPTH, T, 128) orig fp32
            cand_mask = doc_mask[cand_idx]
            q = q_raw[i].to(device)                                     # (m, 128) orig fp32
            stage2_scores[i] = score_query_vs_candidates(q, cand_doc, cand_mask).cpu()
    elapsed_s2 = time.time() - t0
    print(f"  stage 2 done in {elapsed_s2/60:.2f} min")

    # fp32 oracle (full corpus, same query rep × fp32 R·d) — optional
    oracle_top_indices = None
    elapsed_oracle = 0.0
    if not args.skip_oracle:
        print(f"\n[Oracle] full 128-dim fp32 MaxSim over full corpus (= FP128) ...")
        t0 = time.time()
        oracle_top_indices = torch.empty(Q, DEPTH, dtype=torch.int64)
        with torch.no_grad():
            for qa in range(0, Q, args.query_batch):
                qb = min(qa + args.query_batch, Q)
                q_batch = q_raw[qa:qb].to(device)              # original 128-dim
                scores = score_batch_vs_corpus(
                    q_batch, doc_raw_dev, doc_mask, args.score_doc_chunk
                )
                top = torch.topk(scores, DEPTH, dim=-1).indices.cpu()
                oracle_top_indices[qa:qb] = top
        elapsed_oracle = time.time() - t0
        print(f"  oracle done in {elapsed_oracle/60:.2f} min")

    # Build rankings (each is dict[qid] -> list[doc_id])
    print("\n[Build rankings + metrics]")
    rankings_binary: dict[str, list[str]] = {}
    rankings_oracle: dict[str, list[str]] = {}
    rankings_twostage: dict[int, dict[str, list[str]]] = {k: {} for k in args.top_k}
    for i, qid in enumerate(qids):
        binary_ranking = [doc_ids[j] for j in stage1_top_indices[i].tolist()]
        rankings_binary[qid] = binary_ranking
        if oracle_top_indices is not None:
            rankings_oracle[qid] = [doc_ids[j] for j in oracle_top_indices[i].tolist()]
        # Two-stage at each K cutoff: keep first K of binary, sort by stage2 score, then
        # append the remainder of binary unchanged so depth-DEPTH metrics are defined.
        for K in args.top_k:
            head_scores = stage2_scores[i, :K]
            head_order = torch.argsort(head_scores, descending=True).tolist()
            head_ids = [binary_ranking[j] for j in head_order]
            tail_ids = binary_ranking[K:]
            rankings_twostage[K][qid] = head_ids + tail_ids

    qrels_subset = {qid: qrels.get(qid, {}) for qid in qids}

    def metrics_for(rk: dict[str, list[str]], label: str) -> dict:
        m = {
            "mrr@10": float(compute_mrr(rk, qrels_subset, k=10)),
            "ndcg@10": float(compute_ndcg(rk, qrels_subset, k=10)),
            "recall@100": float(compute_recall(rk, qrels_subset, k=100)),
            "recall@1000": float(compute_recall(rk, qrels_subset, k=1000)),
        }
        print(f"  {label:<32s} mrr@10={m['mrr@10']:.4f}  ndcg@10={m['ndcg@10']:.4f}  "
              f"r@100={m['recall@100']:.4f}  r@1000={m['recall@1000']:.4f}")
        return m

    results = {
        "stage1_binary": metrics_for(rankings_binary, "Stage-1 binary only"),
    }
    if oracle_top_indices is not None:
        results["fp32_oracle"] = metrics_for(rankings_oracle, "fp32 oracle")
    results["two_stage"] = {
        str(K): metrics_for(rankings_twostage[K], f"Two-stage K={K} fp32 rerank")
        for K in args.top_k
    }

    bytes_per_token = {
        "binary_only": r / 8,
        # full-precision rerank holds a 128-dim fp32 store for stage-2 candidates
        "binary_plus_fp32_rerank": r / 8 + model.input_dim * 4,  # 8 B sign + 512 B fp32
        "fp32_only": model.input_dim * 4,  # full 128-dim fp32 = 512 B/token
    }

    out = {
        "checkpoint": str(args.checkpoint),
        "config": {
            "r": r,
            "use_scale": bool(model.use_scale),
            "n_queries": len(qids),
            "n_docs": len(doc_ids),
            "max_doc_tokens": args.max_doc_tokens,
            "top_k_cutoffs": args.top_k,
            "embeddings_dir": str(emb_dir),
            "score_doc_chunk": args.score_doc_chunk,
            "query_batch": args.query_batch,
        },
        "metrics": results,
        "bytes_per_token": {k: float(v) for k, v in bytes_per_token.items()},
        "wall_clock": {
            "stage1_seconds": elapsed_s1,
            "stage2_seconds": elapsed_s2,
            "oracle_seconds": elapsed_oracle,
        },
    }

    if args.output is None:
        ckpt_dir = Path(args.checkpoint).resolve().parent.parent
        out_path = ckpt_dir / "results" / "two_stage.json"
    else:
        out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
