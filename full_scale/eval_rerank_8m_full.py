"""Modal app: SubBit stage-1 candidates + fp32 rerank-K sweep at 8.8M.

This is the full-corpus version of the archived local rerank-K sweep, which ran
the same sweep at 100k. The pipeline is:

  Stage 1 (SubBit r=64 1-bit, already-computed by D1):
    Read ``/cache/results/subbit_full_msmarco.per_query.pt`` and take
    ``top1000_indices`` / ``top1000_scores``.

  Stage 2 (this script):
    For each query, fp32 MaxSim rerank the top-K stage-1 candidates
    for K in {10, 100, 1000}. Combine with stage-1 positions K..1000
    in original order. Compute MRR@10 / R@100 / R@1000 per K.

Why a separate Modal job and not a Python rerun
-----------------------------------------------
The 8.8M fp32 corpus is 304 GB and lives on the Modal volume; rerank
needs the full fp32 doc embeddings, so we have to do it on the same
volume. Stage 1 already produced the 1000 candidates per query.
Stage 2 only needs the union of those candidates (worst case
6,980 * 1000 = ~7M doc-rows; typically much smaller after de-dup).

Reads
-----
``/cache/embeddings.bin``, ``/cache/offsets.pt``, ``/cache/meta.json``
``/cache/results/subbit_full_msmarco.per_query.pt`` (from D1)

Writes
------
``/cache/results/rerank_8m_K_sweep.json``
``/cache/results/rerank_8m_K_sweep.per_query.pt``

Per-query sidecar schema
------------------------
{
  "method": "subbit_stage1_fp32_rerank",
  "qids": qids,
  "has_qrels": (Qn,) bool,
  "K_values": [10, 100, 1000],
  "rr@10_per_K":     {K -> (Qn,) float64},
  "recall@100_per_K":{K -> (Qn,) float64},
  "recall@1000_per_K":{K -> (Qn,) float64},
  "stage1_method": "<from D1 sidecar>",
  "config": {...},
}

Run
---
::

    modal run full_scale/eval_rerank_8m_full.py::run

Cost estimate (Modal A100-80GB at $2.50/hr)
-------------------------------------------
* Query encoding (ColBERTv2 on 6,980 q):     ~30s
* Load union of candidate doc-rows (mmap):   ~10-15 min
* fp32 MaxSim rerank (top-1000 / query):     ~60-80 min
* Total wall-clock: ~75-95 min, ~$3.10-3.95.
"""
from __future__ import annotations

import modal

APP_NAME = "subbit-msmarco-rerank-8m"
VOLUME_NAME = "subbit-msmarco-cache-v2"
STAGE1_SIDECAR = "/cache/results/subbit_full_msmarco.per_query.pt"

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False, version=2)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "wget", "build-essential")
    .pip_install(
        "torch==2.3.1",
        "transformers==4.41.2",
        "huggingface_hub==0.23.4",
        "ir-datasets==0.5.7",
        "tqdm==4.66.4",
        "ujson==5.10.0",
        "numpy<2",
    )
    .pip_install("colbert-ai==0.2.21", "faiss-cpu==1.8.0")
    .run_commands(
        "python -c \"from huggingface_hub import snapshot_download;"
        " snapshot_download('colbert-ir/colbertv2.0')\""
    )
)

DIM = 128
MSMARCO_DEV_SMALL = "msmarco-passage/dev/small"
K_VALUES = [10, 100, 1000]


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/cache": vol},
    timeout=14400,
    memory=64 * 1024,
    cpu=8.0,
)
def run(
    query_batch: int = 64,
    top_k_final: int = 1000,
    log_every_queries: int = 500,
    stage1_sidecar: str = STAGE1_SIDECAR,  # override to rerank a different stage-1 source (e.g. the random arm)
):
    """fp32-rerank-K sweep over SubBit r=64 stage-1 candidates."""
    import json, os, time
    from pathlib import Path

    import numpy as np
    import torch

    os.environ["IR_DATASETS_HOME"] = "/cache/ir_datasets"
    DEVICE = "cuda"

    cache = Path("/cache")
    meta = json.loads((cache / "meta.json").read_text())
    if not meta.get("complete"):
        print("WARNING: meta.json says encoding is not complete; results may be partial")
    offsets = torch.load(cache / "offsets.pt", map_location="cpu", weights_only=False)
    n_docs = len(offsets)
    n_total_tokens = meta["total_tokens"]
    print(f"Corpus: {n_docs:,} docs / {n_total_tokens:,} tokens "
          f"({n_total_tokens * DIM * 4 / 1e9:.1f} GB fp32 mmap)")
    mmap_arr = np.memmap(
        cache / "embeddings.bin",
        dtype=np.float32, mode="r",
        shape=(n_total_tokens, DIM),
    )

    # ---- Stage 1 candidates (from D1) ------------------------------------
    stage1_tag = Path(stage1_sidecar).name.replace(
        "_full_msmarco.per_query.pt", "").replace(".per_query.pt", "")
    print(f"Stage-1 source: {stage1_sidecar}  (output tag: {stage1_tag})")
    if not Path(stage1_sidecar).exists():
        raise FileNotFoundError(
            f"Stage 1 sidecar missing: {stage1_sidecar}. "
            "Run D1 (full_scale/eval_subbit_full.py / full_scale/eval_rand_plus_scale_full.py) first."
        )
    d1 = torch.load(stage1_sidecar, map_location="cpu", weights_only=False)
    stage1_method = d1.get("method", "unknown")
    top1000_indices = np.asarray(d1["top1000_indices"], dtype=np.int64)  # (Qn_d1, 1000)
    d1_qids = list(d1["qids"])
    d1_pids = list(d1["pids_sorted_by_length"])
    Qn_d1, top_k_stage1 = top1000_indices.shape
    print(f"Stage 1 sidecar: method={stage1_method}  qids={Qn_d1}  top_k={top_k_stage1}")

    # Build pid -> mmap-offset table from offsets.pt (length-sorted to match D1).
    pids_corpus = list(offsets.keys())
    starts_corpus = np.array([offsets[p][0] for p in pids_corpus], dtype=np.int64)
    lengths_corpus = np.array([offsets[p][1] for p in pids_corpus], dtype=np.int32)
    sort_order = np.argsort(lengths_corpus)
    pids_corpus = [pids_corpus[i] for i in sort_order]
    starts_corpus = starts_corpus[sort_order]
    lengths_corpus = lengths_corpus[sort_order]
    if pids_corpus != d1_pids:
        # If D1's sorted list and ours disagree (e.g. stale offsets.pt
        # vs sidecar), we have to trust the sidecar's row->pid mapping.
        print("WARNING: corpus pid order differs from D1 sidecar; "
              "using D1 sidecar's pid order.")
        pid_to_pos = {p: i for i, p in enumerate(pids_corpus)}
        # Reindex starts/lengths to D1 sidecar's row order.
        starts_d1 = np.array([starts_corpus[pid_to_pos[p]] for p in d1_pids], dtype=np.int64)
        lengths_d1 = np.array([lengths_corpus[pid_to_pos[p]] for p in d1_pids], dtype=np.int32)
    else:
        starts_d1 = starts_corpus
        lengths_d1 = lengths_corpus
    pids_sorted = d1_pids

    # ---- Load qrels + queries (same source as D1) ------------------------
    import ir_datasets
    print("Loading MS MARCO dev queries...")
    ds = ir_datasets.load(MSMARCO_DEV_SMALL)
    qid_to_text = {q.query_id: q.text for q in ds.queries_iter()}
    qrels: dict[str, set[str]] = {}
    for qrel in ds.qrels_iter():
        qrels.setdefault(qrel.query_id, set()).add(qrel.doc_id)

    qids = d1_qids
    texts = [qid_to_text[q] for q in qids if q in qid_to_text]
    if len(texts) != len(qids):
        # Some qids in the sidecar may not be in the queries set
        # (shouldn't happen but be defensive).
        kept = [q for q in qids if q in qid_to_text]
        keep_mask = np.array([q in qid_to_text for q in qids])
        print(f"  warning: {len(qids) - len(kept)} qids in sidecar but "
              f"not in dev queries; dropping.")
        qids = kept
        top1000_indices = top1000_indices[keep_mask]
        texts = [qid_to_text[q] for q in qids]
    print(f"  {len(qids):,} queries, {len(qrels):,} have qrels")

    # ---- Encode queries with ColBERTv2 (fp32) ----------------------------
    print("\nLoading ColBERTv2 encoder...")
    from colbert.modeling.checkpoint import Checkpoint
    from colbert.infra import ColBERTConfig
    from huggingface_hub import snapshot_download
    ckpt_dir = snapshot_download("colbert-ir/colbertv2.0")
    encoder = Checkpoint(ckpt_dir, colbert_config=ColBERTConfig(query_maxlen=32))
    encoder = encoder.eval().to(DEVICE)

    t_q0 = time.time()
    Q_chunks = []
    for i in range(0, len(texts), 512):
        with torch.inference_mode():
            with torch.amp.autocast(DEVICE, dtype=torch.float16):
                Q_c = encoder.queryFromText(texts[i:i+512], bsize=64)
        Q_chunks.append(Q_c.float())
    Q_emb = torch.cat(Q_chunks, dim=0).to(DEVICE)         # (Qn, m_q, 128) fp32
    t_query_encode = time.time() - t_q0
    print(f"  query encoding done in {t_query_encode:.1f}s, shape {tuple(Q_emb.shape)}")
    del encoder
    torch.cuda.empty_cache()

    # ---- Build the set of doc-rows we need to read from the mmap ---------
    # Each query has up to 1000 stage-1 candidates; the union across all
    # queries is bounded by 6,980 * 1000 = ~7M (much smaller after dedup,
    # but we'll allocate based on union and read each only once).
    Qn = len(qids)
    needed_rows = np.unique(top1000_indices.reshape(-1))
    needed_rows = needed_rows[needed_rows >= 0]
    print(f"\nUnion of stage-1 candidates across queries: {len(needed_rows):,} doc rows "
          f"(of {n_docs:,} total)")

    # We can't fit fp32 embeddings for 7M docs in GPU memory at typical
    # avg-token lengths; instead we read them on demand per query. The
    # mmap reads are the slow path; we do them in row-batched form per
    # query.

    # ---- Per-K accumulators ---------------------------------------------
    nan = float("nan")
    per_K_rr   = {K: np.full(Qn, nan, dtype=np.float64) for K in K_VALUES}
    per_K_r100 = {K: np.full(Qn, nan, dtype=np.float64) for K in K_VALUES}
    per_K_r1000 = {K: np.full(Qn, nan, dtype=np.float64) for K in K_VALUES}
    has_qrels = np.zeros(Qn, dtype=bool)
    n_eval_per_K = {K: 0 for K in K_VALUES}
    mrr_per_K   = {K: 0.0 for K in K_VALUES}
    r100_per_K  = {K: 0.0 for K in K_VALUES}
    r1000_per_K = {K: 0.0 for K in K_VALUES}

    rerank_secs_per_K = {K: 0.0 for K in K_VALUES}
    t_rerank_start = time.time()
    K_max = max(K_VALUES)
    print(f"\nReranking: fp32 MaxSim on top-{K_max} per query, "
          f"K sweep {K_VALUES}")

    # Pre-bucket K positions so the head sort is O(K) per K rather than
    # O(K_max) per K.
    for q_idx in range(Qn):
        cand_rows = top1000_indices[q_idx]                                 # (1000,)
        q_emb_fp32 = Q_emb[q_idx]                                           # (m_q, 128)

        # Read fp32 embeddings for the top-K_max candidates from the mmap.
        # We could also read all 1000 (the suffix doesn't need scoring),
        # but reading only the head saves I/O.
        head_rows = cand_rows[:K_max]

        # Compute fp32 MaxSim per candidate. Vectorise: pad to seg_max,
        # einsum on a single (1, m_q, K_max, seg_max) sim tensor.
        seg_lengths = lengths_d1[head_rows]
        seg_max = int(seg_lengths.max())
        D_pad = torch.zeros(K_max, seg_max, DIM, dtype=torch.float16, device=DEVICE)
        for j, row in enumerate(head_rows.tolist()):
            s, ln = int(starts_d1[row]), int(lengths_d1[row])
            D_pad[j, :ln] = torch.from_numpy(
                mmap_arr[s:s+ln].copy()
            ).to(DEVICE, dtype=torch.float16)

        col = torch.arange(seg_max, device=DEVICE)
        valid = col[None, :] < torch.from_numpy(seg_lengths.astype(np.int64)).to(DEVICE)[:, None]

        with torch.inference_mode():
            Q_b = q_emb_fp32.unsqueeze(0).to(torch.float16)          # (1, m_q, 128)
            sim = torch.einsum("qmr,dnr->qmdn", Q_b, D_pad)          # (1, m_q, K_max, seg_max)
            sim.masked_fill_(~valid[None, None, :, :], float("-inf"))
            head_scores = sim.max(dim=-1).values.sum(dim=1).float().squeeze(0)  # (K_max,)
        del D_pad, sim, valid

        head_scores_cpu = head_scores.cpu().numpy()

        relevant = qrels.get(qids[q_idx])
        if not relevant:
            # Still emit reranked rankings, but skip metric updates.
            continue
        has_qrels[q_idx] = True
        n_rel = len(relevant)

        for K in K_VALUES:
            t_K = time.time()
            head_scores_K = head_scores_cpu[:K]
            head_order = np.argsort(-head_scores_K)
            head_rows_K = head_rows[head_order]
            tail_rows_K = cand_rows[K:top_k_stage1]
            full_rows = np.concatenate([head_rows_K, tail_rows_K])
            retrieved = [pids_sorted[r] for r in full_rows if r >= 0]
            rr = 0.0
            for rank, p in enumerate(retrieved[:10], start=1):
                if p in relevant:
                    rr = 1.0 / rank
                    break
            r100 = sum(1 for p in retrieved[:100] if p in relevant) / n_rel
            r1000 = sum(1 for p in retrieved[:1000] if p in relevant) / n_rel
            per_K_rr[K][q_idx] = rr
            per_K_r100[K][q_idx] = r100
            per_K_r1000[K][q_idx] = r1000
            mrr_per_K[K] += rr
            r100_per_K[K] += r100
            r1000_per_K[K] += r1000
            n_eval_per_K[K] += 1
            rerank_secs_per_K[K] += time.time() - t_K

        if (q_idx + 1) % log_every_queries == 0:
            elapsed = time.time() - t_rerank_start
            rate = (q_idx + 1) / elapsed
            eta = (Qn - (q_idx + 1)) / max(rate, 1.0)
            print(f"  query {q_idx+1:>5}/{Qn}  "
                  f"{rate:6.1f} q/s  ETA {eta/60:5.1f}m  "
                  f"GPU mem {torch.cuda.memory_allocated()/1e9:.1f} GB",
                  flush=True)

    rerank_secs = time.time() - t_rerank_start
    print(f"\nRerank done in {rerank_secs/60:.1f} min")

    # ---- Aggregate per-K --------------------------------------------------
    K_sweep_metrics: dict[str, dict[str, float]] = {}
    for K in K_VALUES:
        n_e = n_eval_per_K[K]
        if n_e == 0:
            continue
        K_sweep_metrics[str(K)] = {
            "mrr@10":      mrr_per_K[K]   / n_e,
            "recall@100":  r100_per_K[K]  / n_e,
            "recall@1000": r1000_per_K[K] / n_e,
            "n_eval":      n_e,
            "rerank_secs": rerank_secs_per_K[K],
        }
    for K in K_VALUES:
        if str(K) in K_sweep_metrics:
            m = K_sweep_metrics[str(K)]
            print(f"  K={K:>4}: MRR@10={m['mrr@10']:.4f}  "
                  f"R@100={m['recall@100']:.4f}  "
                  f"R@1000={m['recall@1000']:.4f}  (n={m['n_eval']})")

    # ---- Save aggregate + sidecar ----------------------------------------
    n_eval = int(has_qrels.sum())
    results = {
        "method": "subbit_stage1_fp32_rerank",
        "stage1_source": stage1_sidecar,
        "stage1_method": stage1_method,
        "encoder": "colbert-ir/colbertv2.0",
        "corpus": "msmarco-passage (full)",
        "queries": MSMARCO_DEV_SMALL,
        "n_docs": n_docs,
        "n_total_tokens": n_total_tokens,
        "n_queries_evaluated": n_eval,
        "r_stage1": d1.get("config", {}).get("r"),
        "top_k_stage1": top_k_stage1,
        "K_sweep": K_sweep_metrics,
        "K_values": K_VALUES,
        "wall_clock_sec": {
            "query_encode": t_query_encode,
            "rerank_total": rerank_secs,
            "rerank_per_K": rerank_secs_per_K,
        },
    }
    out_path = cache / "results" / f"rerank_8m_{stage1_tag}_K_sweep.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")

    sidecar = {
        "method": "subbit_stage1_fp32_rerank",
        "qids": qids,
        "has_qrels": has_qrels,
        "K_values": K_VALUES,
        "rr@10_per_K":      {int(K): per_K_rr[K]   for K in K_VALUES},
        "recall@100_per_K": {int(K): per_K_r100[K] for K in K_VALUES},
        "recall@1000_per_K":{int(K): per_K_r1000[K] for K in K_VALUES},
        "stage1_method": stage1_method,
        "config": {
            "stage1_top_k": top_k_stage1,
            "K_values": K_VALUES,
            "query_batch": query_batch,
            "top_k_final": top_k_final,
            "r_stage1": d1.get("config", {}).get("r"),
        },
    }
    sidecar_path = cache / "results" / f"rerank_8m_{stage1_tag}_K_sweep.per_query.pt"
    torch.save(sidecar, sidecar_path)
    print(f"Wrote {sidecar_path} "
          f"({sidecar_path.stat().st_size / 1e6:.1f} MB)")

    vol.commit()


@app.local_entrypoint()
def main():
    run.remote()
