"""Modal app: evaluate a trained SubBit checkpoint on full 8.8M MS MARCO dev.

This is the **generalisation sanity check** for the deployed SubBit model:
the projection $R$ was trained on a 50k MS MARCO triples slice; this script
applies it cold to the full $8{,}841{,}823$-passage corpus and measures
ranking quality on the standard 6,980 dev queries.

Reads
-----
``/cache/embeddings.bin``  (produced by ``modal_encode_msmarco.py::encode``)
``/cache/offsets.pt``       per-pid (start, length) into the mmap
``/cache/meta.json``        sanity metadata
``/ckpt/best.pt``           trained SubBit checkpoint (mounted from local repo)

Writes
------
``/cache/results/subbit_full_msmarco.json``::

    {
      "method": "subbit",
      "n_queries": 6980,
      "n_docs": 8841823,
      "r": 64,
      "use_scale": true,
      "metrics": {"mrr@10": ..., "recall@100": ..., "recall@1000": ...},
      "wall_clock_sec": ...,
      ...
    }

Run
---
::

    modal run full_scale/eval_subbit_full.py::run

Cost estimate (Modal A100-80GB at $2.50/hr, May 2026)
-----------------------------------------------------
* Pre-project corpus (R + sign):          ~3 min
* Query encoding (ColBERTv2 on 6,980 q):  ~30 s
* MaxSim sweep (6,980 q × 8.8M d):        ~30-60 min
* Total wall-clock:                       ~30-60 min  ~$1.5-2.5
"""
from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "subbit-msmarco-eval-subbit"
# v2 cache (post-migration). The original v1 volume "subbit-msmarco-cache" hit
# server-side ENOSPC during commit at ~296 GB; see modal_encode_msmarco.py
# for the migration path. The v2 volume MUST be opened with version=2 so the
# client uses the v2 read path (different gRPC method).
VOLUME_NAME = "subbit-msmarco-cache-v2"

# THE PAPER'S canonical trained R: the 50k-triple checkpoint
# Mounted into the image at /ckpt/best.pt, exactly as the paper run did.
# DO NOT point this at /cache/checkpoints/subbit_500k -- that is a different, 2M-triple model
# (0.3745 @8.8M) and is NOT the trained R the paper reports.
RELEASE_ROOT = Path(__file__).resolve().parents[1]
LOCAL_CKPT = str(RELEASE_ROOT / "artifacts/checkpoints/50k_topk/best.pt")
VOLUME_CKPT = "/ckpt/best.pt"  # in-container path of the mounted LOCAL_CKPT

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True, version=2)

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
    .add_local_file(LOCAL_CKPT, "/ckpt/best.pt", copy=True)
    .run_commands(
        "python -c \"from huggingface_hub import snapshot_download;"
        " snapshot_download('colbert-ir/colbertv2.0')\""
    )
)

DIM = 128
MSMARCO_DEV_SMALL = "msmarco-passage/dev/small"  # 6,980-query standard subset


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/cache": vol},
    timeout=14400,        # 4 hr ceiling, expected ~30-60 min
    memory=64 * 1024,     # 64 GB RAM (offsets dict is ~600 MB; safe margin)
    cpu=8.0,
)
def run(
    doc_chunk: int = 8192,
    query_batch: int = 128,    # 256 OOMs at long-doc chunks; 128 leaves
                               # headroom (worst-case `sim` is Qb*m*seg_n*
                               # seg_max fp16 ~= 15 GB at seg_max=180)
    top_k: int = 1000,
    log_every_chunks: int = 20,
    checkpoint_every_chunks: int = 100,  # persist top-K to disk every N chunks
):
    """Evaluate trained SubBit on full 8.8M MS MARCO dev. Writes JSON to /cache/results/."""
    import os, json, time
    from pathlib import Path

    import numpy as np
    import torch

    os.environ["IR_DATASETS_HOME"] = "/cache/ir_datasets"
    DEVICE = "cuda"

    # ------------------------------------------------------------------
    # 1. Load encoded corpus (mmap) and offsets
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 2. Load trained SubBit weights (no model class needed; just tensors)
    # ------------------------------------------------------------------
    ckpt = torch.load(VOLUME_CKPT, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    R = sd["R.weight"].to(DEVICE).float()  # (r, 128)
    use_scale = "W_scale.weight" in sd
    if use_scale:
        W_scale_w = sd["W_scale.weight"].to(DEVICE).float().squeeze(0)  # (128,)
        W_scale_b = sd["W_scale.bias"].to(DEVICE).float()                # (1,)
    r = R.shape[0]
    cfg = ckpt.get("config", {})
    print(f"Checkpoint: r={r}, input_dim={R.shape[1]}, use_scale={use_scale}")
    print(f"  config: {cfg}")
    print(f"  ||R||_F = {R.norm().item():.4f}")

    # ------------------------------------------------------------------
    # 3. Pre-project corpus: apply R + sign, store packed int8 on GPU
    #    Memory: 595M tokens × 64 dim × 1 B = 38 GB on A100-80
    # ------------------------------------------------------------------
    print(f"\nPre-projecting corpus through R + sign...")
    t0 = time.time()
    packed = torch.empty(n_total_tokens, r, dtype=torch.int8, device=DEVICE)
    PROJECT_CHUNK = 1_000_000
    for start in range(0, n_total_tokens, PROJECT_CHUNK):
        end = min(start + PROJECT_CHUNK, n_total_tokens)
        chunk_np = mmap_arr[start:end].copy()
        chunk = torch.from_numpy(chunk_np).to(DEVICE)
        with torch.amp.autocast(DEVICE, dtype=torch.float16):
            proj = chunk @ R.T  # (chunk, r)
        bin_chunk = torch.sign(proj.float())
        bin_chunk[bin_chunk == 0] = 1
        packed[start:end] = bin_chunk.to(torch.int8)
        del chunk, proj, bin_chunk
    torch.cuda.synchronize()
    print(f"  pre-project done in {time.time()-t0:.1f}s; "
          f"GPU mem {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # ------------------------------------------------------------------
    # 4. Load MS MARCO dev queries + qrels
    # ------------------------------------------------------------------
    print("\nLoading MS MARCO dev queries...")
    import ir_datasets
    ds = ir_datasets.load(MSMARCO_DEV_SMALL)
    queries: list[tuple[str, str]] = [(q.query_id, q.text) for q in ds.queries_iter()]
    qrels: dict[str, set[str]] = {}
    for qrel in ds.qrels_iter():
        qrels.setdefault(qrel.query_id, set()).add(qrel.doc_id)
    print(f"  {len(queries):,} queries, {len(qrels):,} have qrels")

    # ------------------------------------------------------------------
    # 5. Encode queries with ColBERTv2 → apply R + scale head
    # ------------------------------------------------------------------
    print("\nLoading ColBERTv2 encoder...")
    from colbert.modeling.checkpoint import Checkpoint
    from colbert.infra import ColBERTConfig
    from huggingface_hub import snapshot_download
    ckpt_dir = snapshot_download("colbert-ir/colbertv2.0")
    encoder = Checkpoint(ckpt_dir, colbert_config=ColBERTConfig(query_maxlen=32))
    encoder = encoder.eval().to(DEVICE)

    qids = [q[0] for q in queries]
    texts = [q[1] for q in queries]

    print("Encoding queries...")
    t0 = time.time()
    Q_chunks = []
    for i in range(0, len(texts), 512):
        with torch.inference_mode():
            with torch.amp.autocast(DEVICE, dtype=torch.float16):
                Q_c = encoder.queryFromText(texts[i:i+512], bsize=64)
        Q_chunks.append(Q_c.float())
    Q_emb = torch.cat(Q_chunks, dim=0)  # (Qn, 32, 128)
    print(f"  query encoding done in {time.time()-t0:.1f}s, shape {tuple(Q_emb.shape)}")

    # Free the heavy encoder; we only need R and W_scale for queries.
    del encoder
    torch.cuda.empty_cache()

    # Apply R + scale head:
    #   q_proj_i = scale(q_i) * (R @ q_i),  scale = 1 + 0.5 * sigmoid(W·q + b)
    Q_proj = Q_emb @ R.T  # (Qn, m, r)
    if use_scale:
        scale_logit = (Q_emb * W_scale_w).sum(dim=-1) + W_scale_b  # (Qn, m)
        scale = 1.0 + 0.5 * torch.sigmoid(scale_logit)
        Q_proj = Q_proj * scale.unsqueeze(-1)
    print(f"  queries projected: {tuple(Q_proj.shape)}")
    Qn, m_q, _ = Q_proj.shape

    # ------------------------------------------------------------------
    # 6. Chunked asymmetric MaxSim with sort-by-length (smaller padded chunks)
    # ------------------------------------------------------------------
    pids = list(offsets.keys())
    starts = np.array([offsets[p][0] for p in pids], dtype=np.int64)
    lengths = np.array([offsets[p][1] for p in pids], dtype=np.int32)
    sort_order = np.argsort(lengths)  # short docs first → narrower padded chunks
    pids = [pids[i] for i in sort_order]
    starts = starts[sort_order]
    lengths = lengths[sort_order]

    Dn = len(pids)
    n_chunks = (Dn + doc_chunk - 1) // doc_chunk
    sweep_ckpt_path = cache / "results" / "sweep_checkpoint.pt"
    sweep_ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Precompute flat-layout helpers used by the vectorised inner loop.
    #
    # The original implementation built a padded (seg_n, seg_max, r) tensor
    # via a Python for-loop, then materialised a (Qb, m, seg_n, seg_max)
    # similarity tensor (6-15 GB) just to take a max over the last axis.
    #
    # The optimised path instead works in *flat* (token) layout:
    #   D_flat : (T_chunk, r) where T_chunk = sum(seg_lengths)   -- contiguous
    #   sim    : (Qb*m, T_chunk)                                -- single GEMM
    #   reduce : scatter_reduce_(amax) into (Qb*m, seg_n)
    # then sum over the m axis. This eliminates the (Qb, m, seg_n, seg_max)
    # intermediate and the masked_fill, and replaces the einsum with a single
    # cuBLAS matmul. The reduction is done in-place via scatter_reduce_, so
    # peak memory is O(Qb*m*T_chunk) which is `seg_max/avg_len` smaller than
    # the padded version (typically ~2-4x for sort-by-length chunks).
    #
    # The flat layout is built per-chunk inside the loop: a 1-D `gather_idx`
    # of shape (T_chunk,) maps each flat-layout token slot to its absolute
    # row in `packed`. T_chunk * 8 B per chunk is a few MB.
    # ------------------------------------------------------------------
    starts_t = torch.from_numpy(starts).to(DEVICE)
    lengths_t = torch.from_numpy(lengths.astype(np.int64)).to(DEVICE)

    # Resume mid-sweep if a checkpoint exists (e.g. previous run OOM'd or was
    # preempted). The checkpoint stores top-K state + the next chunk index to
    # process. pids/starts/lengths are deterministic (sorted) across runs, so
    # the same chunk index points at the same docs.
    if sweep_ckpt_path.exists():
        snap = torch.load(sweep_ckpt_path, map_location="cpu", weights_only=False)
        top_scores = snap["top_scores"].to(DEVICE)
        top_indices = snap["top_indices"].to(DEVICE)
        start_ci = int(snap["next_ci"])
        print(f"\nMaxSim sweep RESUME: starting from chunk {start_ci}/{n_chunks}")
    else:
        top_scores = torch.full((Qn, top_k), -float("inf"), device=DEVICE)
        top_indices = torch.full((Qn, top_k), -1, dtype=torch.int64, device=DEVICE)
        start_ci = 0
        print(f"\nMaxSim sweep: {Qn:,} queries × {Dn:,} docs "
              f"({n_chunks:,} doc chunks of {doc_chunk}, "
              f"{Qn // query_batch + 1} query batches of {query_batch})")
    t0 = time.time()
    # Note: `starts_t` and `lengths_t` were already built above the resume block.

    for ci, da in enumerate(range(0, Dn, doc_chunk)):
        if ci < start_ci:
            continue
        db = min(da + doc_chunk, Dn)
        seg_lengths = lengths_t[da:db]                  # (seg_n,) int64
        seg_max = int(seg_lengths.max().item())
        seg_n = db - da

        # ----- Vectorised padded chunk build -----
        # Replace the seg_n-iteration Python loop with a single GPU
        # gather + scatter into a padded (seg_n, seg_max, r) tensor.
        # Compared to the original Python loop (~8192 tiny CUDA launches
        # per chunk, ~8.8M total), this is ~3x faster. We keep the
        # padded layout so the downstream einsum + max-reduce stays the
        # same — that path is bounded by cuBLAS, which is fast on
        # A100. The flat-layout + scatter_reduce_(amax) variant tested
        # earlier was 5x slower because PyTorch's CUDA scatter_reduce
        # falls back to atomic max ops with high contention.
        seg_starts = starts_t[da:db]
        chunk_cum = torch.empty(seg_n + 1, dtype=torch.int64, device=DEVICE)
        chunk_cum[0] = 0
        torch.cumsum(seg_lengths, dim=0, out=chunk_cum[1:])
        T_chunk = int(chunk_cum[-1].item())

        chunk_local_doc = torch.repeat_interleave(
            torch.arange(seg_n, device=DEVICE), seg_lengths
        )                                                # (T_chunk,)
        within = torch.arange(T_chunk, device=DEVICE) - chunk_cum[chunk_local_doc]
        gather_idx = seg_starts[chunk_local_doc] + within

        D_pad = torch.zeros(seg_n, seg_max, r, dtype=torch.float16, device=DEVICE)
        D_pad[chunk_local_doc, within] = packed.index_select(0, gather_idx).to(torch.float16)

        # Mask invalid (padding) positions for the max reduction.
        col = torch.arange(seg_max, device=DEVICE)
        valid = col[None, :] < seg_lengths[:, None]      # (seg_n, seg_max)

        for qa in range(0, Qn, query_batch):
            qb_idx = min(qa + query_batch, Qn)
            Q_b = Q_proj[qa:qb_idx].to(torch.float16)    # (Qb, m, r)

            # einsum: (Qb, m, r) × (seg_n, seg_max, r) → (Qb, m, seg_n, seg_max)
            sim = torch.einsum("qmr,dnr->qmdn", Q_b, D_pad)
            sim.masked_fill_(~valid[None, None, :, :], float("-inf"))
            # max over seg_max → (Qb, m, seg_n), then sum over m → (Qb, seg_n)
            scores = sim.max(dim=-1).values.sum(dim=1).float()
            del sim

            # Update per-query top-k
            Qb = qb_idx - qa
            doc_indices = torch.arange(da, db, device=DEVICE).expand(Qb, -1)
            combined_scores = torch.cat([top_scores[qa:qb_idx], scores], dim=1)
            combined_indices = torch.cat([top_indices[qa:qb_idx], doc_indices], dim=1)
            top_scores[qa:qb_idx], top_pos = combined_scores.topk(top_k, dim=1)
            top_indices[qa:qb_idx] = combined_indices.gather(1, top_pos)

        del D_pad, valid, gather_idx, chunk_local_doc, within, chunk_cum

        if (ci + 1) % log_every_chunks == 0:
            elapsed = time.time() - t0
            rate = (ci + 1) * doc_chunk / elapsed
            eta = max(0, (Dn - (ci + 1) * doc_chunk) / max(rate, 1.0))
            print(f"  chunk {ci+1:>5}/{n_chunks}  "
                  f"{rate/1e3:6.0f} K docs/s  ETA {eta/60:5.1f}m  "
                  f"GPU mem {torch.cuda.memory_allocated()/1e9:.1f} GB",
                  flush=True)

        # Periodic top-K checkpoint so an OOM / preemption near the end doesn't
        # nuke the whole sweep. Skip the very last chunk (final state is in the
        # JSON anyway).
        if (ci + 1) % checkpoint_every_chunks == 0 and (ci + 1) < n_chunks:
            torch.save({
                "top_scores": top_scores.cpu(),
                "top_indices": top_indices.cpu(),
                "next_ci": ci + 1,
            }, sweep_ckpt_path)
            vol.commit()
            print(f"  checkpointed at chunk {ci+1}/{n_chunks} → {sweep_ckpt_path}",
                  flush=True)

    torch.cuda.synchronize()
    sweep_secs = time.time() - t0
    print(f"\nSweep done in {sweep_secs/60:.1f} min")
    # Sweep finished cleanly; remove the checkpoint so a future fresh run
    # doesn't accidentally resume from it.
    if sweep_ckpt_path.exists():
        sweep_ckpt_path.unlink()

    # ------------------------------------------------------------------
    # 7. Compute MRR@10 / Recall@100 / Recall@1000 against qrels
    # ------------------------------------------------------------------
    print("\nComputing metrics...")
    top_indices_cpu = top_indices.cpu().numpy()
    top_scores_cpu = top_scores.cpu().numpy()

    mrr_at_10 = 0.0
    recall_at_100 = 0.0
    recall_at_1000 = 0.0
    n_eval = 0
    # Per-query arrays (NaN for queries without qrels; aligned with `qids`).
    import numpy as _np
    pq_rr = _np.full(len(qids), _np.nan, dtype=_np.float32)
    pq_r100 = _np.full(len(qids), _np.nan, dtype=_np.float32)
    pq_r1000 = _np.full(len(qids), _np.nan, dtype=_np.float32)
    pq_has_qrels = _np.zeros(len(qids), dtype=bool)
    for q_idx, qid in enumerate(qids):
        relevant = qrels.get(qid)
        if not relevant:
            continue
        pq_has_qrels[q_idx] = True
        retrieved = [pids[idx] for idx in top_indices_cpu[q_idx] if idx >= 0]
        # MRR@10
        rr = 0.0
        for rank, p in enumerate(retrieved[:10], start=1):
            if p in relevant:
                rr = 1.0 / rank
                break
        pq_rr[q_idx] = rr
        mrr_at_10 += rr
        # Standard MS MARCO recall (numerator: #relevant retrieved at K, denominator: |relevant|)
        n_rel = len(relevant)
        r100 = sum(1 for p in retrieved[:100] if p in relevant) / n_rel
        r1000 = sum(1 for p in retrieved[:1000] if p in relevant) / n_rel
        pq_r100[q_idx] = r100
        pq_r1000[q_idx] = r1000
        recall_at_100 += r100
        recall_at_1000 += r1000
        n_eval += 1

    if n_eval == 0:
        raise RuntimeError("no queries had qrels; check ir-datasets")
    metrics = {
        "mrr@10": mrr_at_10 / n_eval,
        "recall@100": recall_at_100 / n_eval,
        "recall@1000": recall_at_1000 / n_eval,
    }
    print(json.dumps(metrics, indent=2))

    # ------------------------------------------------------------------
    # 8. Save results
    # ------------------------------------------------------------------
    results = {
        "method": "subbit",
        "checkpoint_path": VOLUME_CKPT,
        "checkpoint_config": cfg,
        "encoder": "colbert-ir/colbertv2.0",
        "corpus": "msmarco-passage (full)",
        "queries": MSMARCO_DEV_SMALL,
        "n_docs": n_docs,
        "n_total_tokens": n_total_tokens,
        "n_queries_evaluated": n_eval,
        "r": r,
        "use_scale": use_scale,
        "scoring": "asymmetric",
        "doc_chunk": doc_chunk,
        "query_batch": query_batch,
        "top_k": top_k,
        "metrics": metrics,
        "wall_clock_sec": {
            "sweep": sweep_secs,
        },
    }
    out_path = cache / "results" / "subbit_full_msmarco.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")

    # Per-query sidecar for paired-bootstrap CIs against companion runs
    # Matches the shared full-scale metric schema used by companion jobs.
    sidecar = {
        "method": "subbit_trained",
        "qids": qids,
        "has_qrels": pq_has_qrels,
        "rr@10": pq_rr,
        "recall@100": pq_r100,
        "recall@1000": pq_r1000,
        "top1000_indices": top_indices_cpu.astype(_np.int64),
        "top1000_scores": top_scores_cpu.astype(_np.float32),
        "pids_sorted_by_length": pids,
        "config": {
            "r": r,
            "bytes_per_token": r // 8,
            "rotation": "trained",
            "use_scale": use_scale,
            "doc_chunk": doc_chunk,
            "query_batch": query_batch,
            "top_k": top_k,
        },
    }
    sidecar_path = cache / "results" / "subbit_full_msmarco.per_query.pt"
    torch.save(sidecar, sidecar_path)
    print(f"Wrote {sidecar_path} "
          f"({sidecar_path.stat().st_size / 1e6:.1f} MB)")

    vol.commit()


@app.local_entrypoint()
def main():
    run.remote()
