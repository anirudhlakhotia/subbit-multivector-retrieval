"""Modal app: RANDOM orthogonal R, r=128 + scale on full 8.8M MS MARCO dev.

The r=128 RANDOM arm of the scale-held-constant 8.8M comparison. R is a frozen random
orthonormal 128x128 matrix, W_scale trained with R frozen (checkpoint
ablation_rand128_plus_scale), doc encoded as sign(R.d), query full-precision (asymmetric).
Run identically to ``full_scale/eval_rand_plus_scale_full.py`` (same image, pipeline, metrics)
so the 8.8M r=128 block isolates the projection with the scale head held constant:
trained vs random vs identity, all +scale (16 B/tok). Checkpoint: ablation_rand128_plus_scale.

Reads
-----
``/cache/embeddings.bin``  (produced by ``modal_encode_msmarco.py::encode``)
``/cache/offsets.pt``       per-pid (start, length) into the mmap
``/cache/meta.json``        sanity metadata
``/ckpt/best.pt``           trained R r=128 + scale checkpoint (mounted from local repo)

Writes
------
``/cache/results/rand128_plus_scale_full_msmarco.json``::

    {
      "method": "rand128_plus_scale",
      "n_queries": 6980,
      "n_docs": 8841823,
      "r": 128,
      "use_scale": true,
      "metrics": {"mrr@10": ..., "recall@100": ..., "recall@1000": ...},
      ...
    }

Run (ALWAYS --detach; do NOT pass --rand-seed)
---
::

    modal run --detach full_scale/eval_rand128_plus_scale_full.py::run

Cost estimate (Modal A100-80GB at $2.50/hr, May 2026)
-----------------------------------------------------
* Pre-project corpus (R + sign):          ~3 min
* Query encoding (ColBERTv2 on 6,980 q):  ~30 s
* MaxSim sweep (6,980 q × 8.8M d):        ~30-60 min
* Total wall-clock:                       ~30-60 min  ~$1.5-2.5
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import modal

APP_NAME = "subbit-msmarco-eval-rand128-plus-scale"
# Same v2 volume as canonical SubBit eval — read-only here, so concurrent
# parallel runs are safe. Each parallel script writes to a distinct output
# JSON and uses a distinct sweep-checkpoint path so they don't clobber each
# other if one crashes mid-run.
VOLUME_NAME = "subbit-msmarco-cache-v2"

# Frozen random orthonormal R r=128 + trained scale checkpoint
# (ablation_rand128_plus_scale): R frozen random, only W_scale trained, use_scale=true.
# The r=128 random arm of the scale-held-constant comparison. Do NOT pass --rand-seed
# (R must stay the checkpoint's frozen random R, not a regenerated one).
RELEASE_ROOT = Path(__file__).resolve().parents[1]
LOCAL_CKPT = str(RELEASE_ROOT / "artifacts/checkpoints/ablation_rand128_plus_scale/best.pt")

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True, version=2)

# Reuse the exact same image layers as the encode script so Modal's image
# cache is hit. Only difference: ``add_local_file`` for the trained R.
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
    .add_local_file(LOCAL_CKPT, "/ckpt/best.pt", copy=True)
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
    doc_chunk: int = 4096,
    query_batch: int = 64,     # r=128 packs 76.5 GB; 4096/64 is the proven-safe r=128 config
    top_k: int = 1000,
    log_every_chunks: int = 20,
    checkpoint_every_chunks: int = 100,  # persist top-K to disk every N chunks
    rand_seed: Optional[int] = None,  # if set, override ckpt R with build_random_R(seed)
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
    ckpt = torch.load("/ckpt/best.pt", map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    R_ckpt = sd["R.weight"].to(DEVICE).float()  # (r, 128) from ckpt
    use_scale = "W_scale.weight" in sd
    if use_scale:
        W_scale_w = sd["W_scale.weight"].to(DEVICE).float().squeeze(0)  # (128,)
        W_scale_b = sd["W_scale.bias"].to(DEVICE).float()                # (1,)
    r = R_ckpt.shape[0]
    cfg = ckpt.get("config", {})

    # ------------------------------------------------------------------
    # 2a. Optional: override R with a freshly generated random orthogonal
    #     drawn from a specific seed. Lets us measure the across-seed
    #     sigma-band at 8.8M.
    #     CLI: modal run ... --rand-seed 42
    #     Or:  RAND_SEED=42 modal run ...
    # ------------------------------------------------------------------
    if rand_seed is None:
        env_seed = os.environ.get("RAND_SEED")
        if env_seed is not None and env_seed.strip() != "":
            rand_seed = int(env_seed)
    if rand_seed is not None:
        # Build the random orthogonal R deterministically from the seed (QR of
        # a seeded Gaussian), so every random-R arm uses the same projection.
        torch.manual_seed(int(rand_seed))
        Q_mat, _ = torch.linalg.qr(torch.randn(R_ckpt.shape[1], R_ckpt.shape[1]))
        R = Q_mat.T[:r].contiguous().to(DEVICE).float()
        r_source = f"random_orthogonal_seed_{int(rand_seed)}"
        print(f"R override: seed={rand_seed}, ||R||_F={R.norm().item():.4f}")
    else:
        R = R_ckpt
        r_source = "checkpoint"

    print(f"Checkpoint: r={r}, input_dim={R.shape[1]}, use_scale={use_scale}, R_source={r_source}")
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
        chunk = torch.from_numpy(chunk_np).to(DEVICE).float()
        # fp32 (NOT autocast fp16) so sign(R·d) is bit-reproducible across containers.
        # With a random R, ~1% of projected coords sit within fp16 epsilon of zero;
        # fp16 GEMM nondeterminism flips their signs differently on a resumed container,
        # so the restored first-half top-K is scored against different doc signs than the
        # freshly-recomputed second half -> corrupted resume after a Modal preemption.
        # fp32 (~7 digits vs ~3) makes the projection deterministic; resume is then safe.
        proj = chunk @ R.T  # (chunk, r) fp32
        bin_chunk = torch.sign(proj)
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
    sweep_ckpt_path = cache / "results" / "sweep_checkpoint_rand128_plus_scale.pt"
    sweep_ckpt_path.parent.mkdir(parents=True, exist_ok=True)

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
    starts_t = torch.from_numpy(starts).to(DEVICE)
    lengths_t = torch.from_numpy(lengths).to(DEVICE)

    for ci, da in enumerate(range(0, Dn, doc_chunk)):
        if ci < start_ci:
            continue
        db = min(da + doc_chunk, Dn)
        seg_lengths = lengths_t[da:db]
        seg_max = int(seg_lengths.max().item())
        seg_n = db - da

        # Build padded chunk on GPU (cast int8 → fp16 for matmul throughput)
        D_pad = torch.zeros(seg_n, seg_max, r, dtype=torch.float16, device=DEVICE)
        for i in range(seg_n):
            s, ln = int(starts[da + i]), int(lengths[da + i])
            D_pad[i, :ln] = packed[s:s+ln].to(torch.float16)
        # Mask invalid positions (will be set to -inf in score)
        col = torch.arange(seg_max, device=DEVICE)
        valid = col[None, :] < seg_lengths[:, None]  # (seg_n, seg_max)

        for qa in range(0, Qn, query_batch):
            qb = min(qa + query_batch, Qn)
            Q_b = Q_proj[qa:qb].to(torch.float16)  # (Qb, m, r)
            # einsum: (Qb, m, r) × (seg_n, seg_max, r) → (Qb, m, seg_n, seg_max)
            sim = torch.einsum("qmr,dnr->qmdn", Q_b, D_pad)
            # In-place masked_fill — out-of-place OOMs at long-doc chunks because
            # `sim` is already 7-15 GB depending on query_batch.
            sim.masked_fill_(~valid[None, None, :, :], float("-inf"))
            # max over seg_max → per (q, m, d), sum over m → (Qb, seg_n)
            scores = sim.max(dim=-1).values.sum(dim=1).float()
            del sim

            # Update per-query top-k
            doc_indices = torch.arange(da, db, device=DEVICE).expand(qb - qa, -1)
            combined_scores = torch.cat([top_scores[qa:qb], scores], dim=1)
            combined_indices = torch.cat([top_indices[qa:qb], doc_indices], dim=1)
            top_scores[qa:qb], top_pos = combined_scores.topk(top_k, dim=1)
            top_indices[qa:qb] = combined_indices.gather(1, top_pos)

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
        "method": "rand128_plus_scale",
        "checkpoint_path": "/ckpt/best.pt",
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
    # Seed-suffix the output JSON so multi-seed runs don't clobber.
    # When R came from the checkpoint, keep the canonical name.
    _seed_tag = f"_seed{int(rand_seed)}" if rand_seed is not None else ""
    out_path = cache / "results" / f"rand128_plus_scale_full_msmarco{_seed_tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Add seed + R-source provenance to the JSON for downstream aggregation.
    results.setdefault("provenance", {})
    results["provenance"]["rand_seed"] = (None if rand_seed is None else int(rand_seed))
    results["provenance"]["R_source"] = r_source
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")

    # Per-query sidecar for paired-bootstrap CIs against the trained-R
    # companion run (matches the shared full-scale metric schema).
    sidecar = {
        "method": "rand128_plus_scale",
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
            "rotation": "random_orthogonal",
            "use_scale": use_scale,
            "doc_chunk": doc_chunk,
            "query_batch": query_batch,
            "top_k": top_k,
        },
    }
    sidecar_path = cache / "results" / f"rand128_plus_scale_full_msmarco{_seed_tag}.per_query.pt"
    torch.save(sidecar, sidecar_path)
    print(f"Wrote {sidecar_path} "
          f"({sidecar_path.stat().st_size / 1e6:.1f} MB)")

    vol.commit()


@app.local_entrypoint()
def main():
    run.remote()
