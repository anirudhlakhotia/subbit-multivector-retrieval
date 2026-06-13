"""Modal app: evaluate the ITQ baseline at r=64 on full 8.8M MS MARCO dev.

Matched-format comparison to SubBit at the same 8 B/tok payload. The only
difference vs ``full_scale/eval_subbit_full.py`` is the source of the projection
$R$ and the absence of a query-side scale head:

* SubBit:   $R$ is the trained projection from ``artifacts/checkpoints/50k_topk/best.pt`` plus a 129-parameter scale head ``s(q)``.
* ITQ:      $R$ is the precomputed PCA $\\to$ ITQ rotation from ``artifacts/checkpoints/itq_init/R_r64.pt``; queries are $R \\cdot q$ with no scale head.

Same encoder (ColBERTv2), same corpus (full MS MARCO), same scoring
(asymmetric MaxSim), same metrics path.

Reads
-----
``/cache/embeddings.bin``    full 8.8M corpus mmap (from encode pipeline)
``/cache/offsets.pt``        per-pid (start, length) into the mmap
``/cache/meta.json``         sanity metadata
``/itq/R_r64.pt``            precomputed ITQ projection, shape (64, 128) fp32

Writes
------
``/cache/results/itq_full_msmarco.json``

Run
---
::

    modal run full_scale/eval_itq_full.py::run

Cost ~$1.5-2.5 (~50-70 min on A100-80GB).
"""
from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "subbit-msmarco-eval-itq"
VOLUME_NAME = "subbit-msmarco-cache-v2"

RELEASE_ROOT = Path(__file__).resolve().parents[1]
LOCAL_R = str(RELEASE_ROOT / "artifacts/checkpoints/itq_init/R_r64.pt")

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
    .add_local_file(LOCAL_R, "/itq/R_r64.pt", copy=True)
)

DIM = 128
MSMARCO_DEV_SMALL = "msmarco-passage/dev/small"


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/cache": vol},
    timeout=14400,
    memory=64 * 1024,
    cpu=8.0,
)
def run(
    doc_chunk: int = 8192,
    query_batch: int = 128,
    top_k: int = 1000,
    log_every_chunks: int = 20,
    checkpoint_every_chunks: int = 100,
):
    """Evaluate ITQ at r=64 on full 8.8M MS MARCO dev. Writes JSON to /cache/results/."""
    import os, json, time
    from pathlib import Path

    import numpy as np
    import torch

    os.environ["IR_DATASETS_HOME"] = "/cache/ir_datasets"
    DEVICE = "cuda"

    # ---- Load corpus mmap ---------------------------------------------
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

    # ---- Load ITQ projection R ----------------------------------------
    R = torch.load("/itq/R_r64.pt", map_location=DEVICE, weights_only=False).float()
    if R.shape != (64, DIM):
        raise SystemExit(f"expected R shape (64, {DIM}); got {tuple(R.shape)}")
    r = R.shape[0]
    print(f"ITQ projection R: shape={tuple(R.shape)}, ||R||_F={R.norm().item():.4f}")

    # ---- Pre-project corpus through R + sign --------------------------
    print(f"\nPre-projecting corpus through ITQ R + sign...")
    t0 = time.time()
    packed = torch.empty(n_total_tokens, r, dtype=torch.int8, device=DEVICE)
    PROJECT_CHUNK = 1_000_000
    for start in range(0, n_total_tokens, PROJECT_CHUNK):
        end = min(start + PROJECT_CHUNK, n_total_tokens)
        chunk = torch.from_numpy(mmap_arr[start:end].copy()).to(DEVICE)
        with torch.amp.autocast(DEVICE, dtype=torch.float16):
            proj = chunk @ R.T
        bin_chunk = torch.sign(proj.float())
        bin_chunk[bin_chunk == 0] = 1
        packed[start:end] = bin_chunk.to(torch.int8)
        del chunk, proj, bin_chunk
    torch.cuda.synchronize()
    print(f"  pre-project done in {time.time()-t0:.1f}s; "
          f"GPU mem {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # ---- Load + encode dev queries ------------------------------------
    print("\nLoading MS MARCO dev queries...")
    import ir_datasets
    ds = ir_datasets.load(MSMARCO_DEV_SMALL)
    queries = [(q.query_id, q.text) for q in ds.queries_iter()]
    qrels: dict[str, set[str]] = {}
    for qrel in ds.qrels_iter():
        qrels.setdefault(qrel.query_id, set()).add(qrel.doc_id)
    print(f"  {len(queries):,} queries, {len(qrels):,} have qrels")

    print("\nLoading ColBERTv2 encoder...")
    from colbert.modeling.checkpoint import Checkpoint
    from colbert.infra import ColBERTConfig
    from huggingface_hub import snapshot_download
    ckpt_dir = snapshot_download("colbert-ir/colbertv2.0")
    encoder = Checkpoint(ckpt_dir, colbert_config=ColBERTConfig(query_maxlen=32))
    encoder = encoder.eval().to(DEVICE)

    qids = [q[0] for q in queries]
    texts = [q[1] for q in queries]
    t0 = time.time()
    Q_chunks = []
    for i in range(0, len(texts), 512):
        with torch.inference_mode():
            with torch.amp.autocast(DEVICE, dtype=torch.float16):
                Q_c = encoder.queryFromText(texts[i:i+512], bsize=64)
        Q_chunks.append(Q_c.float())
    Q_emb = torch.cat(Q_chunks, dim=0)  # (Qn, 32, 128)
    print(f"  query encoding done in {time.time()-t0:.1f}s, shape {tuple(Q_emb.shape)}")
    del encoder
    torch.cuda.empty_cache()

    # ---- Apply ITQ R to queries (no scale head; scale head is SubBit-only) -
    Q_proj = Q_emb @ R.T   # (Qn, m, r)
    print(f"  queries projected: {tuple(Q_proj.shape)} (no scale head)")
    Qn, m_q, _ = Q_proj.shape

    # ---- Chunked asymmetric MaxSim, sort docs by length ---------------
    pids = list(offsets.keys())
    starts = np.array([offsets[p][0] for p in pids], dtype=np.int64)
    lengths = np.array([offsets[p][1] for p in pids], dtype=np.int32)
    sort_order = np.argsort(lengths)
    pids = [pids[i] for i in sort_order]
    starts = starts[sort_order]
    lengths = lengths[sort_order]

    Dn = len(pids)
    n_chunks = (Dn + doc_chunk - 1) // doc_chunk
    sweep_ckpt_path = cache / "results" / "itq_sweep_checkpoint.pt"
    sweep_ckpt_path.parent.mkdir(parents=True, exist_ok=True)

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

        D_pad = torch.zeros(seg_n, seg_max, r, dtype=torch.float16, device=DEVICE)
        for i in range(seg_n):
            s, ln = int(starts[da + i]), int(lengths[da + i])
            D_pad[i, :ln] = packed[s:s+ln].to(torch.float16)
        col = torch.arange(seg_max, device=DEVICE)
        valid = col[None, :] < seg_lengths[:, None]

        for qa in range(0, Qn, query_batch):
            qb = min(qa + query_batch, Qn)
            Q_b = Q_proj[qa:qb].to(torch.float16)
            sim = torch.einsum("qmr,dnr->qmdn", Q_b, D_pad)
            sim.masked_fill_(~valid[None, None, :, :], float("-inf"))
            scores = sim.max(dim=-1).values.sum(dim=1).float()
            del sim

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

        if (ci + 1) % checkpoint_every_chunks == 0 and (ci + 1) < n_chunks:
            torch.save({
                "top_scores": top_scores.cpu(),
                "top_indices": top_indices.cpu(),
                "next_ci": ci + 1,
            }, sweep_ckpt_path)
            vol.commit()
            print(f"  checkpointed at chunk {ci+1}/{n_chunks}", flush=True)

    torch.cuda.synchronize()
    sweep_secs = time.time() - t0
    print(f"\nSweep done in {sweep_secs/60:.1f} min")
    if sweep_ckpt_path.exists():
        sweep_ckpt_path.unlink()

    # ---- Metrics ------------------------------------------------------
    print("\nComputing metrics...")
    top_indices_cpu = top_indices.cpu().numpy()
    mrr_at_10, recall_at_100, recall_at_1000, n_eval = 0.0, 0.0, 0.0, 0
    for q_idx, qid in enumerate(qids):
        relevant = qrels.get(qid)
        if not relevant:
            continue
        retrieved = [pids[idx] for idx in top_indices_cpu[q_idx] if idx >= 0]
        rr = 0.0
        for rank, p in enumerate(retrieved[:10], start=1):
            if p in relevant:
                rr = 1.0 / rank
                break
        mrr_at_10 += rr
        n_rel = len(relevant)
        recall_at_100 += sum(1 for p in retrieved[:100] if p in relevant) / n_rel
        recall_at_1000 += sum(1 for p in retrieved[:1000] if p in relevant) / n_rel
        n_eval += 1

    metrics = {
        "mrr@10": mrr_at_10 / n_eval,
        "recall@100": recall_at_100 / n_eval,
        "recall@1000": recall_at_1000 / n_eval,
    }
    print(json.dumps(metrics, indent=2))

    results = {
        "method": "itq",
        "R_path": "/itq/R_r64.pt",
        "encoder": "colbert-ir/colbertv2.0",
        "corpus": "msmarco-passage (full)",
        "queries": MSMARCO_DEV_SMALL,
        "n_docs": n_docs,
        "n_total_tokens": n_total_tokens,
        "n_queries_evaluated": n_eval,
        "r": r,
        "use_scale": False,
        "scoring": "asymmetric",
        "doc_chunk": doc_chunk,
        "query_batch": query_batch,
        "top_k": top_k,
        "metrics": metrics,
        "wall_clock_sec": {"sweep": sweep_secs},
    }
    out_path = cache / "results" / "itq_full_msmarco.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    vol.commit()
    print(f"\nWrote {out_path}")


@app.local_entrypoint()
def main():
    run.remote()
