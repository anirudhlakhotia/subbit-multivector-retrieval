"""Modal app: BEIR-9 cross-domain ablation -- sign(d) vs random R r=64.

Runs **both methods on all 9 cached BEIR datasets in one container session**
so we pay container startup + model warm-up once, not per-dataset. Each
method is pre-projected once per dataset (write into a single int8 ±1
GPU tensor, mmap-style chunked sweep with no per-chunk re-projection).

Methods evaluated per dataset (matched 16 B/tok, r=128):
    1. sign(d)          -- identity rotation, no projection
    2. random R r=64   -- canonical seed=42 R (matches the 8.8M run)

No fp32 reference computed -- corpus at nq scale doesn't fit on A100-80GB
in fp32. Reference numbers for retention come from
``outputs/50k_topk/results/cross_domain_results.json`` (existing local
runs cover the BEIR-5 small set) and the published ColBERTv2 paper for
the larger four (trec-covid, touche-2020, quora, nq).

Reads (per dataset):
    /cache/beir/<dataset>/embeddings.bin     fp32 mmap
    /cache/beir/<dataset>/offsets.pt         {pid -> (start, length)}
    /cache/beir/<dataset>/queries.pt         {qid -> Tensor}
    /cache/beir/<dataset>/qrels.json         {qid -> {pid -> int}}
    /cache/beir/<dataset>/meta.json

Writes (per dataset):
    /cache/results/beir_<dataset>_sign_vs_random_r64.json     aggregate metrics
    /cache/results/beir_<dataset>_sign_vs_random_r64.per_query.pt
                                                                per-query sidecar

Plus a roll-up:
    /cache/results/beir9_sign_vs_random_r64_summary.json

Run::

    modal run --detach full_scale/eval_beir9_full_r64.py::run

Cost estimate (A100-80GB at ~$2.50/hr):
    Per-dataset wall-clock is dominated by the sweep einsums. With
    doc_chunk=4096, query_batch=64, expected ~5-50 min for the small
    datasets combined, ~30-50 min for nq alone. Total: ~$2-3.50 for all
    nine.
"""
from __future__ import annotations

import modal

APP_NAME = "subbit-eval-beir9-r64"
VOLUME_NAME = "subbit-msmarco-cache-v2"

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False, version=2)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "wget", "build-essential")
    .pip_install(
        "torch==2.3.1",
        "transformers==4.41.2",
        "huggingface_hub==0.23.4",
        "tqdm==4.66.4",
        "ujson==5.10.0",
        "numpy<2",
    )
)

DIM = 128
R_DIM = 64
BYTES_PER_TOK = 8
DATASETS_DEFAULT = ",".join([
    "nfcorpus", "scifact", "arguana", "scidocs", "fiqa",
    "trec-covid", "webis-touche2020", "quora", "nq",
])


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/cache": vol},
    timeout=10800,            # 3 hours; expected <90 min
    memory=64 * 1024,
    cpu=8.0,
)
def run(
    datasets: str = DATASETS_DEFAULT,
    doc_chunk: int = 4096,
    query_batch: int = 64,
    top_k: int = 1000,
    seed: int = 42,
):
    import os, json, math, time
    from pathlib import Path
    import numpy as np
    import torch

    DEVICE = "cuda"
    ds_list = [d.strip() for d in datasets.split(",") if d.strip()]
    print(f"BEIR-9 eval: {len(ds_list)} datasets", flush=True)
    print(f"  datasets: {ds_list}", flush=True)
    print(f"  doc_chunk={doc_chunk}  query_batch={query_batch}  "
          f"top_k={top_k}  seed={seed}", flush=True)

    # ---- Random R r=64 (canonical seed=42, first R_DIM rows of Q.T;
    #      same construction as full_scale/eval_rand_plus_scale_full.py
    #      RAND_SEED path). --------------------------------------------------
    torch.manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(DIM, DIM))
    R = Q.T[:R_DIM].contiguous().to(DEVICE).float()       # (R_DIM, DIM)
    print(f"\nR r={R_DIM}: shape={tuple(R.shape)}  "
          f"||R||_F={R.norm().item():.4f} (seed={seed})", flush=True)
    Rcpu = R.cpu()
    print(f"  R[0, :4]  = {Rcpu[0, :4].tolist()}", flush=True)
    print(f"  R[32, :4] = {Rcpu[32, :4].tolist()}", flush=True)
    del Rcpu

    cache = Path("/cache")
    results_dir = cache / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    def idcg_at(k):
        return sum(1.0 / math.log2(i + 1) for i in range(1, k + 1))

    def compute_metrics(top_indices_cpu, top_scores_cpu, qids, pids, qrels):
        """Aggregate IR metrics + per-query arrays. Returns (agg_dict, pq_dict)."""
        Q_n = len(qids)
        pq = {
            "rr@10": np.full(Q_n, np.nan, dtype=np.float64),
            "recall@10": np.full(Q_n, np.nan, dtype=np.float64),
            "recall@100": np.full(Q_n, np.nan, dtype=np.float64),
            "recall@1000": np.full(Q_n, np.nan, dtype=np.float64),
            "ndcg@10": np.full(Q_n, np.nan, dtype=np.float64),
            "ndcg@100": np.full(Q_n, np.nan, dtype=np.float64),
            "ndcg@1000": np.full(Q_n, np.nan, dtype=np.float64),
            "ap@1000": np.full(Q_n, np.nan, dtype=np.float64),
            "success@1": np.full(Q_n, np.nan, dtype=np.float64),
            "success@10": np.full(Q_n, np.nan, dtype=np.float64),
            "first_relevant_rank": np.full(Q_n, -1, dtype=np.int64),
            "n_relevant": np.zeros(Q_n, dtype=np.int64),
        }
        for q_idx, qid in enumerate(qids):
            rel_set = set(qrels.get(qid, {}).keys()) if qrels.get(qid) else set()
            rel_set = {p for p in rel_set if qrels[qid].get(p, 0) > 0}
            if not rel_set:
                continue
            retrieved = [pids[idx] for idx in top_indices_cpu[q_idx] if idx >= 0]
            n_rel = len(rel_set)
            pq["n_relevant"][q_idx] = n_rel
            first_rank = -1
            for rank, p in enumerate(retrieved, start=1):
                if p in rel_set:
                    first_rank = rank
                    break
            pq["first_relevant_rank"][q_idx] = first_rank
            pq["rr@10"][q_idx] = (1.0 / first_rank) if 1 <= first_rank <= 10 else 0.0
            pq["success@1"][q_idx] = 1.0 if first_rank == 1 else 0.0
            pq["success@10"][q_idx] = 1.0 if 1 <= first_rank <= 10 else 0.0
            hit10 = sum(1 for p in retrieved[:10] if p in rel_set)
            hit100 = sum(1 for p in retrieved[:100] if p in rel_set)
            hit1000 = sum(1 for p in retrieved[:1000] if p in rel_set)
            pq["recall@10"][q_idx] = hit10 / n_rel
            pq["recall@100"][q_idx] = hit100 / n_rel
            pq["recall@1000"][q_idx] = hit1000 / n_rel
            dcg10 = dcg100 = dcg1000 = 0.0
            ap_num = 0.0; ap_hits = 0
            for i, p in enumerate(retrieved[:1000], start=1):
                if p in rel_set:
                    disc = 1.0 / math.log2(i + 1)
                    if i <= 10:  dcg10 += disc
                    if i <= 100: dcg100 += disc
                    dcg1000 += disc
                    ap_hits += 1
                    ap_num += ap_hits / i
            pq["ndcg@10"][q_idx]   = dcg10 / idcg_at(min(10, n_rel))
            pq["ndcg@100"][q_idx]  = dcg100 / idcg_at(min(100, n_rel))
            pq["ndcg@1000"][q_idx] = dcg1000 / idcg_at(min(1000, n_rel))
            pq["ap@1000"][q_idx]   = ap_num / n_rel
        has = ~np.isnan(pq["rr@10"])
        n_eval = int(has.sum())
        agg = {
            "n_eval": n_eval,
            "mrr@10":      float(pq["rr@10"][has].mean()),
            "recall@10":   float(pq["recall@10"][has].mean()),
            "recall@100":  float(pq["recall@100"][has].mean()),
            "recall@1000": float(pq["recall@1000"][has].mean()),
            "ndcg@10":     float(pq["ndcg@10"][has].mean()),
            "ndcg@100":    float(pq["ndcg@100"][has].mean()),
            "ndcg@1000":   float(pq["ndcg@1000"][has].mean()),
            "map@1000":    float(pq["ap@1000"][has].mean()),
            "success@1":   float(pq["success@1"][has].mean()),
            "success@10":  float(pq["success@10"][has].mean()),
        }
        return agg, pq

    def sweep_method(method_name, packed_corpus_i8, query_proj, query_mask,
                     qids, pids, starts_t, lengths_t, n_docs_eff):
        """Run the chunked MaxSim sweep.

        packed_corpus_i8: int8 +-1 (n_total_tokens, R_DIM) on GPU. Cast to
                          fp16 happens per-chunk inside the loop so we never
                          materialise a full fp16 copy (would OOM at nq).
        starts_t:         (n_docs_eff,) int64 -- ORIGINAL start positions in
                          the packed corpus (from offsets.pt), reordered to
                          match the length-sorted pids order. MUST be the
                          true starts, NOT cumsum of sorted lengths -- the
                          corpus is in original encoding order, not sorted.
        lengths_t:        (n_docs_eff,) int32 -- per-doc lengths in same order.
        query_proj:       fp32 (n_q, m_max, R_DIM) on GPU
        query_mask:       bool (n_q, m_max) on GPU
        """
        Qn, m_max, _ = query_proj.shape
        top_scores = torch.full((Qn, top_k), -float("inf"), device=DEVICE)
        top_indices = torch.full((Qn, top_k), -1, dtype=torch.int64, device=DEVICE)

        t_load = t_score = t_topk = 0.0

        n_chunks = (n_docs_eff + doc_chunk - 1) // doc_chunk
        t_sweep0 = time.time()
        for ci, da in enumerate(range(0, n_docs_eff, doc_chunk)):
            db = min(da + doc_chunk, n_docs_eff)
            seg_lengths = lengths_t[da:db]
            seg_starts = starts_t[da:db]
            seg_max = int(seg_lengths.max().item())
            seg_n = db - da

            # ---- (1) Build D_pad: slice int8 corpus + cast to fp16 per-doc.
            # Per-chunk D_pad is small (e.g. 4096 x 200 x 128 x 2 = 200 MB) so
            # the fp16 materialisation is contained; the int8 corpus stays
            # resident, no full-corpus fp16 copy is ever created.
            t = time.time()
            D_pad = torch.zeros(seg_n, seg_max, R_DIM, dtype=torch.float16, device=DEVICE)
            for i in range(seg_n):
                s = int(seg_starts[i].item())
                ln = int(seg_lengths[i].item())
                D_pad[i, :ln] = packed_corpus_i8[s:s+ln].to(torch.float16)
            torch.cuda.synchronize()
            t_load += time.time() - t

            col = torch.arange(seg_max, device=DEVICE)
            valid = col[None, :] < seg_lengths[:, None]

            for qa in range(0, Qn, query_batch):
                qb = min(qa + query_batch, Qn)
                Q_b = query_proj[qa:qb].to(torch.float16)
                Qmask_b = query_mask[qa:qb]

                # ---- (2) Score ------------------------------------------
                t = time.time()
                sim = torch.einsum("qmr,dnr->qmdn", Q_b, D_pad)
                sim.masked_fill_(~valid[None, None, :, :], float("-inf"))
                per_qtok_max = sim.max(dim=-1).values
                per_qtok_max = per_qtok_max.masked_fill(~Qmask_b[:, :, None], 0.0)
                # Cast to fp32 BEFORE the sum-over-q-tokens. Summing 32
                # fp16 maxes in fp16 accumulates ~0.005 rounding error each
                # -> ~0.16 score noise per query, enough to flip borderline
                # rank decisions and drift MRR by 0.01-0.03 from fp32 ref.
                # Cast tensor is tiny (~32 MB at nq scale); cost is negligible.
                scores = per_qtok_max.float().sum(dim=1)
                torch.cuda.synchronize()
                t_score += time.time() - t
                del sim, per_qtok_max

                # ---- (3) Top-k merge -------------------------------------
                t = time.time()
                doc_indices = torch.arange(da, db, device=DEVICE).expand(qb - qa, -1)
                combined_scores = torch.cat([top_scores[qa:qb], scores], dim=1)
                combined_indices = torch.cat([top_indices[qa:qb], doc_indices], dim=1)
                top_scores[qa:qb], top_pos = combined_scores.topk(top_k, dim=1)
                top_indices[qa:qb] = combined_indices.gather(1, top_pos)
                torch.cuda.synchronize()
                t_topk += time.time() - t

            del D_pad
        sweep_secs = time.time() - t_sweep0
        return (top_scores.cpu().numpy(), top_indices.cpu().numpy(),
                sweep_secs, t_load, t_score, t_topk)

    summary = []

    for ds_name in ds_list:
        ds_dir = cache / "beir" / ds_name
        if not (ds_dir / "embeddings.bin").exists():
            print(f"\n[SKIP] {ds_name}: not encoded (missing embeddings.bin)",
                  flush=True)
            summary.append({"dataset": ds_name, "status": "missing"})
            continue

        print(f"\n=== {ds_name} ===", flush=True)
        t_ds0 = time.time()
        try:
            meta = json.loads((ds_dir / "meta.json").read_text())
            n_docs = meta["n_docs"]
            n_total_tokens = meta["total_tokens"]
            print(f"  n_docs={n_docs:,}  n_total_tokens={n_total_tokens:,}",
                  flush=True)

            mmap_arr = np.memmap(
                ds_dir / "embeddings.bin", dtype=np.float32, mode="r",
                shape=(n_total_tokens, DIM),
            )
            offsets = torch.load(ds_dir / "offsets.pt", weights_only=False)
            queries = torch.load(ds_dir / "queries.pt", weights_only=False)
            qrels = json.loads((ds_dir / "qrels.json").read_text())

            # Filter to judged queries.
            judged_qids = sorted(q for q in queries
                                  if q in qrels and any(v > 0 for v in qrels[q].values()))
            n_q = len(judged_qids)
            if n_q == 0:
                print(f"  [SKIP] no judged queries", flush=True)
                summary.append({"dataset": ds_name, "status": "no_qrels"})
                continue
            print(f"  n_judged_queries={n_q:,}", flush=True)

            # Doc list, length-sorted (shorter docs first -> narrower padded chunks).
            pids = list(offsets.keys())
            starts = np.array([offsets[p][0] for p in pids], dtype=np.int64)
            lengths = np.array([offsets[p][1] for p in pids], dtype=np.int32)
            sort_order = np.argsort(lengths)
            pids = [pids[i] for i in sort_order]
            starts = starts[sort_order]
            lengths = lengths[sort_order]

            # ---- Pre-pack full corpus into GPU as int8 ±1 sign(d) AND sign(R d) ----
            # Two corpora resident on GPU simultaneously: each is 1 byte/coord.
            # nq worst case: 2.7M docs * ~60 tokens * 128 * 1 B * 2 methods = ~40 GB.
            # Fits A100-80GB with ~40 GB headroom.
            print(f"  pre-packing corpora (sign(d) + random R r=64 sign-coded)...",
                  flush=True)
            t = time.time()
            packed_sign_d = torch.empty(n_total_tokens, R_DIM,
                                        dtype=torch.int8, device=DEVICE)
            packed_rand_R = torch.empty(n_total_tokens, R_DIM,
                                        dtype=torch.int8, device=DEVICE)
            PROJECT_CHUNK = 1_000_000
            for ps in range(0, n_total_tokens, PROJECT_CHUNK):
                pe = min(ps + PROJECT_CHUNK, n_total_tokens)
                chunk_np = mmap_arr[ps:pe].copy()
                chunk = torch.from_numpy(chunk_np).to(DEVICE)   # (PC, DIM=128)

                # sign(d) at r=R_DIM: keep first R_DIM coords, sign-code.
                # (At R_DIM=128 this matches the original r=128 sign(d) path.)
                d_sub = chunk[:, :R_DIM].contiguous()
                bin_sd = torch.sign(d_sub)
                bin_sd[bin_sd == 0] = 1.0
                packed_sign_d[ps:pe] = bin_sd.to(torch.int8)

                # sign(R d) at r=R_DIM: project full DIM through R (R_DIM, DIM)
                # then sign-code. R was built as Q.T[:R_DIM] above.
                with torch.amp.autocast(DEVICE, dtype=torch.float16):
                    proj = chunk @ R.T                          # (PC, R_DIM)
                proj = proj.float()
                bin_rr = torch.sign(proj)
                bin_rr[bin_rr == 0] = 1.0
                packed_rand_R[ps:pe] = bin_rr.to(torch.int8)
                del chunk, d_sub, bin_sd, proj, bin_rr
            torch.cuda.synchronize()
            print(f"    packed in {time.time()-t:.1f}s "
                  f"({2 * n_total_tokens * R_DIM / 1e9:.1f} GB GPU), "
                  f"GPU mem {torch.cuda.memory_allocated()/1e9:.1f} GB",
                  flush=True)

            # ---- Pre-encode queries on GPU ---------------------------------
            t = time.time()
            q_real_lengths = [queries[q].shape[0] for q in judged_qids]
            m_max = max(q_real_lengths)
            Q_emb = torch.zeros(n_q, m_max, DIM, dtype=torch.float32, device=DEVICE)
            Q_mask = torch.zeros(n_q, m_max, dtype=torch.bool, device=DEVICE)
            for i, qid in enumerate(judged_qids):
                qe = queries[qid].float().to(DEVICE)
                Q_emb[i, :qe.shape[0]] = qe
                Q_mask[i, :qe.shape[0]] = True
            # For sign(d) method, query is raw fp32. For random R, query is Rq.
            Q_proj_R = Q_emb @ R.T
            print(f"    queries built in {time.time()-t:.1f}s "
                  f"({Q_emb.shape})", flush=True)

            lengths_t = torch.from_numpy(lengths).to(DEVICE)
            # Critical: starts_t comes from offsets.pt directly, NOT from
            # cumsum of sorted lengths. The packed corpus is in original
            # encoding order; doing cumsum of sorted lengths produces
            # length-monotone fake positions that don't index the real
            # token bytes -> MaxSim scores garbage tokens -> MRR ~ 0.
            starts_t = torch.from_numpy(starts).to(DEVICE)

            # ---- Sweep: sign(d) -------------------------------------------
            # At r=R_DIM, sign(d) keeps the first R_DIM coords. Query must
            # match -- slice Q_emb to first R_DIM dims for the einsum.
            print(f"  sweep: sign(d) r={R_DIM}...", flush=True)
            Q_sd = Q_emb[..., :R_DIM].contiguous()
            ts_sd, ti_sd, sweep_sd, l_sd, sc_sd, tk_sd = sweep_method(
                "sign_d", packed_sign_d,
                Q_sd, Q_mask, judged_qids, pids, starts_t, lengths_t, n_docs)
            print(f"    sign(d) sweep done in {sweep_sd:.1f}s "
                  f"(load {l_sd:.1f}s  score {sc_sd:.1f}s  topk {tk_sd:.1f}s)",
                  flush=True)
            agg_sd, pq_sd = compute_metrics(ti_sd, ts_sd, judged_qids, pids, qrels)
            print(f"    sign(d): MRR@10={agg_sd['mrr@10']:.4f}  "
                  f"NDCG@10={agg_sd['ndcg@10']:.4f}  "
                  f"R@100={agg_sd['recall@100']:.4f}",
                  flush=True)
            # Free packed_sign_d to make headroom for the second sweep.
            del packed_sign_d
            torch.cuda.empty_cache()

            # ---- Sweep: random R r=64 -----------------------------------
            print(f"  sweep: random R r=64...", flush=True)
            ts_rr, ti_rr, sweep_rr, l_rr, sc_rr, tk_rr = sweep_method(
                "rand_R", packed_rand_R,
                Q_proj_R, Q_mask, judged_qids, pids, starts_t, lengths_t, n_docs)
            print(f"    random R sweep done in {sweep_rr:.1f}s "
                  f"(load {l_rr:.1f}s  score {sc_rr:.1f}s  topk {tk_rr:.1f}s)",
                  flush=True)
            agg_rr, pq_rr = compute_metrics(ti_rr, ts_rr, judged_qids, pids, qrels)
            print(f"    random R: MRR@10={agg_rr['mrr@10']:.4f}  "
                  f"NDCG@10={agg_rr['ndcg@10']:.4f}  "
                  f"R@100={agg_rr['recall@100']:.4f}",
                  flush=True)

            # ---- Free memory before next dataset --------------------------
            # packed_sign_d already freed after first sweep above.
            del packed_rand_R, Q_emb, Q_proj_R, Q_mask, Q_sd
            torch.cuda.empty_cache()

            # ---- Save per-dataset JSON + sidecar --------------------------
            ds_secs = time.time() - t_ds0
            delta_mrr = agg_sd["mrr@10"] - agg_rr["mrr@10"]
            delta_ndcg = agg_sd["ndcg@10"] - agg_rr["ndcg@10"]
            print(f"  delta sign(d) - random R: MRR@10={delta_mrr:+.4f}  "
                  f"NDCG@10={delta_ndcg:+.4f}", flush=True)

            agg_out = {
                "dataset": ds_name,
                "n_docs": n_docs,
                "n_total_tokens": n_total_tokens,
                "n_queries_evaluated": agg_sd["n_eval"],
                "r": R_DIM,
                "bytes_per_token": BYTES_PER_TOK,
                "scoring": "asymmetric",
                "seed": seed,
                "methods": {
                    "sign_d_r64":         {**agg_sd, "sweep_secs": sweep_sd,
                                            "t_load": l_sd, "t_score": sc_sd,
                                            "t_topk": tk_sd},
                    "random_R_r64_sign":  {**agg_rr, "sweep_secs": sweep_rr,
                                            "t_load": l_rr, "t_score": sc_rr,
                                            "t_topk": tk_rr},
                },
                "delta_sign_d_minus_random_R": {
                    "mrr@10":  delta_mrr,
                    "ndcg@10": delta_ndcg,
                    "recall@100":  agg_sd["recall@100"]  - agg_rr["recall@100"],
                    "recall@1000": agg_sd["recall@1000"] - agg_rr["recall@1000"],
                },
                "wall_clock_sec": ds_secs,
            }
            out_json = results_dir / f"beir_{ds_name}_sign_vs_random_r64.json"
            out_json.write_text(json.dumps(agg_out, indent=2))
            print(f"    wrote {out_json}", flush=True)

            sidecar = {
                "dataset": ds_name,
                "qids": judged_qids,
                "pids_sorted_by_length": pids,
                "sign_d": {
                    "top1000_indices": ti_sd.astype(np.int64),
                    "top1000_scores": ts_sd.astype(np.float32),
                    "pq": pq_sd,
                },
                "random_R": {
                    "top1000_indices": ti_rr.astype(np.int64),
                    "top1000_scores": ts_rr.astype(np.float32),
                    "pq": pq_rr,
                },
            }
            sidecar_path = results_dir / f"beir_{ds_name}_sign_vs_random_r64.per_query.pt"
            torch.save(sidecar, sidecar_path)
            print(f"    wrote {sidecar_path} "
                  f"({sidecar_path.stat().st_size / 1e6:.1f} MB)", flush=True)

            summary.append({
                "dataset": ds_name, "status": "ok",
                "n_docs": n_docs, "n_queries_evaluated": agg_sd["n_eval"],
                "sign_d_mrr10": agg_sd["mrr@10"],
                "random_R_mrr10": agg_rr["mrr@10"],
                "delta_mrr10": delta_mrr,
                "wall_clock_min": ds_secs / 60,
            })
            vol.commit()

        except Exception as e:
            import traceback
            print(f"  ERROR on {ds_name}: {e!r}", flush=True)
            traceback.print_exc()
            summary.append({"dataset": ds_name, "status": "error",
                            "error": repr(e)})
            vol.commit()
            continue

    # ---- Roll-up summary --------------------------------------------------
    print("\n=== BEIR-9 SUMMARY ===", flush=True)
    print(f"{'dataset':<18} {'sign(d) MRR':>12} {'rand R MRR':>11} "
          f"{'delta':>9}  {'winner':>10}", flush=True)
    print("-" * 70, flush=True)
    for r in summary:
        if r["status"] != "ok":
            print(f"  {r['dataset']:<18}  {r['status']}", flush=True)
            continue
        winner = ("sign(d)" if r["delta_mrr10"] > 0.001 else
                  ("rand R" if r["delta_mrr10"] < -0.001 else "tie"))
        print(f"{r['dataset']:<18} {r['sign_d_mrr10']:>12.4f} "
              f"{r['random_R_mrr10']:>11.4f} {r['delta_mrr10']:>+9.4f} "
              f"{winner:>10s}", flush=True)

    rollup_path = results_dir / "beir9_sign_vs_random_r64_summary.json"
    rollup_path.write_text(json.dumps({
        "R_construction": ("torch.manual_seed(42); Q,_ = torch.linalg.qr("
                            "torch.randn(128,128)); R = Q.T"),
        "doc_chunk": doc_chunk,
        "query_batch": query_batch,
        "top_k": top_k,
        "seed": seed,
        "summary": summary,
    }, indent=2))
    vol.commit()
    print(f"\nWrote rollup: {rollup_path}", flush=True)


@app.local_entrypoint()
def main(datasets: str = DATASETS_DEFAULT):
    run.remote(datasets=datasets)
