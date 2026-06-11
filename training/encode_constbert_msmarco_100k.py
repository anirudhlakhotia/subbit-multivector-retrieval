"""Encode the same MS MARCO 100k passage subset with pinecone/ConstBERT.

Mirrors the layout of `data/embeddings/jina_colbert_v2/100k/` so the
downstream lex-coverage analysis can reuse the existing pid set, qrels,
and query universe.

Writes:
  data/embeddings/constbert/100k/doc_embeddings.pt   # {pid: (32, 128)}
  data/embeddings/constbert/100k/query_embeddings.pt # {qid: (32, 128)}
  data/embeddings/constbert/100k/qrels.tsv           # copy of msmarco/100k qrels

The pid universe is taken from data/embeddings/msmarco/100k/doc_embeddings.pt
so the corpus is bit-identical to the ColBERTv2 / Jina runs.

Usage:
    .venv/bin/python training/encode_constbert_msmarco_100k.py \
        --batch-size 16 --device mps

Smoke-test mode (encodes only the first --max-docs / --max-queries):
    .venv/bin/python training/encode_constbert_msmarco_100k.py \
        --max-docs 500 --max-queries 50 --output-dir data/embeddings/constbert/100k_smoke
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(".").resolve()))
from src.subbit.encoders import ConstBERTEncoder
from src.subbit.utils import get_device

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SRC = REPO_ROOT / "data/embeddings/msmarco/100k"
DEFAULT_DST = REPO_ROOT / "data/embeddings/constbert/100k"
DEFAULT_IR_DATASETS = Path.home() / ".ir_datasets/msmarco-passage"


def load_doc_text(pid_set: set[str], collection_tsv: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with collection_tsv.open("r", encoding="utf-8") as f:
        for line in f:
            pid, text = line.rstrip("\n").split("\t", 1)
            if pid in pid_set:
                out[pid] = text
                if len(out) == len(pid_set):
                    break
    missing = pid_set - set(out.keys())
    if missing:
        raise RuntimeError(
            f"missing {len(missing)} pids from {collection_tsv}; e.g. {next(iter(missing))}"
        )
    return out


def load_query_text(qid_set: set[str], queries_tsv: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with queries_tsv.open("r", encoding="utf-8") as f:
        for line in f:
            qid, text = line.rstrip("\n").split("\t", 1)
            if qid in qid_set:
                out[qid] = text
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", type=Path, default=DEFAULT_SRC,
                    help="ColBERTv2 100k cache; defines the pid + qid universe.")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_DST)
    ap.add_argument("--device", default=None,
                    help="auto / cpu / mps / cuda. Default: utils.get_device('auto').")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-docs", type=int, default=None, help="Smoke-test: limit doc count.")
    ap.add_argument("--max-queries", type=int, default=None, help="Smoke-test: limit query count.")
    ap.add_argument("--collection-tsv", type=Path,
                    default=DEFAULT_IR_DATASETS / "collection.tsv")
    ap.add_argument("--queries-tsv", type=Path,
                    default=DEFAULT_IR_DATASETS / "dev/small/queries.tsv")
    ap.add_argument("--skip-docs", action="store_true", help="Skip doc encoding (queries only).")
    ap.add_argument("--skip-queries", action="store_true", help="Skip query encoding (docs only).")
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else get_device("auto")
    print(f"[constbert] device={device}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Define the doc + query universe from the ColBERTv2 100k cache so the
    # corpora are bit-identical across encoders.
    print(f"[constbert] reading pid/qid universe from {args.src_dir}", flush=True)
    src_docs = torch.load(args.src_dir / "doc_embeddings.pt", map_location="cpu", weights_only=False)
    src_qrys = torch.load(args.src_dir / "query_embeddings.pt", map_location="cpu", weights_only=False)
    pids = list(src_docs.keys())
    qids = list(src_qrys.keys())
    del src_docs, src_qrys

    if args.max_docs is not None:
        pids = pids[: args.max_docs]
    if args.max_queries is not None:
        qids = qids[: args.max_queries]
    print(f"[constbert] {len(pids)} docs, {len(qids)} queries", flush=True)

    # Pull surface text
    print(f"[constbert] loading collection text ({len(pids)} pids)...", flush=True)
    t0 = time.perf_counter()
    doc_text = load_doc_text(set(pids), args.collection_tsv)
    print(f"[constbert]   loaded {len(doc_text)} doc texts in {time.perf_counter()-t0:.1f}s", flush=True)

    print(f"[constbert] loading query text ({len(qids)} qids)...", flush=True)
    qry_text = load_query_text(set(qids), args.queries_tsv)
    missing_q = set(qids) - set(qry_text.keys())
    if missing_q:
        print(f"[constbert]   WARNING: {len(missing_q)} qids missing from queries.tsv; dropping", flush=True)
        qids = [q for q in qids if q in qry_text]

    enc = ConstBERTEncoder()

    # Encode docs
    if not args.skip_docs:
        out_doc = args.output_dir / "doc_embeddings.pt"
        print(f"[constbert] encoding {len(pids)} docs (bsize={args.batch_size}) -> {out_doc}", flush=True)
        t0 = time.perf_counter()
        doc_texts_ordered = [doc_text[pid] for pid in pids]
        doc_idx_emb = enc.encode_documents(doc_texts_ordered, batch_size=args.batch_size, device=device)
        # Re-key by pid string
        doc_emb = {pids[i]: doc_idx_emb[i] for i in range(len(pids))}
        # Sanity: shapes + no NaN/Inf
        sample = doc_emb[pids[0]]
        print(f"[constbert]   first doc shape={tuple(sample.shape)} dtype={sample.dtype}", flush=True)
        bad = 0
        for k, v in doc_emb.items():
            if torch.isnan(v).any() or torch.isinf(v).any():
                bad += 1
        if bad:
            raise RuntimeError(f"{bad} doc embeddings contain NaN/Inf")
        torch.save(doc_emb, out_doc)
        print(f"[constbert]   wrote {out_doc} ({len(doc_emb)} entries) in {time.perf_counter()-t0:.1f}s", flush=True)

    # Encode queries
    if not args.skip_queries:
        out_qry = args.output_dir / "query_embeddings.pt"
        print(f"[constbert] encoding {len(qids)} queries -> {out_qry}", flush=True)
        t0 = time.perf_counter()
        qry_texts_ordered = [qry_text[qid] for qid in qids]
        qry_idx_emb = enc.encode_queries(qry_texts_ordered, batch_size=args.batch_size, device=device)
        qry_emb = {qids[i]: qry_idx_emb[i] for i in range(len(qids))}
        sample = qry_emb[qids[0]]
        print(f"[constbert]   first query shape={tuple(sample.shape)} dtype={sample.dtype}", flush=True)
        bad = 0
        for k, v in qry_emb.items():
            if torch.isnan(v).any() or torch.isinf(v).any():
                bad += 1
        if bad:
            raise RuntimeError(f"{bad} query embeddings contain NaN/Inf")
        torch.save(qry_emb, out_qry)
        print(f"[constbert]   wrote {out_qry} ({len(qry_emb)} entries) in {time.perf_counter()-t0:.1f}s", flush=True)

    # Copy qrels
    src_qrels = args.src_dir / "qrels.tsv"
    dst_qrels = args.output_dir / "qrels.tsv"
    if src_qrels.exists() and not dst_qrels.exists():
        shutil.copy2(src_qrels, dst_qrels)
        print(f"[constbert] copied qrels -> {dst_qrels}", flush=True)

    print("[constbert] done.", flush=True)


if __name__ == "__main__":
    main()
