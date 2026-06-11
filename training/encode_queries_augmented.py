"""Re-encode the MS MARCO dev queries WITH standard ColBERT query augmentation.

The local 100k slice's ``query_embeddings.pt`` was encoded with augmentation
STRIPPED (encode_corpus.py keeps only real tokens -> ~8 tokens/query, no query
expansion). That is non-standard and is the sole reason local MRR (0.8169)
differs from the augmented Modal pipeline (0.8505). This script produces a
drop-in augmented query cache so the local 100k evals can be re-run on the
*standard* (queryFromText, 32-token [MASK]-augmented) convention -- on the Mac,
no Modal cost.

It writes ``query_embeddings_aug.pt`` (dict qid -> (32,128)) next to the slice;
the original non-aug file is left untouched. Point evals at it via
``run_baseline_comparison.py --query-embeddings <path>``.

    python training/encode_queries_augmented.py \
        --out data/embeddings/msmarco/100k/query_embeddings_aug.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))  # make `src` importable


def load_colbert(device: str):
    """Load ColBERTv2 with query_maxlen=32 (standard augmentation), patching out
    the optional CUDA extension so it loads on CPU/MPS (mirrors encode_corpus.py)."""
    import os
    from huggingface_hub import snapshot_download
    from colbert.infra import ColBERTConfig
    from colbert.modeling.checkpoint import Checkpoint
    import colbert.modeling.colbert as colbert_module

    os.environ["COLBERT_LOAD_TORCH_EXTENSION_VERBOSE"] = "False"
    original = colbert_module.ColBERT.try_load_torch_extensions
    colbert_module.ColBERT.try_load_torch_extensions = classmethod(lambda cls, use_gpu: None)
    try:
        src = snapshot_download("colbert-ir/colbertv2.0")
    except Exception:
        src = "colbert-ir/colbertv2.0"
    try:
        ckpt = Checkpoint(src, colbert_config=ColBERTConfig(query_maxlen=32))
        ckpt = ckpt.to(device)
    finally:
        colbert_module.ColBERT.try_load_torch_extensions = original
    return ckpt.eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/embeddings/msmarco/100k/query_embeddings_aug.pt")
    ap.add_argument("--dataset", default="msmarco-passage/dev/small")
    ap.add_argument("--device", default=None, help="auto/cpu/mps/cuda (default: auto)")
    ap.add_argument("--batch", type=int, default=512)
    args = ap.parse_args()

    from src.subbit.utils import get_device
    device = str(get_device(args.device or "auto"))
    print(f"device: {device}")

    import ir_datasets
    ds = ir_datasets.load(args.dataset)
    qrel_qids = {qr.query_id for qr in ds.qrels_iter()}
    texts_by_qid = {q.query_id: q.text for q in ds.queries_iter() if q.query_id in qrel_qids}
    qids = list(texts_by_qid)
    print(f"{len(qids):,} dev queries to encode (augmented, query_maxlen=32)")

    ckpt = load_colbert(device)

    out: dict[str, torch.Tensor] = {}
    for i in range(0, len(qids), args.batch):
        batch_qids = qids[i:i + args.batch]
        batch_texts = [texts_by_qid[q] for q in batch_qids]
        with torch.inference_mode():
            Q = ckpt.queryFromText(batch_texts, bsize=64)  # (b, 32, 128), augmented
        for j, qid in enumerate(batch_qids):
            out[qid] = Q[j].float().cpu().clone()
        print(f"  encoded {min(i+args.batch, len(qids)):,}/{len(qids):,}", flush=True)

    sample = next(iter(out.values()))
    print(f"sample shape: {tuple(sample.shape)} (expect (32,128))")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print(f"wrote {len(out):,} augmented query embeddings -> {args.out}")


if __name__ == "__main__":
    main()
