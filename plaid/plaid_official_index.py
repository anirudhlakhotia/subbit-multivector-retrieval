#!/usr/bin/env python3
"""Index a collection with the OFFICIAL PLAID engine (colbert-ai).

Run inside .venv-plaid:
    .venv-plaid/bin/python plaid/plaid_official_index.py \
        --collection data/plaid_official/collection_100k.tsv \
        --name msmarco100k.nbits2 --nbits 2 \
        2>&1 | tee /tmp/plaid_official_index.log

doc_maxlen=180 matches the repo's 100k slice convention. CPU-only is fine
(colbert-ai falls back when no CUDA); indexing 100k docs on the M4 is the
slow part (BERT encode + faiss k-means), expect ~1-2 h.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# torch's JIT extension loader resolves the `ninja` BINARY via PATH in spawned
# subprocesses; invoking .venv-plaid/bin/python directly does not put the venv
# bin on PATH, so prepend it here (children inherit os.environ).
os.environ["PATH"] = f"{Path(sys.executable).parent}:{os.environ.get('PATH', '')}"

# torch and the faiss-cpu wheel each bundle their own libomp.dylib on macOS;
# faiss's k-means initializes the second copy and aborts (OMP Error #15).
# KMP_DUPLICATE_LIB_OK is the standard workaround for this pairing. Backstop
# against the "silently incorrect" caveat: the built index's MRR@10 is
# validated downstream against the codec-emulation expectation (b=2 ~ 0.859
# aug) -- corrupted k-means cannot pass that check.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--collection", default="data/plaid_official/collection_100k.tsv")
    p.add_argument("--name", required=True, help="index name, e.g. msmarco100k.nbits2")
    p.add_argument("--nbits", type=int, default=2)
    p.add_argument("--doc-maxlen", type=int, default=180)
    p.add_argument("--checkpoint", default="colbert-ir/colbertv2.0")
    p.add_argument("--root", default="data/plaid_official/experiments")
    args = p.parse_args()

    from colbert import Indexer
    from colbert.infra import ColBERTConfig, Run, RunConfig

    with Run().context(RunConfig(nranks=1, experiment="plaid_official", root=args.root)):
        config = ColBERTConfig(nbits=args.nbits, doc_maxlen=args.doc_maxlen)
        indexer = Indexer(checkpoint=args.checkpoint, config=config)
        indexer.index(name=args.name, collection=args.collection, overwrite=True)
    print(f"index '{args.name}' built under {args.root}")


if __name__ == "__main__":
    main()
