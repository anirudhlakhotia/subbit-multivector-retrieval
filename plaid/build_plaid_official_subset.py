#!/usr/bin/env python3
"""Build the official-PLAID input files for the exact 100k diagnostic slice.

Produces (under data/plaid_official/):
  collection_100k.tsv   new_id \t passage_text     (new_id = 0..N-1, line order,
                                                    as the official ColBERT
                                                    Collection loader expects)
  pid_map.tsv           new_id \t original_pid     (to map search results back
                                                    onto the repo's qrels)
  queries_dev.tsv       qid \t query_text          (the 6,980 judged dev queries)

Inputs:
  - the 100k slice's pids   = keys of data/embeddings/msmarco/100k/doc_embeddings.pt
  - raw passage text        = ~/.ir_datasets/msmarco-passage/collection.tsv
  - dev query text          = located under ~/.ir_datasets/msmarco-passage/dev/
  - judged qids             = data/embeddings/msmarco/100k/qrels.tsv

Pure I/O; ~2-3 min (streams the 8.8M-line collection once).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
EMB = REPO / "data/embeddings/msmarco/100k"
IRDS = Path.home() / ".ir_datasets/msmarco-passage"
OUT = REPO / "data/plaid_official"


def find_queries_file() -> Path:
    """Locate the dev queries tsv inside the ir_datasets cache."""
    candidates = list((IRDS / "dev").rglob("*queries*")) + list((IRDS / "dev").rglob("*.tsv"))
    for c in candidates:
        if c.is_file():
            # validate: first line looks like "qid\ttext"
            with open(c, errors="replace") as f:
                first = f.readline().rstrip("\n").split("\t")
            if len(first) >= 2 and first[0].isdigit():
                return c
    raise FileNotFoundError(
        f"No dev queries tsv found under {IRDS/'dev'} — extract "
        f"queries.dev.small.tsv from collectionandqueries.tar.gz if needed."
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    print("[1/3] loading 100k pid list from doc_embeddings.pt (keys only)...", flush=True)
    store = torch.load(EMB / "doc_embeddings.pt", map_location="cpu", weights_only=False)
    pids = sorted(store.keys(), key=lambda p: int(p))
    pid_set = set(pids)
    del store
    print(f"      {len(pids):,} pids (min={pids[0]}, max={pids[-1]})", flush=True)

    print("[2/3] streaming collection.tsv -> collection_100k.tsv ...", flush=True)
    coll = IRDS / "collection.tsv"
    found: dict[str, str] = {}
    with open(coll, errors="replace") as f:
        for line in f:
            pid, _, text = line.partition("\t")
            if pid in pid_set:
                found[pid] = text.rstrip("\n")
                if len(found) == len(pid_set):
                    break
    missing = pid_set - found.keys()
    if missing:
        print(f"      WARNING: {len(missing)} pids not found in collection.tsv "
              f"(e.g. {sorted(missing, key=int)[:5]})", flush=True)
    with open(OUT / "collection_100k.tsv", "w") as fc, open(OUT / "pid_map.tsv", "w") as fm:
        for new_id, pid in enumerate(p for p in pids if p in found):
            fc.write(f"{new_id}\t{found[pid]}\n")
            fm.write(f"{new_id}\t{pid}\n")
    print(f"      wrote {len(found):,} passages", flush=True)

    print("[3/3] writing judged dev queries ...", flush=True)
    judged = set()
    with open(EMB / "qrels.tsv") as f:
        for line in f:
            judged.add(line.split("\t")[0].strip())
    qf = find_queries_file()
    print(f"      queries source: {qf}", flush=True)
    n = 0
    with open(qf, errors="replace") as f, open(OUT / "queries_dev.tsv", "w") as out:
        for line in f:
            qid, _, text = line.partition("\t")
            if qid in judged:
                out.write(f"{qid}\t{text.rstrip(chr(10))}\n")
                n += 1
    print(f"      wrote {n:,} queries (judged: {len(judged):,})", flush=True)
    if n != len(judged):
        print("      WARNING: judged/written mismatch — check the queries source file.")
    print("done.")


if __name__ == "__main__":
    main()
