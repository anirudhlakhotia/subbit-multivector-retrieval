from __future__ import annotations

"""Encode MS MARCO corpus with frozen ColBERTv2.

Produces cached embedding files that the training pipeline consumes.
ColBERT is only run here — training only updates R on cached embeddings.

Output:
  embeddings_dir/
    query_embeddings.pt     → dict[qid → (m, 128) tensor]
    doc_embeddings.pt       → dict[pid → (n, 128) tensor]
    # Legacy loaders may still accept:
    passage_embeddings.pt   → dict[pid → (n, 128) tensor]
    
  For large-scale (memory-mapped):
    embeddings_dir/
      passages/
        embeddings.bin      → (total_tokens, 128) float32 mmap
        offsets.pt          → dict[pid → (start, length)]
        meta.json           → {"dim": 128, "total_tokens": N}
      queries/
        query_embeddings.pt → dict[qid → (m, 128)]

Usage:
  python training/encode_corpus.py --config configs/base.yaml
  python training/encode_corpus.py --config configs/base.yaml --debug
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from omegaconf import OmegaConf

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.subbit.utils import setup_logging, get_device, seed_everything, ensure_dir

logger = logging.getLogger(__name__)


def _build_token_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer,
    *,
    skiplist_ids: set[int] | None = None,
    remove_special_tokens: bool = True,
) -> torch.Tensor:
    """Build ColBERT-style valid-token mask."""
    mask = attention_mask.bool()

    if tokenizer.pad_token_id is not None:
        mask = mask & (input_ids != tokenizer.pad_token_id)

    if skiplist_ids:
        skip = torch.tensor(sorted(skiplist_ids), dtype=input_ids.dtype, device=input_ids.device)
        mask = mask & ~torch.isin(input_ids, skip)

    if remove_special_tokens:
        if tokenizer.cls_token_id is not None:
            mask = mask & (input_ids != tokenizer.cls_token_id)
        if tokenizer.sep_token_id is not None:
            mask = mask & (input_ids != tokenizer.sep_token_id)

    return mask


def _validate_embedding_tensor(emb: torch.Tensor, item_id: str, kind: str,
                                expected_dim: int | None = None) -> None:
    """Fail fast if encoder does not emit the expected token-embedding shape."""
    if emb.ndim != 2:
        raise RuntimeError(f"{kind} embedding for {item_id} must be rank-2, got shape={tuple(emb.shape)}")
    if expected_dim is not None and emb.shape[1] != expected_dim:
        raise RuntimeError(
            f"{kind} embedding for {item_id} must have dim={expected_dim}, got shape={tuple(emb.shape)}"
        )


def load_colbert_model(
    device: torch.device,
    query_maxlen: int = 32,
    doc_maxlen: int = 180,
) -> tuple:
    """Load frozen ColBERTv2 model for encoding.

    `colbert-ai` is required. We prefer a cached local snapshot to avoid Hub
    retries in offline environments, but still fall back to the repo id if
    the snapshot is not already present.

    Args:
        device: Torch device.
        query_maxlen: Maximum query token length.
        doc_maxlen: Maximum document token length.
    """
    try:
        import os
        from huggingface_hub import snapshot_download
        from colbert.infra import ColBERTConfig
        from colbert.modeling.checkpoint import Checkpoint
        import colbert.modeling.colbert as colbert_module

        os.environ["COLBERT_LOAD_TORCH_EXTENSION_VERBOSE"] = "False"
        original_try_load = colbert_module.ColBERT.try_load_torch_extensions
        colbert_module.ColBERT.try_load_torch_extensions = classmethod(lambda cls, use_gpu: None)

        model_name = "colbert-ir/colbertv2.0"
        model_source = model_name
        try:
            model_source = snapshot_download(model_name, local_files_only=True)
            logger.info("Loaded ColBERTv2 from cached snapshot: %s", model_source)
        except Exception:
            logger.info("No cached ColBERTv2 snapshot found; loading from %s", model_name)

        config = ColBERTConfig(
            doc_maxlen=doc_maxlen,
            query_maxlen=query_maxlen,
        )
        try:
            checkpoint = Checkpoint(model_source, colbert_config=config)
            checkpoint = checkpoint.to(device)
        finally:
            colbert_module.ColBERT.try_load_torch_extensions = original_try_load

        logger.info("Loaded ColBERTv2 checkpoint")
        return checkpoint, config
    except ImportError as exc:
        raise RuntimeError(
            "`training/encode_corpus.py` requires `colbert-ai`. "
            "Install it with `pip install colbert-ai` before encoding embeddings."
        ) from exc


def encode_passages_colbert(checkpoint, passages: dict, device: torch.device, batch_size: int = 64):
    """Encode passages using ColBERT checkpoint.

    Args:
        checkpoint: ColBERT Checkpoint object.
        passages: dict mapping pid → passage_text.
        device: Device for computation.
        batch_size: Encoding batch size.

    Returns:
        dict mapping pid → (n_tokens, 128) tensor.
    """
    embeddings = {}
    pids = list(passages.keys())

    for i in tqdm(range(0, len(pids), batch_size), desc="Encoding passages"):
        batch_pids = pids[i : i + batch_size]
        batch_texts = [passages[pid] for pid in batch_pids]

        with torch.no_grad():
            input_ids, attention_mask = checkpoint.doc_tokenizer.tensorize(batch_texts)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            embs = checkpoint.doc(input_ids, attention_mask, keep_dims=True, to_cpu=False)

            skiplist_ids = set()
            if hasattr(checkpoint, "skiplist"):
                skiplist_ids = {int(x) for x in checkpoint.skiplist if isinstance(x, int)}

            token_mask = _build_token_mask(
                input_ids=input_ids,
                attention_mask=attention_mask,
                tokenizer=checkpoint.raw_tokenizer,
                skiplist_ids=skiplist_ids,
                remove_special_tokens=True,
            )

            for j, pid in enumerate(batch_pids):
                emb = embs[j][token_mask[j]]
                if emb.numel() == 0:
                    raise RuntimeError(f"Empty doc embedding after masking for pid={pid}")
                _validate_embedding_tensor(emb, pid, "doc")
                embeddings[pid] = emb.cpu()

    return embeddings


def encode_queries_colbert(checkpoint, queries: dict, device: torch.device, batch_size: int = 64):
    """Encode queries using ColBERT checkpoint.

    Args:
        checkpoint: ColBERT Checkpoint object.
        queries: dict mapping qid → query_text.
        device: Device for computation.
        batch_size: Encoding batch size.

    Returns:
        dict mapping qid → (m_tokens, 128) tensor.
    """
    embeddings = {}
    qids = list(queries.keys())

    for i in tqdm(range(0, len(qids), batch_size), desc="Encoding queries"):
        batch_qids = qids[i : i + batch_size]
        batch_texts = [queries[qid] for qid in batch_qids]

        with torch.no_grad():
            input_ids, attention_mask = checkpoint.query_tokenizer.tensorize(batch_texts)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            embs = checkpoint.query(input_ids, attention_mask, to_cpu=False)

            token_mask = _build_token_mask(
                input_ids=input_ids,
                attention_mask=attention_mask,
                tokenizer=checkpoint.raw_tokenizer,
                skiplist_ids=None,
                remove_special_tokens=True,
            )

            for j, qid in enumerate(batch_qids):
                emb = embs[j][token_mask[j]]
                if emb.numel() == 0:
                    raise RuntimeError(f"Empty query embedding after masking for qid={qid}")
                _validate_embedding_tensor(emb, qid, "query")
                embeddings[qid] = emb.cpu()

    return embeddings


def load_msmarco_passages(data_dir: Path, max_passages: int = -1, include_qrel_docs: bool = True) -> dict:
    """Load MS MARCO passages from file.
    
    Args:
        data_dir: Data directory (unused for ir_datasets)
        max_passages: Maximum passages to load (-1 for all)
        include_qrel_docs: If True, ensure all docs referenced in qrels are included
    """
    passages = {}

    # Try ir_datasets first
    try:
        import ir_datasets
        
        # First, get all doc IDs referenced in qrels (so we don't miss relevant docs)
        # PRIORITIZE dev qrels (for evaluation), then fill with train qrels
        required_doc_ids = set()
        if include_qrel_docs and max_passages > 0:
            # Dev qrels first - these are needed for evaluation
            dev_ds = ir_datasets.load("msmarco-passage/dev/small")
            for qrel in dev_ds.qrels_iter():
                required_doc_ids.add(qrel.doc_id)
            logger.info(f"Found {len(required_doc_ids)} docs from dev qrels")
            
            # Fill remaining with train qrels
            if len(required_doc_ids) < max_passages:
                train_ds = ir_datasets.load("msmarco-passage/train")
                for qrel in train_ds.qrels_iter():
                    required_doc_ids.add(qrel.doc_id)
                    if len(required_doc_ids) >= max_passages:
                        break
            logger.info(f"Found {len(required_doc_ids)} docs total from dev+train qrels")
        
        # Load passages - need to use docs_store for random access
        dataset = ir_datasets.load("msmarco-passage")
        docs_store = dataset.docs_store()
        
        # First load all required docs
        if required_doc_ids:
            logger.info(f"Loading {len(required_doc_ids)} qrel-referenced docs...")
            for doc_id in required_doc_ids:
                try:
                    doc = docs_store.get(doc_id)
                    passages[doc.doc_id] = doc.text
                except:
                    pass  # Some docs might not exist
            logger.info(f"Loaded {len(passages)} qrel-referenced passages")
        
        # Fill remaining slots with sequential docs if needed
        if max_passages > 0 and len(passages) < max_passages:
            remaining = max_passages - len(passages)
            logger.info(f"Loading {remaining} additional passages...")
            count = 0
            for doc in dataset.docs_iter():
                if doc.doc_id not in passages:
                    passages[doc.doc_id] = doc.text
                    count += 1
                    if count >= remaining:
                        break
        elif max_passages <= 0:
            for doc in dataset.docs_iter():
                passages[doc.doc_id] = doc.text
        
        logger.info(f"Loaded {len(passages)} passages total via ir_datasets")
        return passages
    except ImportError:
        pass

    # Fallback to file
    collection_path = data_dir / "collection.tsv"
    if collection_path.exists():
        with open(collection_path) as f:
            for line in f:
                pid, text = line.strip().split("\t", 1)
                passages[pid] = text
                if 0 < max_passages <= len(passages):
                    break
        logger.info(f"Loaded {len(passages)} passages from {collection_path}")
    else:
        logger.error(f"No passage collection found at {collection_path}")

    return passages


def load_msmarco_queries(
    data_dir: Path,
    max_queries: int = -1,
    encoded_pids: set[str] | None = None,
) -> dict:
    """Load MS MARCO dev AND train queries.

    Always loads all dev/small queries (needed for evaluation).
    Then loads train queries whose positive docs are in *encoded_pids*
    (needed for training triples). If *encoded_pids* is None, only dev
    queries are loaded (backward-compatible behaviour).

    Args:
        data_dir: Data directory (unused for ir_datasets path).
        max_queries: Max total queries to load (-1 for all).
        encoded_pids: Set of encoded passage IDs. When provided, train
            queries that have at least one positive in this set are included.
    """
    queries = {}

    try:
        import ir_datasets

        # Always load dev queries (evaluation)
        dev_ds = ir_datasets.load("msmarco-passage/dev/small")
        for query in dev_ds.queries_iter():
            queries[query.query_id] = query.text
        logger.info(f"Loaded {len(queries)} dev queries via ir_datasets")

        # Load train queries whose positives are in the encoded corpus
        if encoded_pids is not None:
            train_ds = ir_datasets.load("msmarco-passage/train")
            # Find which train queries have positives in encoded set
            train_qids_with_pos = set()
            for qrel in train_ds.qrels_iter():
                if qrel.relevance > 0 and qrel.doc_id in encoded_pids:
                    train_qids_with_pos.add(qrel.query_id)

            train_count = 0
            for query in train_ds.queries_iter():
                if query.query_id in train_qids_with_pos:
                    queries[query.query_id] = query.text
                    train_count += 1
                if 0 < max_queries <= len(queries):
                    break
            logger.info(f"Loaded {train_count} train queries with positives in encoded docs")
        elif 0 < max_queries < len(queries):
            # Trim dev queries if over limit and no train loading
            trimmed = dict(list(queries.items())[:max_queries])
            queries = trimmed

        logger.info(f"Total queries: {len(queries)}")
        return queries
    except ImportError:
        pass

    queries_path = data_dir / "queries.dev.small.tsv"
    if queries_path.exists():
        with open(queries_path) as f:
            for line in f:
                qid, text = line.strip().split("\t", 1)
                queries[qid] = text
                if 0 < max_queries <= len(queries):
                    break
        logger.info(f"Loaded {len(queries)} queries from {queries_path}")
    else:
        logger.error(f"No queries found at {queries_path}")

    return queries


def save_as_mmap(embeddings: dict, output_dir: Path, dim: int = 128) -> None:
    """Save embeddings as memory-mapped file for large-scale storage.

    Creates:
      output_dir/embeddings.bin   → (total_tokens, dim) float32
      output_dir/offsets.pt       → dict[id → (start_idx, length)]
      output_dir/meta.json        → metadata
    """
    output_dir = ensure_dir(output_dir)

    # Calculate total tokens
    total_tokens = sum(emb.shape[0] for emb in embeddings.values())
    logger.info(f"Creating mmap: {total_tokens} tokens, {dim}d, "
                f"{total_tokens * dim * 4 / 1e9:.2f} GB")

    # Create memory-mapped file
    mmap = np.memmap(
        output_dir / "embeddings.bin",
        dtype=np.float32,
        mode="w+",
        shape=(total_tokens, dim),
    )

    offsets = {}
    current_idx = 0

    for id_, emb in tqdm(embeddings.items(), desc="Writing mmap"):
        length = emb.shape[0]
        mmap[current_idx : current_idx + length] = emb.numpy()
        offsets[id_] = (current_idx, length)
        current_idx += length

    mmap.flush()
    torch.save(offsets, output_dir / "offsets.pt")

    meta = {"dim": dim, "total_tokens": total_tokens, "num_items": len(embeddings)}
    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Saved mmap to {output_dir}")


def _export_triples_and_qrels(
    output_dir: Path,
    encoded_pids: set[str],
    encoded_qids: set[str],
    passage_embs: dict[str, torch.Tensor] | None = None,
    query_embs: dict[str, torch.Tensor] | None = None,
    mine_hard_negatives: bool = False,
) -> None:
    """Export training triples and dev qrels filtered to encoded IDs.

    Triples are built from train qrels + random negatives (no huge download),
    or hard negatives using mean-pooled approximation of available embeddings.
    """
    import random as _rand

    try:
        import ir_datasets
    except ImportError:
        logger.warning("ir_datasets not installed, skipping triples/qrels export")
        return

    # Export dev qrels
    qrels_path = output_dir / "qrels.tsv"
    if not qrels_path.exists():
        dev_ds = ir_datasets.load("msmarco-passage/dev/small")
        count = 0
        with open(qrels_path, "w") as f:
            for qrel in dev_ds.qrels_iter():
                if qrel.query_id in encoded_qids and qrel.doc_id in encoded_pids:
                    f.write(f"{qrel.query_id}\t0\t{qrel.doc_id}\t{qrel.relevance}\n")
                    count += 1
        logger.info(f"Exported {count} dev qrels to {qrels_path}")

    # Build training triples from train qrels + random negatives
    triples_path = output_dir / "triples.tsv"
    if not triples_path.exists():
        train_ds = ir_datasets.load("msmarco-passage/train")
        # Collect train-query positives that exist in our encoded corpus
        positives: dict[str, set[str]] = {}
        for qrel in train_ds.qrels_iter():
            if (qrel.query_id in encoded_qids and qrel.doc_id in encoded_pids
                    and qrel.relevance > 0):
                positives.setdefault(qrel.query_id, set()).add(qrel.doc_id)

        if not positives:
            logger.warning(
                "No train queries found in encoded set — no triples exported. "
                "Ensure train queries are encoded (not just dev queries)."
            )
            return

        logger.info(f"Found {len(positives)} train queries with positives in encoded set")

        pid_list = sorted(encoded_pids)
        rng = _rand.Random(42)
        negs_per_positive = 5
        count = 0
        
        # Optionally mine hard negatives
        hard_negatives_map = {}
        if mine_hard_negatives and passage_embs is not None and query_embs is not None:
            logger.info("Mining hard negatives via mean-pooling approximation...")
            device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
            
            # Mean pool valid train queries
            qid_list = sorted(positives.keys())
            q_pool = []
            for qid in qid_list:
                q_pool.append(query_embs[qid].mean(dim=0))
            q_tensor = torch.stack(q_pool).to(device)  # (M, 128)
            
            # Mean pool all docs
            d_pool = []
            for pid in pid_list:
                d_pool.append(passage_embs[pid].mean(dim=0))
            d_tensor = torch.stack(d_pool).to(device)  # (N, 128)
            
            q_batch_size = 1000
            top_k_candidates = 200 + negs_per_positive
            
            logger.info(f"Computing dense similarities for {len(qid_list)} queries vs {len(pid_list)} docs")
            for i in tqdm(range(0, len(qid_list), q_batch_size), desc="Mining hard negatives"):
                q_batch = q_tensor[i:i+q_batch_size]
                batch_scores = q_batch @ d_tensor.T  # (batch, N)
                
                topk = batch_scores.topk(min(top_k_candidates, len(pid_list)), dim=1)
                
                for j in range(q_batch.shape[0]):
                    qid = qid_list[i+j]
                    pos_pids = positives[qid]
                    top_indices = topk.indices[j].tolist()
                    
                    negs = []
                    for idx in top_indices:
                        neg_pid = pid_list[idx]
                        if neg_pid not in pos_pids:
                            negs.append(neg_pid)
                            if len(negs) == negs_per_positive:
                                break
                    hard_negatives_map[qid] = negs

        # Write triples
        with open(triples_path, "w") as f:
            for qid, pos_pids in positives.items():
                for pos_pid in pos_pids:
                    if qid in hard_negatives_map:
                        chosen_negs = hard_negatives_map[qid]
                    else:
                        chosen_negs = []
                        for _ in range(negs_per_positive):
                            neg_pid = rng.choice(pid_list)
                            while neg_pid in pos_pids:
                                neg_pid = rng.choice(pid_list)
                            chosen_negs.append(neg_pid)
                    
                    for neg_pid in chosen_negs:
                        f.write(f"{qid}\t{pos_pid}\t{neg_pid}\n")
                        count += 1
        logger.info(f"Exported {count} training triples to {triples_path}")


def main():
    parser = argparse.ArgumentParser(description="Encode MS MARCO with ColBERTv2")
    parser.add_argument(
        "--config", action="append", default=None,
        help="Config file(s). Multiple allowed; later files override earlier ones.",
    )
    parser.add_argument("--debug", action="store_true", help="Use debug subset")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-passages", type=int, default=None,
                        help="Max passages to encode (-1 for all)")
    parser.add_argument("--max-queries", type=int, default=None,
                        help="Max queries to encode (-1 for all)")
    parser.add_argument("--use-mmap", action="store_true", help="Save as memory-mapped files")
    parser.add_argument("--mine-hard-negatives", action="store_true",
                        help="Mine hard negatives via mean pooling instead of random")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (overrides training.seed in config)")
    parser.add_argument(
        "overrides", nargs="*",
        help="Config overrides in dotlist format (e.g., data.embeddings_dir=path).",
    )
    args = parser.parse_args()

    setup_logging()

    # Config stacking: multiple --config files + dotlist overrides (train.py pattern).
    config_files = list(args.config or [])
    true_overrides = []
    for o in args.overrides:
        if o.endswith((".yaml", ".yml")):
            config_files.append(o)
        else:
            true_overrides.append(o)
    if not config_files:
        config_files = ["configs/base.yaml"]
    cfg = OmegaConf.load(config_files[0])
    for extra in config_files[1:]:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(extra))
    if true_overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(true_overrides))

    if args.max_passages is not None or args.max_queries is not None:
        max_passages = args.max_passages if args.max_passages is not None else -1
        max_queries = args.max_queries if args.max_queries is not None else -1
    elif args.debug or cfg.get("debug", {}).get("enabled", False):
        max_passages = cfg.get("debug", {}).get("num_passages", 10000)
        max_queries = cfg.get("debug", {}).get("num_queries", 100)
        logger.info(f"DEBUG MODE: {max_passages} passages, {max_queries} queries")
    else:
        max_passages = -1
        max_queries = -1
    logger.info(f"Encoding: max_passages={max_passages}, max_queries={max_queries}")

    device = get_device(cfg.get("hardware", {}).get("device", "auto"))
    seed = args.seed if args.seed is not None else cfg.get("training", {}).get("seed", 42)
    seed_everything(seed)

    output_dir = Path(args.output_dir or cfg.get("data", {}).get("embeddings_dir", "data/embeddings/msmarco"))
    ensure_dir(output_dir)

    # Load data
    data_dir = Path(cfg.get("data", {}).get("dataset_dir", "data/msmarco"))
    passages = load_msmarco_passages(data_dir, max_passages)
    encoded_pids = set(passages.keys()) if passages else None
    queries = load_msmarco_queries(data_dir, max_queries, encoded_pids=encoded_pids)

    if not passages and not queries:
        logger.info("No data files found. Generating debug data instead.")
        from src.subbit.data import create_debug_data
        create_debug_data(output_dir, num_queries=max_queries if max_queries > 0 else 100,
                         num_passages=max_passages if max_passages > 0 else 10000)
        logger.info("Debug data created. Skipping ColBERT encoding.")
        return

    # Load model
    query_maxlen = cfg.get("data", {}).get("max_query_tokens", 32)
    doc_maxlen = cfg.get("data", {}).get("max_doc_tokens", 180)
    checkpoint, colbert_config = load_colbert_model(device, query_maxlen=query_maxlen, doc_maxlen=doc_maxlen)

    # Encode
    if passages:
        logger.info(f"Encoding {len(passages)} passages...")
        passage_embs = encode_passages_colbert(checkpoint, passages, device, args.batch_size)

        if args.use_mmap:
            save_as_mmap(passage_embs, output_dir / "passages")
        else:
            torch.save(passage_embs, output_dir / "doc_embeddings.pt")
            logger.info(f"Saved doc embeddings to {output_dir / 'doc_embeddings.pt'}")

    if queries:
        logger.info(f"Encoding {len(queries)} queries...")
        query_embs = encode_queries_colbert(checkpoint, queries, device, args.batch_size)
        torch.save(query_embs, output_dir / "query_embeddings.pt")
        logger.info(f"Saved query embeddings to {output_dir / 'query_embeddings.pt'}")

    # Export triples and qrels so train.py can find them alongside embeddings
    encoded_pids = set(passages.keys()) if passages else set()
    encoded_qids = set(queries.keys()) if queries else set()
    _export_triples_and_qrels(
        output_dir, encoded_pids, encoded_qids,
        passage_embs=passage_embs if 'passage_embs' in locals() else None,
        query_embs=query_embs if 'query_embs' in locals() else None,
        mine_hard_negatives=args.mine_hard_negatives
    )

    logger.info("Encoding complete!")


if __name__ == "__main__":
    main()
