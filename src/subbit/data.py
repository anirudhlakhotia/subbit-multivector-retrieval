"""Data loading for MS MARCO and BEIR datasets.

Pipeline:
  1. encode_corpus.py generates cached ColBERT embeddings as .pt files
  2. This module loads those cached embeddings for training/evaluation
  3. No ColBERT inference happens during training — only R is trained

Expected cache format:
  embeddings_dir/
    queries/
      {qid}.pt         → (m, 128) float tensor
    passages/
      {pid}.pt          → (n, 128) float tensor
    OR (more efficient):
    query_embeddings.pt → dict[qid → (m, 128)]
    doc_embeddings.pt → dict[pid → (n, 128)]
    # Legacy compatibility:
    passage_embeddings.pt → dict[pid → (n, 128)]
    
  For large-scale storage:
    passage_embeddings.bin → memory-mapped float32 array
    passage_offsets.pt     → dict[pid → (start_idx, length)]
"""
from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)

QUERY_EMBEDDING_FILENAMES = ("query_embeddings.pt",)
DOC_EMBEDDING_FILENAMES = ("doc_embeddings.pt", "passage_embeddings.pt")


def resolve_embedding_cache_path(
    embeddings_dir: str | Path,
    kind: str,
    *,
    required: bool = True,
) -> Path:
    """Resolve the on-disk cache file for query/doc embedding dicts.

    Canonical naming is `query_embeddings.pt` / `doc_embeddings.pt`, but we keep
    compatibility with older `passage_embeddings.pt` caches while the repo is
    being migrated.
    """
    embeddings_dir = Path(embeddings_dir)
    if kind == "query":
        candidates = QUERY_EMBEDDING_FILENAMES
    elif kind == "doc":
        candidates = DOC_EMBEDDING_FILENAMES
    else:
        raise ValueError(f"Unknown embedding kind: {kind}")

    for filename in candidates:
        path = embeddings_dir / filename
        if path.exists():
            return path

    canonical = embeddings_dir / candidates[0]
    if required:
        raise FileNotFoundError(
            f"No {kind} embedding cache found in {embeddings_dir}. "
            f"Tried: {', '.join(candidates)}"
        )
    return canonical


def load_embedding_cache(
    embeddings_dir: str | Path,
    kind: str,
    *,
    map_location: str | torch.device | None = "cpu",
    mmap: bool | None = None,
) -> dict:
    """Load a query/doc embedding dict with canonical/legacy filename support."""
    path = resolve_embedding_cache_path(embeddings_dir, kind)
    load_kwargs = {"weights_only": False}
    if map_location is not None:
        load_kwargs["map_location"] = map_location
    if mmap is not None:
        load_kwargs["mmap"] = mmap

    logger.info(f"Loading {kind} embeddings from {path}")
    return torch.load(path, **load_kwargs)


def load_embedding_caches(
    embeddings_dir: str | Path,
    *,
    map_location: str | torch.device | None = "cpu",
    mmap: bool | None = None,
) -> tuple[dict, dict]:
    """Load query/doc embedding dict caches from a directory."""
    query_embs = load_embedding_cache(
        embeddings_dir,
        "query",
        map_location=map_location,
        mmap=mmap,
    )
    doc_embs = load_embedding_cache(
        embeddings_dir,
        "doc",
        map_location=map_location,
        mmap=mmap,
    )
    return query_embs, doc_embs


class EmbeddingStore:
    """Efficient storage and retrieval of precomputed embeddings.

    Supports two modes:
      - dict mode: All embeddings in a single dict (fits in RAM)
      - mmap mode: Memory-mapped numpy array + offset index (large corpora)
    """

    def __init__(self, path: str | Path, mode: str = "auto"):
        self.path = Path(path)
        self.mode = mode
        self._data = None
        self._offsets = None
        self._mmap = None
        self._dim = None

        if mode == "auto":
            if (self.path / "embeddings.bin").exists():
                self.mode = "mmap"
            elif self.path.suffix == ".pt":
                self.mode = "dict"
            else:
                self.mode = "dir"

    def load(self) -> None:
        """Load embedding data into memory (dict mode) or open mmap."""
        if self.mode == "dict":
            logger.info(f"Loading embeddings from {self.path}")
            self._data = torch.load(self.path, map_location="cpu", weights_only=False)
            # Infer dimension from first embedding
            if self._data:
                first_val = next(iter(self._data.values()))
                self._dim = first_val.shape[-1]
            logger.info(f"Loaded {len(self._data)} embeddings ({self._dim}d)")

        elif self.mode == "mmap":
            offsets_path = self.path / "offsets.pt"
            mmap_path = self.path / "embeddings.bin"
            meta_path = self.path / "meta.json"

            with open(meta_path) as f:
                meta = json.load(f)
            self._dim = meta["dim"]
            total_tokens = meta["total_tokens"]

            self._offsets = torch.load(offsets_path, map_location="cpu", weights_only=False)
            self._mmap = np.memmap(
                mmap_path, dtype=np.float32, mode="r", shape=(total_tokens, self._dim)
            )
            logger.info(
                f"Memory-mapped {total_tokens} tokens ({self._dim}d) from {mmap_path}"
            )

        elif self.mode == "dir":
            logger.info(f"Using per-file embedding directory: {self.path}")

    @property
    def dim(self) -> int:
        """Embedding dimension. Available after load()."""
        if self._dim is not None:
            return self._dim
        # Fallback for dir mode: infer from first file
        if self.mode == "dir":
            pts = list(self.path.glob("*.pt"))
            if pts:
                emb = torch.load(pts[0], map_location="cpu", weights_only=False)
                self._dim = emb.shape[-1]
                return self._dim
        raise RuntimeError("Embedding dimension unknown — call load() first.")

    def get(self, id: str | int) -> torch.Tensor:
        """Retrieve embeddings for a given ID.

        Returns:
            (num_tokens, dim) float tensor.
        """
        if self.mode == "dict":
            return self._data[id]

        elif self.mode == "mmap":
            start, length = self._offsets[id]
            arr = self._mmap[start : start + length].copy()
            return torch.from_numpy(arr)

        elif self.mode == "dir":
            return torch.load(self.path / f"{id}.pt", map_location="cpu", weights_only=False)

    def get_all_ids(self) -> list:
        """Return all available IDs."""
        if self.mode == "dict":
            return list(self._data.keys())
        elif self.mode == "mmap":
            return list(self._offsets.keys())
        elif self.mode == "dir":
            return [p.stem for p in self.path.glob("*.pt")]

    def sample_embeddings(
        self,
        n: int,
        seed: int = 42,
        ids: Optional[list[str] | set[str] | tuple[str, ...]] = None,
    ) -> torch.Tensor:
        """Sample n random token embeddings (for PCA initialization).

        Returns:
            (n, dim) float tensor of randomly sampled token embeddings.
        """
        rng = np.random.RandomState(seed)

        if ids is not None:
            candidate_ids = list(ids)
            if not candidate_ids:
                raise ValueError("Cannot sample embeddings from an empty ID subset.")
            # Batched: pick a chunk of doc IDs at once instead of one per
            # iteration. With ~70 tokens/doc, ceil(n / 70) docs is a safe
            # over-estimate; we draw 2× to be safe against short docs.
            n_docs_needed = max(1, (n // 32) * 2)
            chosen = rng.choice(np.asarray(candidate_ids), size=n_docs_needed, replace=True)
            sampled = [self.get(did) for did in chosen]
            concatenated = torch.cat(sampled, dim=0)
            if len(concatenated) < n:
                # Rare: top up with another batch.
                extra = rng.choice(np.asarray(candidate_ids), size=n_docs_needed, replace=True)
                concatenated = torch.cat([concatenated] + [self.get(d) for d in extra], dim=0)
            indices = rng.choice(len(concatenated), size=min(n, len(concatenated)), replace=False)
            return concatenated[indices]

        if self.mode == "dict":
            all_ids = np.asarray(list(self._data.keys()))
            n_docs_needed = max(1, (n // 32) * 2)
            chosen = rng.choice(all_ids, size=n_docs_needed, replace=True)
            sampled = [self._data[did] for did in chosen]
            concatenated = torch.cat(sampled, dim=0)
            if len(concatenated) < n:
                extra = rng.choice(all_ids, size=n_docs_needed, replace=True)
                concatenated = torch.cat([concatenated] + [self._data[d] for d in extra], dim=0)
            indices = rng.choice(len(concatenated), size=min(n, len(concatenated)), replace=False)
            return concatenated[indices]

        elif self.mode == "mmap":
            total = self._mmap.shape[0]
            indices = rng.choice(total, size=min(n, total), replace=False)
            return torch.from_numpy(self._mmap[indices].copy())

        elif self.mode == "dir":
            all_files = list(self.path.glob("*.pt"))
            sampled = []
            rng.shuffle(all_files)
            for f in all_files:
                if len(sampled) >= n:
                    break
                embs = torch.load(f, map_location="cpu", weights_only=False)
                sampled.append(embs)
            concatenated = torch.cat(sampled, dim=0)
            indices = rng.choice(len(concatenated), size=min(n, len(concatenated)), replace=False)
            return concatenated[indices]


class TriplesDataset(Dataset):
    """Dataset of (query, positive_doc, negative_doc) embedding triples.

    Each item returns a dict with:
        - query_embs: (m, 128) float tensor
        - pos_doc_embs: (n, 128) float tensor
        - neg_doc_embs: (n', 128) float tensor
    """

    def __init__(
        self,
        triples_path: str | Path,
        query_store: EmbeddingStore,
        doc_store: EmbeddingStore,
        max_query_tokens: int = 32,
        max_doc_tokens: int = 180,
        max_samples: int = -1,
    ):
        self.triples_path = Path(triples_path)
        self.query_store = query_store
        self.doc_store = doc_store
        self.max_query_tokens = max_query_tokens
        self.max_doc_tokens = max_doc_tokens

        # Load triples
        logger.info(f"Loading triples from {self.triples_path}")
        self.triples = self._load_triples(max_samples)
        logger.info(f"Loaded {len(self.triples)} triples")

    def _load_triples(self, max_samples: int) -> list[tuple]:
        """Load (qid, pos_pid, neg_pid) triples from file."""
        triples = []

        if self.triples_path.suffix == ".pt":
            raw = torch.load(self.triples_path, map_location="cpu", weights_only=False)
            for item in raw:
                triples.append(tuple(item))
                if 0 < max_samples <= len(triples):
                    break

        elif self.triples_path.suffix in (".tsv", ".txt"):
            with open(self.triples_path) as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 3:
                        triples.append((parts[0], parts[1], parts[2]))
                    if 0 < max_samples <= len(triples):
                        break

        elif self.triples_path.suffix == ".jsonl":
            with open(self.triples_path) as f:
                for line in f:
                    item = json.loads(line)
                    triples.append((item["qid"], item["pos_pid"], item["neg_pid"]))
                    if 0 < max_samples <= len(triples):
                        break

        return triples

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, idx: int) -> dict:
        qid, pos_pid, neg_pid = self.triples[idx]

        query_embs = self.query_store.get(qid)
        pos_doc_embs = self.doc_store.get(pos_pid)
        neg_doc_embs = self.doc_store.get(neg_pid)

        # Truncate to max lengths
        query_embs = query_embs[: self.max_query_tokens]
        pos_doc_embs = pos_doc_embs[: self.max_doc_tokens]
        neg_doc_embs = neg_doc_embs[: self.max_doc_tokens]

        return {
            "query_embs": query_embs,
            "pos_doc_embs": pos_doc_embs,
            "neg_doc_embs": neg_doc_embs,
            "qid": qid,
            "pos_pid": pos_pid,
            "neg_pid": neg_pid,
        }


def collate_triples(batch: list[dict]) -> dict:
    """Collate a batch of triples with padding.

    Pads query, positive doc, and negative doc embeddings to the max
    length in the batch and creates attention masks.
    """
    # Find max lengths in batch
    max_q_len = max(item["query_embs"].shape[0] for item in batch)
    max_pos_len = max(item["pos_doc_embs"].shape[0] for item in batch)
    max_neg_len = max(item["neg_doc_embs"].shape[0] for item in batch)
    dim = batch[0]["query_embs"].shape[1]

    batch_size = len(batch)

    # Initialize padded tensors
    q_embs = torch.zeros(batch_size, max_q_len, dim)
    pos_embs = torch.zeros(batch_size, max_pos_len, dim)
    neg_embs = torch.zeros(batch_size, max_neg_len, dim)
    q_mask = torch.zeros(batch_size, max_q_len, dtype=torch.bool)
    pos_mask = torch.zeros(batch_size, max_pos_len, dtype=torch.bool)
    neg_mask = torch.zeros(batch_size, max_neg_len, dtype=torch.bool)

    qids, pos_pids, neg_pids = [], [], []

    for i, item in enumerate(batch):
        q_len = item["query_embs"].shape[0]
        pos_len = item["pos_doc_embs"].shape[0]
        neg_len = item["neg_doc_embs"].shape[0]

        q_embs[i, :q_len] = item["query_embs"]
        pos_embs[i, :pos_len] = item["pos_doc_embs"]
        neg_embs[i, :neg_len] = item["neg_doc_embs"]

        q_mask[i, :q_len] = True
        pos_mask[i, :pos_len] = True
        neg_mask[i, :neg_len] = True

        qids.append(item["qid"])
        pos_pids.append(item["pos_pid"])
        neg_pids.append(item["neg_pid"])

    return {
        "query_embs": q_embs,
        "pos_doc_embs": pos_embs,
        "neg_doc_embs": neg_embs,
        "query_mask": q_mask,
        "pos_doc_mask": pos_mask,
        "neg_doc_mask": neg_mask,
        "qids": qids,
        "pos_pids": pos_pids,
        "neg_pids": neg_pids,
    }


class EvalDataset(Dataset):
    """Dataset for evaluation: queries with their candidate documents.

    Each item returns query embeddings and a list of candidate doc embeddings.
    """

    def __init__(
        self,
        qrels: dict[str, dict[str, int]],
        query_store: EmbeddingStore,
        doc_store: EmbeddingStore,
        max_query_tokens: int = 32,
        max_doc_tokens: int = 180,
    ):
        self.qrels = qrels
        self.query_store = query_store
        self.doc_store = doc_store
        self.max_query_tokens = max_query_tokens
        self.max_doc_tokens = max_doc_tokens
        self.query_ids = list(qrels.keys())

    def __len__(self) -> int:
        return len(self.query_ids)

    def __getitem__(self, idx: int) -> dict:
        qid = self.query_ids[idx]
        query_embs = self.query_store.get(qid)[: self.max_query_tokens]

        return {
            "qid": qid,
            "query_embs": query_embs,
            "relevant_pids": self.qrels[qid],
        }


def load_qrels(path: str | Path) -> dict[str, dict[str, int]]:
    """Load query relevance judgments from TREC-format or TSV file.

    Returns:
        dict mapping qid → {pid → relevance_score}
    """
    qrels = {}
    path = Path(path)

    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 4:
                # TREC format: qid 0 pid rel
                qid, _, pid, rel = parts
            elif len(parts) == 3:
                # Simple format: qid pid rel
                qid, pid, rel = parts
            elif len(parts) == 2:
                # Minimal: qid pid (assume relevant)
                qid, pid = parts
                rel = "1"
            else:
                continue

            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][pid] = int(rel)

    logger.info(f"Loaded qrels: {len(qrels)} queries from {path}")
    return qrels


def create_debug_data(
    output_dir: str | Path,
    num_queries: int = 100,
    num_passages: int = 10000,
    dim: int = 128,
    max_query_tokens: int = 32,
    max_doc_tokens: int = 180,
    seed: int = 42,
    with_semantic_signal: bool = True,
) -> None:
    """Create synthetic debug data for fast development iteration.

    Args:
        output_dir: Directory to save debug data.
        num_queries: Number of synthetic queries.
        num_passages: Number of synthetic passages.
        dim: Embedding dimension.
        max_query_tokens: Max tokens per query.
        max_doc_tokens: Max tokens per passage.
        seed: Random seed.
        with_semantic_signal: If True, relevant docs share latent topics with
            queries, giving meaningful MaxSim scores for benchmarking. If False,
            generates pure random embeddings (faster, for pipeline testing only).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    logger.info(f"Creating debug data: {num_queries} queries, {num_passages} passages")

    if with_semantic_signal:
        # Generate latent topic vectors for semantic signal
        num_topics = num_queries
        topic_vectors = torch.randn(num_topics, dim)
        topic_vectors = torch.nn.functional.normalize(topic_vectors, dim=-1)
        noise_scale = 0.3

        # Generate query embeddings with topic signal
        query_embs = {}
        for i in range(num_queries):
            topic_vec = topic_vectors[i % num_topics]
            n_tokens = rng.randint(8, max_query_tokens + 1)
            noise = torch.randn(n_tokens, dim) * noise_scale
            embs = topic_vec.unsqueeze(0).expand(n_tokens, -1) + noise
            embs = torch.nn.functional.normalize(embs, dim=-1)
            query_embs[str(i)] = embs

        # Generate passage embeddings - first num_queries share topics with queries
        passage_embs = {}
        for i in range(num_passages):
            n_tokens = rng.randint(20, max_doc_tokens + 1)
            if i < num_queries:
                topic_vec = topic_vectors[i]
                noise = torch.randn(n_tokens, dim) * noise_scale
                embs = topic_vec.unsqueeze(0).expand(n_tokens, -1) + noise
            else:
                embs = torch.randn(n_tokens, dim)
            embs = torch.nn.functional.normalize(embs, dim=-1)
            passage_embs[str(i)] = embs

        # Triples: query i's relevant doc is passage i
        triples = []
        passage_ids = list(range(num_passages))
        for qid in range(num_queries):
            pos_pid = qid
            neg_pid = rng.choice([p for p in passage_ids if p != pos_pid])
            triples.append((str(qid), str(pos_pid), str(neg_pid)))
    else:
        # Pure random embeddings (original behavior for pipeline testing)
        query_embs = {}
        for i in range(num_queries):
            n_tokens = rng.randint(8, max_query_tokens + 1)
            embs = torch.randn(n_tokens, dim)
            embs = torch.nn.functional.normalize(embs, dim=-1)
            query_embs[str(i)] = embs

        passage_embs = {}
        for i in range(num_passages):
            n_tokens = rng.randint(20, max_doc_tokens + 1)
            embs = torch.randn(n_tokens, dim)
            embs = torch.nn.functional.normalize(embs, dim=-1)
            passage_embs[str(i)] = embs

        triples = []
        passage_ids = list(range(num_passages))
        for qid in range(num_queries):
            pos_pid = rng.choice(passage_ids)
            neg_pid = rng.choice(passage_ids)
            while neg_pid == pos_pid:
                neg_pid = rng.choice(passage_ids)
            triples.append((str(qid), str(pos_pid), str(neg_pid)))

    torch.save(query_embs, output_dir / "query_embeddings.pt")
    torch.save(passage_embs, output_dir / "doc_embeddings.pt")

    # Save triples
    with open(output_dir / "triples.tsv", "w") as f:
        for qid, pos, neg in triples:
            f.write(f"{qid}\t{pos}\t{neg}\n")

    # Save qrels
    qrels = {}
    for qid, pos_pid, _ in triples:
        if qid not in qrels:
            qrels[qid] = {}
        qrels[qid][pos_pid] = 1

    with open(output_dir / "qrels.tsv", "w") as f:
        for qid, pids in qrels.items():
            for pid, rel in pids.items():
                f.write(f"{qid}\t0\t{pid}\t{rel}\n")

    logger.info(f"Debug data saved to {output_dir}")
    logger.info(f"  {len(query_embs)} queries, {len(passage_embs)} passages, {len(triples)} triples")


class BEIRDataset(Dataset):
    """Dataset wrapper for BEIR benchmark datasets.

    Handles loading and preprocessing of BEIR datasets for cross-domain
    evaluation. Supports both pre-computed embeddings and on-the-fly encoding.

    Expected structure:
        beir_dir/{dataset_name}/
            corpus.jsonl      → {"_id": str, "title": str, "text": str}
            queries.jsonl     → {"_id": str, "text": str}
            qrels/test.tsv    → query_id  corpus_id  score

        Or with pre-computed embeddings:
        embeddings_dir/{dataset_name}/
            doc_embeddings.pt   → dict[doc_id → (n, dim) tensor]
            query_embeddings.pt → dict[query_id → (m, dim) tensor]
    """

    def __init__(
        self,
        dataset_name: str,
        beir_dir: str | Path = "data/beir",
        embeddings_dir: str | Path | None = None,
        split: str = "test",
        max_query_tokens: int = 32,
        max_doc_tokens: int = 180,
    ):
        self.dataset_name = dataset_name
        self.beir_dir = Path(beir_dir)
        self.embeddings_dir = Path(embeddings_dir) if embeddings_dir else None
        self.split = split
        self.max_query_tokens = max_query_tokens
        self.max_doc_tokens = max_doc_tokens

        self.corpus = {}
        self.queries = {}
        self.qrels = {}
        self.query_ids = []
        self.doc_ids = []

        self._doc_embeddings = None
        self._query_embeddings = None

        self._load_dataset()

    def _load_dataset(self) -> None:
        """Load BEIR dataset from disk."""
        dataset_path = self.beir_dir / self.dataset_name

        if not dataset_path.exists():
            self._download_dataset()

        corpus_path = dataset_path / "corpus.jsonl"
        queries_path = dataset_path / "queries.jsonl"
        qrels_path = dataset_path / "qrels" / f"{self.split}.tsv"

        if corpus_path.exists():
            logger.info(f"Loading BEIR corpus from {corpus_path}")
            with open(corpus_path) as f:
                for line in f:
                    doc = json.loads(line)
                    self.corpus[doc["_id"]] = {
                        "title": doc.get("title", ""),
                        "text": doc.get("text", ""),
                    }
            self.doc_ids = list(self.corpus.keys())
            logger.info(f"Loaded {len(self.corpus)} documents")

        if queries_path.exists():
            logger.info(f"Loading BEIR queries from {queries_path}")
            with open(queries_path) as f:
                for line in f:
                    query = json.loads(line)
                    self.queries[query["_id"]] = query.get("text", "")
            logger.info(f"Loaded {len(self.queries)} queries")

        if qrels_path.exists():
            logger.info(f"Loading qrels from {qrels_path}")
            with open(qrels_path) as f:
                next(f)
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 3:
                        qid, did, rel = parts[0], parts[1], int(parts[2])
                        if qid not in self.qrels:
                            self.qrels[qid] = {}
                        self.qrels[qid][did] = rel
            self.query_ids = list(self.qrels.keys())
            logger.info(f"Loaded qrels for {len(self.query_ids)} queries")

        if self.embeddings_dir:
            self._load_embeddings()

    def _download_dataset(self) -> None:
        """Download BEIR dataset using the beir library."""
        try:
            from beir import util
            url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{self.dataset_name}.zip"
            util.download_and_unzip(url, str(self.beir_dir))
            logger.info(f"Downloaded {self.dataset_name} to {self.beir_dir}")
        except ImportError:
            raise ImportError("Install beir package: pip install beir")

    def _load_embeddings(self) -> None:
        """Load pre-computed embeddings if available."""
        emb_path = self.embeddings_dir / self.dataset_name

        try:
            self._doc_embeddings = load_embedding_cache(emb_path, "doc", map_location="cpu")
        except FileNotFoundError:
            pass

        try:
            self._query_embeddings = load_embedding_cache(emb_path, "query", map_location="cpu")
        except FileNotFoundError:
            pass

    def get_doc_text(self, doc_id: str) -> str:
        """Get document text for encoding."""
        doc = self.corpus.get(doc_id, {})
        title = doc.get("title", "")
        text = doc.get("text", "")
        return f"{title} {text}".strip() if title else text

    def get_query_text(self, query_id: str) -> str:
        """Get query text for encoding."""
        return self.queries.get(query_id, "")

    def get_doc_embedding(self, doc_id: str) -> torch.Tensor | None:
        """Get pre-computed document embedding."""
        if self._doc_embeddings is not None:
            return self._doc_embeddings.get(doc_id)
        return None

    def get_query_embedding(self, query_id: str) -> torch.Tensor | None:
        """Get pre-computed query embedding."""
        if self._query_embeddings is not None:
            return self._query_embeddings.get(query_id)
        return None

    def __len__(self) -> int:
        return len(self.query_ids)

    def __getitem__(self, idx: int) -> dict:
        qid = self.query_ids[idx]
        return {
            "query_id": qid,
            "query_text": self.get_query_text(qid),
            "query_embedding": self.get_query_embedding(qid),
            "relevant_docs": self.qrels.get(qid, {}),
        }

    def get_all_doc_ids(self) -> list[str]:
        """Return all document IDs."""
        return self.doc_ids

    def get_qrels(self) -> dict[str, dict[str, int]]:
        """Return qrels dict."""
        return self.qrels
