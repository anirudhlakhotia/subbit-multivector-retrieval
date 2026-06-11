"""Tests for embedding cache helpers and subset-aware sampling."""

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.subbit.data import EmbeddingStore, load_embedding_caches, resolve_embedding_cache_path


def test_resolve_doc_embedding_cache_prefers_canonical_name(tmp_path):
    torch.save({}, tmp_path / "doc_embeddings.pt")
    torch.save({}, tmp_path / "passage_embeddings.pt")

    resolved = resolve_embedding_cache_path(tmp_path, "doc")

    assert resolved == tmp_path / "doc_embeddings.pt"


def test_resolve_doc_embedding_cache_falls_back_to_legacy_name(tmp_path):
    torch.save({}, tmp_path / "passage_embeddings.pt")

    resolved = resolve_embedding_cache_path(tmp_path, "doc")

    assert resolved == tmp_path / "passage_embeddings.pt"


def test_sample_embeddings_respects_id_subset(tmp_path):
    path = tmp_path / "doc_embeddings.pt"
    torch.save(
        {
            "d0": torch.ones(4, 2),
            "d1": torch.full((4, 2), 2.0),
        },
        path,
    )

    store = EmbeddingStore(path, mode="dict")
    store.load()

    sample = store.sample_embeddings(3, ids=["d1"])

    assert sample.shape == (3, 2)
    assert torch.all(sample == 2.0)


def test_load_embedding_caches_supports_legacy_doc_filename(tmp_path):
    query_embs = {"q0": torch.ones(1, 2)}
    doc_embs = {"d0": torch.zeros(1, 2)}
    torch.save(query_embs, tmp_path / "query_embeddings.pt")
    torch.save(doc_embs, tmp_path / "passage_embeddings.pt")

    loaded_queries, loaded_docs = load_embedding_caches(tmp_path, map_location="cpu")

    assert loaded_queries.keys() == query_embs.keys()
    assert loaded_docs.keys() == doc_embs.keys()


# ---------------------------------------------------------------------------
# EmbeddingStore.dim property
# ---------------------------------------------------------------------------


def test_embedding_store_dim_dict_mode(tmp_path):
    """dim property returns correct dimension after loading dict-mode store."""
    path = tmp_path / "embeddings.pt"
    torch.save({"d0": torch.randn(5, 64), "d1": torch.randn(3, 64)}, path)

    store = EmbeddingStore(path, mode="dict")
    store.load()

    assert store.dim == 64


def test_embedding_store_dim_inferred_from_first_entry(tmp_path):
    """dim property works even with a single entry."""
    path = tmp_path / "embeddings.pt"
    torch.save({"d0": torch.randn(2, 256)}, path)

    store = EmbeddingStore(path, mode="dict")
    store.load()

    assert store.dim == 256


def test_embedding_store_dim_raises_before_load(tmp_path):
    """dim property raises RuntimeError if called before load()."""
    path = tmp_path / "embeddings.pt"
    torch.save({"d0": torch.randn(2, 128)}, path)

    store = EmbeddingStore(path, mode="dict")

    with pytest.raises(RuntimeError, match="call load"):
        _ = store.dim


import pytest


# ---------------------------------------------------------------------------
# collate_triples Tests
# ---------------------------------------------------------------------------

from src.subbit.data import collate_triples


def test_collate_triples_pads_to_max_length():
    """Collated batch should pad all items to the same (max) length."""
    batch = [
        {
            "query_embs": torch.randn(3, 128),
            "pos_doc_embs": torch.randn(10, 128),
            "neg_doc_embs": torch.randn(8, 128),
            "qid": "q0", "pos_pid": "d0", "neg_pid": "d1",
        },
        {
            "query_embs": torch.randn(5, 128),
            "pos_doc_embs": torch.randn(6, 128),
            "neg_doc_embs": torch.randn(12, 128),
            "qid": "q1", "pos_pid": "d2", "neg_pid": "d3",
        },
    ]
    collated = collate_triples(batch)

    assert collated["query_embs"].shape == (2, 5, 128)    # max_q=5
    assert collated["pos_doc_embs"].shape == (2, 10, 128)  # max_pos=10
    assert collated["neg_doc_embs"].shape == (2, 12, 128)  # max_neg=12


def test_collate_triples_masks_are_correct():
    """Masks should be True for real tokens, False for padding."""
    batch = [
        {
            "query_embs": torch.randn(2, 128),
            "pos_doc_embs": torch.randn(4, 128),
            "neg_doc_embs": torch.randn(3, 128),
            "qid": "q0", "pos_pid": "d0", "neg_pid": "d1",
        },
        {
            "query_embs": torch.randn(5, 128),
            "pos_doc_embs": torch.randn(2, 128),
            "neg_doc_embs": torch.randn(6, 128),
            "qid": "q1", "pos_pid": "d2", "neg_pid": "d3",
        },
    ]
    collated = collate_triples(batch)

    # First item: 2 real query tokens out of 5 padded
    assert collated["query_mask"][0, :2].all()
    assert not collated["query_mask"][0, 2:].any()

    # Second item: 2 real pos tokens out of 4 padded
    assert collated["pos_doc_mask"][1, :2].all()
    assert not collated["pos_doc_mask"][1, 2:].any()


def test_collate_triples_preserves_values():
    """Original embeddings should be preserved (not overwritten by padding)."""
    q = torch.randn(3, 128)
    batch = [
        {
            "query_embs": q,
            "pos_doc_embs": torch.randn(5, 128),
            "neg_doc_embs": torch.randn(4, 128),
            "qid": "q0", "pos_pid": "d0", "neg_pid": "d1",
        },
    ]
    collated = collate_triples(batch)
    assert torch.allclose(collated["query_embs"][0, :3], q)
