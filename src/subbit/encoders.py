"""Encoder protocol and implementations for multi-vector retrieval.

Defines the ``MultiVectorEncoder`` protocol so any encoder that produces
per-token embeddings can be used with the SubBit pipeline.  Concrete
implementations are provided for ColBERT.

Usage::

    from src.subbit.encoders import ColBERTEncoder

    encoder = ColBERTEncoder()                       # loads colbert-ir/colbertv2.0
    q_embs  = encoder.encode_queries(["what is IR?"])
    d_embs  = encoder.encode_documents(["Information retrieval is ..."])
"""
from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class MultiVectorEncoder(Protocol):
    """Any encoder that produces per-token embeddings."""

    @property
    def dim(self) -> int:
        """Embedding dimension (128 for ColBERTv2)."""
        ...

    def encode_queries(
        self,
        texts: list[str],
        *,
        batch_size: int = 32,
        device: torch.device | str = "cpu",
    ) -> dict[int, torch.Tensor]:
        """Encode queries → {idx: (num_tokens, dim)} on CPU."""
        ...

    def encode_documents(
        self,
        texts: list[str],
        *,
        batch_size: int = 32,
        device: torch.device | str = "cpu",
    ) -> dict[int, torch.Tensor]:
        """Encode documents → {idx: (num_tokens, dim)} on CPU."""
        ...


# ---------------------------------------------------------------------------
# ColBERT implementation
# ---------------------------------------------------------------------------

class ColBERTEncoder:
    """ColBERT multi-vector encoder.

    Tries to use the ``colbert`` library first; falls back to
    ``transformers`` if unavailable.
    """

    _dim: int = 128

    def __init__(
        self,
        model_name: str = "colbert-ir/colbertv2.0",
        query_maxlen: int = 32,
        doc_maxlen: int = 180,
    ) -> None:
        self.model_name = model_name
        self.query_maxlen = query_maxlen
        self.doc_maxlen = doc_maxlen
        self._checkpoint = None  # lazy loaded

    @property
    def dim(self) -> int:
        return self._dim

    def _load(self) -> None:
        if self._checkpoint is not None:
            return
        try:
            from colbert.infra import ColBERTConfig
            from colbert.modeling.checkpoint import Checkpoint

            config = ColBERTConfig(
                doc_maxlen=self.doc_maxlen, query_maxlen=self.query_maxlen
            )
            self._checkpoint = Checkpoint(self.model_name, colbert_config=config)
            self._backend = "colbert"
        except ImportError:
            raise RuntimeError(
                "colbert-ai is required for ColBERTEncoder. "
                "Install it with: pip install colbert-ai\n"
                "The raw Transformers fallback produces 768d hidden states "
                "without ColBERT's learned 768→128 projection or L2 "
                "normalization, which would produce invalid results with "
                "the 128d→r sub-bit pipeline."
            )
        logger.info("Loaded ColBERT encoder via %s backend", self._backend)

    def encode_queries(
        self,
        texts: list[str],
        *,
        batch_size: int = 32,
        device: torch.device | str = "cpu",
    ) -> dict[int, torch.Tensor]:
        return self._encode(texts, is_query=True, batch_size=batch_size, device=device)

    def encode_documents(
        self,
        texts: list[str],
        *,
        batch_size: int = 32,
        device: torch.device | str = "cpu",
    ) -> dict[int, torch.Tensor]:
        return self._encode(texts, is_query=False, batch_size=batch_size, device=device)

    def _encode(
        self,
        texts: list[str],
        *,
        is_query: bool,
        batch_size: int,
        device: torch.device | str,
    ) -> dict[int, torch.Tensor]:
        self._load()
        device = torch.device(device) if isinstance(device, str) else device
        embeddings: dict[int, torch.Tensor] = {}
        max_length = self.query_maxlen if is_query else self.doc_maxlen

        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding", leave=False):
            batch_texts = texts[i : i + batch_size]
            batch_indices = list(range(i, min(i + batch_size, len(texts))))

            with torch.no_grad():
                if is_query:
                    embs = self._checkpoint.queryFromText(batch_texts)
                else:
                    embs = self._checkpoint.docFromText(batch_texts)
                if isinstance(embs, list):
                    for idx, emb in zip(batch_indices, embs):
                        embeddings[idx] = emb.cpu()
                else:
                    for j, idx in enumerate(batch_indices):
                        embeddings[idx] = embs[j].cpu()

        return embeddings


# ---------------------------------------------------------------------------
# ConstBERT implementation (Lassance et al. 2024 / pinecone/ConstBERT)
# ---------------------------------------------------------------------------


class ConstBERTEncoder:
    """ConstBERT (constant-space late interaction) encoder.

    Produces a fixed K vectors per document (K=32 for the public
    pinecone/ConstBERT checkpoint), via a learned 250->32 doc-token
    aggregation on top of BERT-base. Queries are padded to query_maxlen
    (32) tokens. Both are L2-normalised, dim=128.

    Used for cross-encoder-family checks: if a SubBit / lex-coverage
    finding holds on ConstBERT as well as ColBERTv2/Jina, the
    "same-family encoders" critique is defused.
    """

    def __init__(
        self,
        model_name: str = "pinecone/ConstBERT",
        query_maxlen: int = 32,
        doc_maxlen: int = 250,
    ) -> None:
        self.model_name = model_name
        self.query_maxlen = query_maxlen
        self.doc_maxlen = doc_maxlen
        self._model = None
        self._device = None
        self._dim_cached = 128

    @property
    def dim(self) -> int:
        return self._dim_cached

    def _load(self, device: torch.device | str = "cpu") -> None:
        if self._model is not None and self._device == device:
            return
        from transformers import AutoModel

        model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
        # The HF code wraps doc/query forward in torch.amp.autocast("cuda").
        # On MPS / CPU that's a noop or a warning source - turn it off.
        if hasattr(model, "amp_manager"):
            model.amp_manager.activated = False
        model.eval()
        model.to(device)
        self._model = model
        self._device = device
        self._dim_cached = int(getattr(model, "dim", 128))
        logger.info(
            "Loaded ConstBERT (%s) on %s; dim=%d, K=%d",
            self.model_name,
            device,
            self._dim_cached,
            int(model.doc_project.out_features),
        )

    def encode_queries(
        self,
        texts: list[str],
        *,
        batch_size: int = 32,
        device: torch.device | str = "cpu",
    ) -> dict[int, torch.Tensor]:
        self._load(device)
        out: dict[int, torch.Tensor] = {}
        for i in tqdm(
            range(0, len(texts), batch_size),
            desc="ConstBERT Q",
            leave=False,
            disable=len(texts) < batch_size * 2,
        ):
            batch = texts[i : i + batch_size]
            with torch.no_grad():
                Q = self._model.encode_queries(batch, bsize=batch_size, to_cpu=True)
            # Q: (B, query_maxlen, dim) on CPU
            Q = Q.float().cpu()
            for j, txt in enumerate(batch):
                out[i + j] = Q[j].contiguous()
        return out

    def encode_documents(
        self,
        texts: list[str],
        *,
        batch_size: int = 32,
        device: torch.device | str = "cpu",
    ) -> dict[int, torch.Tensor]:
        self._load(device)
        out: dict[int, torch.Tensor] = {}
        for i in tqdm(
            range(0, len(texts), batch_size),
            desc="ConstBERT D",
            leave=False,
            disable=len(texts) < batch_size * 2,
        ):
            batch = texts[i : i + batch_size]
            with torch.no_grad():
                D = self._model.encode_documents(batch, bsize=batch_size, keep_dims=True, to_cpu=True)
            if isinstance(D, tuple):
                D = D[0]
            # D: (B, K, dim) on CPU
            D = D.float().cpu()
            for j, txt in enumerate(batch):
                out[i + j] = D[j].contiguous()
        return out


# ---------------------------------------------------------------------------
# Pre-computed embedding "encoder" (for cached embeddings)
# ---------------------------------------------------------------------------

class PrecomputedEncoder:
    """Adapter that wraps pre-computed embeddings as an encoder.

    Useful when embeddings are already cached on disk and you want
    a uniform encoder interface in pipelines.
    """

    def __init__(self, input_dim: int, query_embs: dict, doc_embs: dict) -> None:
        self._dim = input_dim
        self._query_embs = query_embs
        self._doc_embs = doc_embs

    @property
    def dim(self) -> int:
        return self._dim

    def encode_queries(self, texts: list[str], **kwargs) -> dict[int, torch.Tensor]:
        return dict(self._query_embs)

    def encode_documents(self, texts: list[str], **kwargs) -> dict[int, torch.Tensor]:
        return dict(self._doc_embs)
