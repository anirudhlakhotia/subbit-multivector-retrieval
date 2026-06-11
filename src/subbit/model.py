"""SubBitModel: learned projection plus sign binarization for ColBERT tokens."""
from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA

from .utils import project_to_stiefel

logger = logging.getLogger(__name__)


class InitMethod(str, Enum):
    """Initialization methods for the projection matrix R."""

    PCA = "pca"
    RANDOM_ORTHOGONAL = "random_orthogonal"
    IDENTITY = "identity"
    FROM_FILE = "from_file"


class StraightThroughSign(torch.autograd.Function):
    """Sign function with a straight-through gradient estimator."""

    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        result = torch.sign(input)
        result[result == 0] = 1.0
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


def ste_sign(x: torch.Tensor) -> torch.Tensor:
    """Apply sign with straight-through gradients."""

    return StraightThroughSign.apply(x)


def hard_sign(x: torch.Tensor) -> torch.Tensor:
    """Apply sign without useful gradients."""

    result = torch.sign(x)
    result[result == 0] = 1.0
    return result


class SubBitModel(nn.Module):
    """Learned subspace projection and sign coding for late interaction.

    The active paper model stores each document token as ``sign(R d)`` and
    scores with projected fp32 query tokens. When ``use_scale=True`` the query
    projection is multiplied by the bounded token-dependent scale head used in
    the paper.
    """

    def __init__(
        self,
        input_dim: int = 128,
        projected_dim: int = 64,
        init_method: str | InitMethod = InitMethod.PCA,
        orthogonal_constraint: bool = True,
        use_scale: bool = True,
        init_path: str | None = None,
        freeze_R: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.projected_dim = projected_dim
        self.init_method = InitMethod(init_method)
        self.orthogonal_constraint = orthogonal_constraint
        self.use_scale = use_scale
        self.freeze_R = freeze_R

        self.R = nn.Linear(input_dim, projected_dim, bias=False)

        if self.use_scale:
            self.W_scale = nn.Linear(input_dim, 1)
            nn.init.normal_(self.W_scale.weight, std=0.01)
            nn.init.constant_(self.W_scale.bias, 0.0)

        if self.init_method == InitMethod.RANDOM_ORTHOGONAL:
            self._init_random_orthogonal()
        elif self.init_method == InitMethod.IDENTITY:
            self._init_identity()
        elif self.init_method == InitMethod.FROM_FILE:
            if init_path is None:
                raise ValueError("init_method=from_file requires init_path")
            self._init_from_file(init_path)

        self._pca_fitted = self.init_method != InitMethod.PCA

        if self.freeze_R:
            self.R.weight.requires_grad_(False)
            logger.info("R frozen: projection will not be updated during training")

        logger.info(
            "SubBitModel: %dd -> %dd (%.3f bits/dim, %.1f bytes/token) | init=%s | ortho=%s",
            input_dim,
            projected_dim,
            self.bits_per_dim,
            self.bytes_per_token,
            self.init_method.value,
            orthogonal_constraint,
        )

    def _init_random_orthogonal(self) -> None:
        """Initialize R with orthonormal rows."""

        q, _ = torch.linalg.qr(torch.randn(self.input_dim, self.projected_dim))
        self.R.weight.data = q.T

    def _init_identity(self) -> None:
        """Initialize R as the leading rows of identity."""

        self.R.weight.data = torch.eye(self.input_dim)[: self.projected_dim]

    def _init_from_file(self, path: str) -> None:
        """Initialize R from a tensor file or checkpoint."""

        r = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(r, dict):
            r = r.get("R") if "R" in r else r.get("state_dict", {}).get("R.weight", r)
        if not isinstance(r, torch.Tensor):
            raise ValueError(f"init_path {path} did not yield a tensor (got {type(r)})")
        if r.shape == (self.projected_dim, self.input_dim):
            self.R.weight.data = r.float().contiguous()
        elif r.dim() == 2 and r.shape[0] >= self.projected_dim and r.shape[1] == self.input_dim:
            self.R.weight.data = r[: self.projected_dim].float().contiguous()
            logger.info(
                "R initialized from leading %d rows of source %s with shape %s",
                self.projected_dim,
                path,
                tuple(r.shape),
            )
        else:
            raise ValueError(
                f"R from {path} has shape {tuple(r.shape)}, expected "
                f"({self.projected_dim}, {self.input_dim}) or a larger row-compatible matrix"
            )
        logger.info("R initialized from file: %s", path)

    def fit_pca(self, embeddings: torch.Tensor | np.ndarray, verbose: bool = True) -> None:
        """Initialize R with PCA components fitted on embedding data."""

        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().float().numpy()
        if embeddings.shape[1] != self.input_dim:
            raise ValueError(
                f"Embedding dim {embeddings.shape[1]} != model input_dim {self.input_dim}"
            )

        pca = PCA(n_components=self.projected_dim)
        pca.fit(embeddings)
        self.R.weight.data = torch.tensor(pca.components_, dtype=torch.float32)
        self._pca_fitted = True

        if verbose:
            explained = pca.explained_variance_ratio_.sum()
            logger.info(
                "PCA init: %d components explain %.1f%% of variance from %d samples",
                self.projected_dim,
                100 * explained,
                len(embeddings),
            )

    def project(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Project embeddings to the learned subspace."""

        return self.R(embeddings)

    def document_projection(self, doc_embeddings: torch.Tensor) -> torch.Tensor:
        """Return document-side continuous logits before sign."""

        return self.project(doc_embeddings)

    def document_logits(self, doc_embeddings: torch.Tensor) -> torch.Tensor:
        """Return document-side pre-sign logits."""

        return self.document_projection(doc_embeddings)

    def binarize_logits(self, logits: torch.Tensor, use_ste: bool = True) -> torch.Tensor:
        """Binarize logits with sign."""

        return ste_sign(logits) if use_ste else hard_sign(logits)

    def binarize(self, projected: torch.Tensor, use_ste: bool = True) -> torch.Tensor:
        """Binarize projected embeddings with sign."""

        return self.binarize_logits(projected, use_ste=use_ste)

    def encode_document(self, doc_embeddings: torch.Tensor, use_ste: bool = True) -> torch.Tensor:
        """Encode document tokens as sign-coded projected vectors."""

        return self.binarize_logits(self.document_logits(doc_embeddings), use_ste=use_ste)

    def encode_document_rabitq(self, doc_embeddings: torch.Tensor) -> dict:
        """Encode document tokens with RaBitQ correction factors."""

        projected = self.document_projection(doc_embeddings)
        binary = self.binarize_logits(projected, use_ste=False)
        with torch.no_grad():
            norm = projected.norm(dim=-1, keepdim=False)
            normalized = projected / (norm.unsqueeze(-1) + 1e-10)
            vdot = (normalized * binary).sum(dim=-1)
        return {"binary": binary, "norm": norm, "vdot": vdot}

    def encode_query(
        self,
        query_embeddings: torch.Tensor,
        symmetric: bool = False,
        use_ste: bool = True,
    ) -> torch.Tensor:
        """Encode query tokens.

        Asymmetric scoring keeps query vectors in fp32 projected space.
        Symmetric scoring additionally sign-codes the query.
        """

        projected = self.project(query_embeddings)
        if self.use_scale:
            scale = 1.0 + 0.5 * torch.sigmoid(self.W_scale(query_embeddings))
            projected = projected * scale
        if symmetric:
            return self.binarize(projected, use_ste=use_ste)
        return projected

    def apply_orthogonal_constraint(self) -> None:
        """Project R back onto the Stiefel manifold."""

        if self.freeze_R:
            return
        with torch.no_grad():
            self.R.weight.data = project_to_stiefel(self.R.weight.data)

    @property
    def bits_per_dim(self) -> float:
        """Bits per original embedding dimension."""

        return self.projected_dim / self.input_dim

    @property
    def bytes_per_token(self) -> float:
        """Stored document bytes per token."""

        return self.projected_dim / 8

    def get_projection_matrix(self) -> torch.Tensor:
        """Return a copy of R."""

        return self.R.weight.data.clone()

    def save(self, path: str | Path) -> None:
        """Save the model weights and paper-model config."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": {
                    "input_dim": self.input_dim,
                    "projected_dim": self.projected_dim,
                    "init_method": self.init_method.value,
                    "orthogonal_constraint": self.orthogonal_constraint,
                    "use_scale": self.use_scale,
                    "freeze_R": self.freeze_R,
                },
            },
            path,
        )
        logger.info("Saved model to %s", path)

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> "SubBitModel":
        """Load a saved model.

        Checkpoints produced by earlier research branches may contain extra
        config keys. The paper repo ignores those keys and loads only the
        paper-model state.
        """

        checkpoint = torch.load(path, map_location=device, weights_only=False)
        raw_config = dict(checkpoint["config"])
        allowed = {
            "input_dim",
            "projected_dim",
            "init_method",
            "orthogonal_constraint",
            "use_scale",
            "freeze_R",
        }
        config = {k: raw_config[k] for k in allowed if k in raw_config}
        config.setdefault("input_dim", 128)
        config.setdefault("projected_dim", 64)
        config.setdefault("init_method", InitMethod.RANDOM_ORTHOGONAL.value)
        config.setdefault("orthogonal_constraint", False)
        config.setdefault("use_scale", True)
        config.setdefault("freeze_R", False)

        restore_init_method = config.get("init_method")
        init_config = dict(config)
        if restore_init_method == InitMethod.FROM_FILE.value:
            init_config["init_method"] = InitMethod.RANDOM_ORTHOGONAL.value

        model = cls(**init_config)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        if restore_init_method == InitMethod.FROM_FILE.value:
            model.init_method = InitMethod.FROM_FILE
        model._pca_fitted = True
        logger.info("Loaded model from %s", path)
        return model

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, projected_dim={self.projected_dim}, "
            f"bits_per_dim={self.bits_per_dim:.3f}, bytes_per_token={self.bytes_per_token:.1f}, "
            f"init={self.init_method.value}, orthogonal={self.orthogonal_constraint}"
        )
