"""Paper loss for training the SubBit projection matrix R."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def orthogonal_regularization(model) -> torch.Tensor:
    """Penalize deviation from orthonormal rows: ||R R^T - I||_F^2."""

    r = model.R.weight
    rrt = r @ r.T
    eye = torch.eye(rrt.shape[0], device=rrt.device)
    return torch.norm(rrt - eye, p="fro") ** 2


def boundary_guard_loss(
    q_proj: torch.Tensor,
    d_bin: torch.Tensor,
    q_full: torch.Tensor,
    d_full: torch.Tensor,
    k: int = 5,
    q_mask: torch.Tensor | None = None,
    d_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match the top-K scores, and guard the highest non-top-K sentinel."""

    batch_size = q_proj.shape[0]
    total_loss_topk = torch.tensor(0.0, device=q_proj.device)
    total_loss_guard = torch.tensor(0.0, device=q_proj.device)
    total_tokens = 0

    for i in range(batch_size):
        q_p = q_proj[i]
        d_b = d_bin[i]
        q_f = q_full[i]
        d_f = d_full[i]

        full_sim = q_f @ d_f.T
        if d_mask is not None and d_mask[i] is not None:
            full_sim = full_sim.masked_fill(~d_mask[i].unsqueeze(0), float("-inf"))
            if d_mask[i].sum() < k + 1:
                continue

        teacher_topk_scores, teacher_topk_idx = full_sim.topk(k, dim=-1)

        proj_sim = (q_p @ d_b.T) * (1.0 / math.sqrt(d_b.shape[-1]))
        if d_mask is not None and d_mask[i] is not None:
            proj_sim = proj_sim.masked_fill(~d_mask[i].unsqueeze(0), float("-inf"))

        student_topk_scores = proj_sim.gather(-1, teacher_topk_idx)
        mse_topk = F.mse_loss(
            student_topk_scores, teacher_topk_scores, reduction="none"
        ).sum(dim=-1)

        topk_mask = torch.zeros_like(proj_sim, dtype=torch.bool).scatter_(
            -1, teacher_topk_idx, True
        )
        student_non_topk = proj_sim.masked_fill(topk_mask, float("-inf"))
        sentinel_idx = student_non_topk.argmax(dim=-1, keepdim=True).detach()
        student_guard_scores = proj_sim.gather(-1, sentinel_idx).squeeze(-1)
        teacher_guard_scores = full_sim.gather(-1, sentinel_idx).squeeze(-1)
        mse_guard = F.mse_loss(
            student_guard_scores, teacher_guard_scores, reduction="none"
        )

        if q_mask is not None and q_mask[i] is not None:
            valid = q_mask[i]
            if valid.sum() > 0:
                total_loss_topk = total_loss_topk + mse_topk[valid].sum()
                total_loss_guard = total_loss_guard + mse_guard[valid].sum()
                total_tokens += valid.sum()
        else:
            total_loss_topk = total_loss_topk + mse_topk.sum()
            total_loss_guard = total_loss_guard + mse_guard.sum()
            total_tokens += q_p.shape[0]

    if total_tokens == 0:
        zero = torch.tensor(0.0, device=q_proj.device)
        return zero, zero
    return total_loss_topk / total_tokens, total_loss_guard / total_tokens


class SubBitLoss(nn.Module):
    """Boundary-guard loss used by the paper training recipe."""

    def __init__(
        self,
        boundary_topk_weight: float = 1.0,
        boundary_fp_weight: float = 1.0,
        boundary_k: int = 5,
        ortho_weight: float = 0.001,
        ste_query: bool = True,
        ste_doc: bool = True,
    ):
        super().__init__()
        self.boundary_topk_weight = boundary_topk_weight
        self.boundary_fp_weight = boundary_fp_weight
        self.boundary_k = boundary_k
        self.ortho_weight = ortho_weight
        self.ste_query = ste_query
        self.ste_doc = ste_doc

    def forward(
        self,
        model,
        q_emb: torch.Tensor,
        d_pos_emb: torch.Tensor,
        d_neg_emb: torch.Tensor,
        q_mask: torch.Tensor | None = None,
        d_pos_mask: torch.Tensor | None = None,
        d_neg_mask: torch.Tensor | None = None,
        d_pos_bin: torch.Tensor | None = None,
        d_neg_bin: torch.Tensor | None = None,
        step: int | None = None,
    ) -> dict[str, torch.Tensor]:
        q_proj = model.encode_query(q_emb, symmetric=False, use_ste=self.ste_query)
        if d_pos_bin is None:
            d_pos_bin = model.encode_document(d_pos_emb, use_ste=self.ste_doc)
        else:
            d_pos_bin = d_pos_bin.to(q_proj.device, dtype=q_proj.dtype)
        if d_neg_bin is None:
            d_neg_bin = model.encode_document(d_neg_emb, use_ste=self.ste_doc)
        else:
            d_neg_bin = d_neg_bin.to(q_proj.device, dtype=q_proj.dtype)

        topk_loss, fp_loss = boundary_guard_loss(
            q_proj,
            d_pos_bin,
            q_emb,
            d_pos_emb,
            k=self.boundary_k,
            q_mask=q_mask,
            d_mask=d_pos_mask,
        )
        ortho_loss = orthogonal_regularization(model)

        total = (
            self.boundary_topk_weight * topk_loss
            + self.boundary_fp_weight * fp_loss
            + self.ortho_weight * ortho_loss
        )

        return {
            "total": total,
            "boundary_topk": topk_loss.detach(),
            "boundary_fp": fp_loss.detach(),
            "ortho": ortho_loss.detach(),
        }
