"""Training loop for the subspace projection matrix R.

Only R is trained — ColBERT embeddings are frozen (pre-computed and cached).
This makes training very fast: R has only r × 128 parameters (e.g., 8,192 for r=64).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from .model import SubBitModel
from .losses import SubBitLoss
from .data import TriplesDataset, EmbeddingStore, collate_triples
from .utils import (
    count_parameters,
    ensure_dir,
    format_metrics,
    get_device,
)

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer for the SubBitModel.

    Handles:
      - Training loop with combined loss
      - LR scheduling (warmup + cosine decay)
      - Optional Stiefel constraint enforcement
      - Periodic evaluation and checkpointing
      - CSV logging
      - Early stopping
    """

    def __init__(
        self,
        model: SubBitModel,
        train_dataset: TriplesDataset,
        eval_fn: Optional[callable] = None,
        config: dict = None,
    ):
        self.config = config or {}
        self.device = get_device(self.config.get("hardware", {}).get("device", "auto"))
        self.model = model.to(self.device)

        # Validate embedding dimension matches model expectation
        if hasattr(train_dataset, "doc_store") and hasattr(train_dataset.doc_store, "dim"):
            data_dim = train_dataset.doc_store.dim
            if data_dim != model.input_dim:
                raise RuntimeError(
                    f"Embedding dimension mismatch: data has {data_dim}d embeddings "
                    f"but model expects input_dim={model.input_dim}. "
                    f"Check your config or ensure embeddings were produced by the correct encoder."
                )

        # Training config
        train_cfg = self.config.get("training", {})
        self.epochs = train_cfg.get("epochs", 10)
        self.batch_size = train_cfg.get("batch_size", 32)
        self.lr = train_cfg.get("lr", 1e-3)
        self.weight_decay = train_cfg.get("weight_decay", 1e-4)
        self.warmup_steps = train_cfg.get("warmup_steps", 250)
        self.max_steps = train_cfg.get("max_steps", -1)
        self.gradient_clip = train_cfg.get("gradient_clip", 1.0)
        self.save_every = train_cfg.get("save_every", 2500)
        self.eval_every = train_cfg.get("eval_every", 2500)
        self.patience = train_cfg.get("patience", 10)

        # Loss schema lives under training.loss.*.
        loss_cfg = train_cfg.get("loss") or {}
        self.criterion = SubBitLoss(
            boundary_topk_weight=loss_cfg.get("boundary_topk_weight", 1.0),
            boundary_fp_weight=loss_cfg.get("boundary_fp_weight", 1.0),
            boundary_k=loss_cfg.get("boundary_k", 5),
            ortho_weight=loss_cfg.get("ortho_weight", train_cfg.get("ortho_weight", 0.001)),
            ste_query=train_cfg.get("ste_query", train_cfg.get("use_ste", True)),
            ste_doc=train_cfg.get("ste_doc", True),
        )

        # DataLoader
        hw_cfg = self.config.get("hardware", {})
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=hw_cfg.get("num_workers", 4),
            pin_memory=hw_cfg.get("pin_memory", True),
            collate_fn=collate_triples,
            drop_last=True,
        )

        # Optimizer — only over parameters that require grad. With
        # freeze_R=True the projection is excluded automatically.
        self.optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # LR Scheduler: linear warmup → cosine decay
        total_steps = len(self.train_loader) * self.epochs
        if self.max_steps > 0:
            total_steps = min(total_steps, self.max_steps)

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=self.warmup_steps,
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max(total_steps - self.warmup_steps, 1),
            eta_min=self.lr * 0.01,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.warmup_steps],
        )

        self.eval_fn = eval_fn

        # Frozen-R fast path: when R is fixed, sign(R·d) is constant per doc.
        # Precompute once and skip the per-step projection + binarization in
        # the loss. Costs ~one full pass over the unique training docs at
        # startup; saves O(steps × batch × tokens × r) flops thereafter.
        self._doc_bin_cache: dict | None = None
        self._cache_dtype = torch.int8
        doc_codes_are_static = (
            getattr(self.model, "freeze_R", False)
        )
        if doc_codes_are_static:
            self._build_frozen_doc_bin_cache(train_dataset)

        # Paths
        paths_cfg = self.config.get("paths", {})
        self.checkpoint_dir = ensure_dir(paths_cfg.get("checkpoint_dir", "outputs/checkpoints"))
        self.output_dir = ensure_dir(paths_cfg.get("output_dir", "outputs"))

        # Logging
        self.logger_backend = self._setup_logging()

        # State
        self.global_step = 0
        self.best_metric = -float("inf")
        self.best_step: int = 0
        self.best_eval_metrics: dict = {}
        self.eval_history: list[dict] = []
        self.last_train_losses: dict = {}
        self.last_eval_measurement: dict = {}
        self.patience_counter = 0

        logger.info(f"Trainer initialized:")
        logger.info(f"  Model parameters: {count_parameters(self.model):,}")
        logger.info(f"  Training samples: {len(train_dataset):,}")
        logger.info(f"  Batch size: {self.batch_size}")
        logger.info(f"  Total steps: {total_steps:,}")
        logger.info(f"  LR: {self.lr}, Warmup: {self.warmup_steps}")

    def _setup_logging(self):
        """Initialize CSV logging."""
        log_cfg = self.config.get("logging", {})
        backend = log_cfg.get("backend", "csv")

        # CSV fallback
        logger.info("Using CSV logging")
        return None

    def _log_metrics(self, metrics: dict, step: int, prefix: str = "train") -> None:
        """No-op — CSV logging is file-based, not callback-based."""
        return

    def train(self) -> dict:
        """Run the full training loop.

        Returns:
            dict with final metrics and best checkpoint path.
        """
        logger.info("Starting training...")
        start_time = time.time()
        log_every = self.config.get("logging", {}).get("log_every", 50)

        for epoch in range(self.epochs):
            self.model.train()
            epoch_losses: dict[str, float] = {}
            epoch_steps = 0

            for batch_idx, batch in enumerate(self.train_loader):
                # Move to device
                q_embs = batch["query_embs"].to(self.device)
                pos_embs = batch["pos_doc_embs"].to(self.device)
                neg_embs = batch["neg_doc_embs"].to(self.device)
                q_mask = batch["query_mask"].to(self.device)
                pos_mask = batch["pos_doc_mask"].to(self.device)
                neg_mask = batch["neg_doc_mask"].to(self.device)

                # Frozen-R fast path: look up precomputed sign(R·d) instead of
                # recomputing it every step. Falls through to the standard path
                # when the cache wasn't built (e.g. R is trainable).
                d_pos_bin = d_neg_bin = None
                if self._doc_bin_cache is not None:
                    d_pos_bin = self._lookup_doc_bins(batch["pos_pids"], pos_mask)
                    d_neg_bin = self._lookup_doc_bins(batch["neg_pids"], neg_mask)

                # Forward
                losses = self.criterion(
                    self.model, q_embs, pos_embs, neg_embs,
                    q_mask=q_mask, d_pos_mask=pos_mask, d_neg_mask=neg_mask,
                    d_pos_bin=d_pos_bin, d_neg_bin=d_neg_bin,
                    step=self.global_step,
                )

                # Backward
                self.optimizer.zero_grad()
                losses["total"].backward()

                if self.gradient_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

                self.optimizer.step()
                self.scheduler.step()

                # Orthogonal constraint
                if self.model.orthogonal_constraint:
                    self.model.apply_orthogonal_constraint()

                # Track losses — keys come from the criterion's return dict so new
                # loss terms are picked up automatically without trainer edits.
                for k, v in losses.items():
                    if isinstance(v, torch.Tensor):
                        epoch_losses[k] = epoch_losses.get(k, 0.0) + v.item()
                epoch_steps += 1
                self.global_step += 1

                # Logging
                if self.global_step % log_every == 0:
                    avg_losses = {k: v / epoch_steps for k, v in epoch_losses.items()}
                    avg_losses["lr"] = self.scheduler.get_last_lr()[0]
                    logger.info(
                        f"[Epoch {epoch+1}/{self.epochs}] "
                        f"[Step {self.global_step}] "
                        f"{format_metrics(avg_losses, 'train')}"
                    )
                    self._log_metrics(avg_losses, self.global_step, "train")
                    self.last_train_losses = dict(avg_losses)

                # Evaluation
                if self.eval_fn and self.global_step % self.eval_every == 0:
                    eval_metrics = self._evaluate()
                    measurement = eval_metrics.pop("_measurement", None)
                    self._log_metrics(eval_metrics, self.global_step, "eval")

                    history_entry = {"step": self.global_step, **eval_metrics}
                    self.eval_history.append(history_entry)
                    if measurement is not None:
                        self.last_eval_measurement = measurement

                    # Checkpointing (best model)
                    primary_metric = eval_metrics.get("mrr@10", eval_metrics.get("recall@100", 0))
                    if primary_metric > self.best_metric:
                        self.best_metric = primary_metric
                        self.best_step = self.global_step
                        self.best_eval_metrics = dict(eval_metrics)
                        self.patience_counter = 0
                        self._save_checkpoint("best")
                        logger.info(f"  → New best: {primary_metric:.4f}")
                    else:
                        self.patience_counter += 1

                    if self.patience_counter >= self.patience:
                        logger.info(f"Early stopping at step {self.global_step}")
                        break

                # Periodic checkpoint
                if self.global_step % self.save_every == 0:
                    self._save_checkpoint(f"step_{self.global_step}")

                # Max steps
                if 0 < self.max_steps <= self.global_step:
                    logger.info(f"Reached max_steps={self.max_steps}")
                    break

            if self.patience_counter >= self.patience:
                break
            if 0 < self.max_steps <= self.global_step:
                break

        # Final save
        self._save_checkpoint("final")
        elapsed = time.time() - start_time

        result = {
            "total_steps": self.global_step,
            "best_metric": self.best_metric,
            "best_step": self.best_step,
            "best_eval_metrics": self.best_eval_metrics,
            "eval_history": self.eval_history,
            "final_train_losses": self.last_train_losses,
            "last_eval_measurement": self.last_eval_measurement,
            "training_time_seconds": elapsed,
            "best_checkpoint": str(self.checkpoint_dir / "best.pt"),
        }
        log_summary = {
            "best_metric": self.best_metric,
            "best_step": self.best_step,
            "total_steps": self.global_step,
            "training_time_seconds": elapsed,
            "best_checkpoint": str(self.checkpoint_dir / "best.pt"),
        }
        logger.info(f"Training complete in {elapsed:.1f}s | {format_metrics(log_summary)}")

        return result

    def _build_frozen_doc_bin_cache(self, train_dataset) -> None:
        """Precompute sign(R·d) for every unique doc in the training triples.

        Stored as int8 ({-1, +1}) on CPU to keep peak memory low; per-step
        lookup pads to the batch's max doc length and casts to float on the
        training device. With R frozen this is bit-identical to calling
        model.encode_document on every step but avoids the matmul + sign.
        """
        doc_store = getattr(train_dataset, "doc_store", None)
        triples = getattr(train_dataset, "triples", None)
        if doc_store is None or triples is None:
            logger.warning(
                "freeze_R=True but train_dataset lacks doc_store/triples; "
                "frozen-R cache disabled, falling back to per-step encoding."
            )
            return

        unique_pids = sorted({pid for _, pid, _ in triples} | {pid for _, _, pid in triples})
        max_doc_tokens = getattr(train_dataset, "max_doc_tokens", 180)
        r = self.model.projected_dim
        logger.info(
            "Frozen-R fast path: precomputing sign(R·d) for %d unique docs (r=%d)…",
            len(unique_pids), r,
        )

        cache: dict = {}
        # Encode in chunks to avoid materialising every doc at once.
        chunk = 1024
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(unique_pids), chunk):
                pids = unique_pids[start : start + chunk]
                for pid in pids:
                    d = doc_store.get(pid)[:max_doc_tokens].to(self.device)
                    bin_d = self.model.encode_document(d, use_ste=False)
                    # {-1, +1} → int8 on CPU. Float cast happens at lookup time.
                    cache[pid] = bin_d.to(torch.int8).cpu()
        if was_training:
            self.model.train()

        self._doc_bin_cache = cache
        # Memory accounting for the log line.
        total_bytes = sum(t.numel() for t in cache.values())
        logger.info(
            "Frozen-R cache: %d docs, %.1f MB int8 on CPU.",
            len(cache), total_bytes / (1024 * 1024),
        )

    def _lookup_doc_bins(self, pids: list[str], mask: torch.Tensor) -> torch.Tensor:
        """Pad cached binary codes to match the batch's (B, max_n, r) shape.

        Uses the float-side mask to decide max_n so the loss's per-sample
        masking lines up with the float doc tensor without modification.
        """
        B, max_n = mask.shape
        r = self.model.projected_dim
        out = torch.zeros(B, max_n, r, dtype=torch.float32, device=self.device)
        for i, pid in enumerate(pids):
            cached = self._doc_bin_cache[pid]
            n = min(cached.shape[0], max_n)
            out[i, :n] = cached[:n].to(self.device, dtype=torch.float32)
        return out

    def _evaluate(self) -> dict:
        """Run evaluation using the provided eval function."""
        self.model.eval()
        with torch.no_grad():
            metrics = self.eval_fn(self.model, self.device)
        self.model.train()

        scalar_metrics = {k: v for k, v in metrics.items() if not k.startswith("_")}
        logger.info(f"  Eval @ step {self.global_step}: {format_metrics(scalar_metrics, 'eval')}")
        return metrics

    def _save_checkpoint(self, name: str) -> None:
        """Save model checkpoint."""
        path = self.checkpoint_dir / f"{name}.pt"
        self.model.save(path)
        logger.info(f"  Saved checkpoint: {path}")
