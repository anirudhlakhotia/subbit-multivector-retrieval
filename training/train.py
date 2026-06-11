"""Train the subspace projection matrix R.

Usage:
  # Standard training (reproduces paper r=64 checkpoint)
  python training/train.py --config configs/base.yaml

  # Debug mode (small data, few steps)
  python training/train.py --config configs/base.yaml --debug

  # Override projected dimension
  python training/train.py --config configs/base.yaml model.projected_dim=32
"""
from __future__ import annotations

import argparse
import logging
import sys
from functools import partial
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.subbit.model import SubBitModel
from src.subbit.data import (
    EmbeddingStore,
    TriplesDataset,
    create_debug_data,
    load_qrels,
    resolve_embedding_cache_path,
)
from src.subbit.training import Trainer
from src.subbit.evaluation import evaluate_retrieval
from src.subbit.utils import setup_logging, get_device, seed_everything, ensure_dir

logger = logging.getLogger(__name__)


def _load_config(
    config_files: list[str] | None,
    overrides: list[str] | None,
) -> "OmegaConf":
    """Load stacked configs, then apply CLI dot-list overrides."""
    config_files = config_files or ["configs/base.yaml"]

    cfg = OmegaConf.load(config_files[0])
    for extra in config_files[1:]:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(extra))

    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

    return cfg


def make_eval_fn(query_store, doc_store, qrels, doc_ids, metrics, max_queries):
    """Create an evaluation function closure for the Trainer."""

    def eval_fn(model, device):
        return evaluate_retrieval(
            model=model,
            query_store=query_store,
            doc_store=doc_store,
            qrels=qrels,
            doc_ids=doc_ids,
            device=device,
            metrics=metrics,
            max_queries=max_queries,
        )

    return eval_fn


def main():
    parser = argparse.ArgumentParser(description="Train subspace projection R")
    parser.add_argument("--config", type=str, action="append", default=None,
                        help="Config file(s). Multiple allowed; later files override earlier ones.")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (overrides training.seed in config)")
    parser.add_argument(
        "overrides", nargs="*",
        help="Config overrides in dotlist format (e.g., model.projected_dim=32)",
    )
    args = parser.parse_args()

    # Load and merge config files (later files override earlier ones)
    config_files = args.config or []
    
    true_overrides = []
    for o in args.overrides:
        if o.endswith(".yaml") or o.endswith(".yml"):
            config_files.append(o)
        else:
            true_overrides.append(o)
    args.overrides = true_overrides

    if not config_files:
        config_files = ["configs/base.yaml"]
        
    cfg = _load_config(config_files, args.overrides)

    if args.debug:
        cfg.debug.enabled = True

    if args.seed is not None:
        cfg.training.seed = args.seed

    setup_logging()
    seed_everything(cfg.training.seed)
    device = get_device(cfg.hardware.device)

    # Paths
    embeddings_dir = Path(cfg.data.embeddings_dir)
    output_dir = ensure_dir(cfg.paths.output_dir)
    checkpoint_dir = ensure_dir(cfg.paths.checkpoint_dir)

    # ---------------------------------------------------------------------------
    # Data
    # ---------------------------------------------------------------------------
    if cfg.debug.enabled:
        logger.info("=== DEBUG MODE ===")
        debug_dir = embeddings_dir / "debug"

        if not (debug_dir / "query_embeddings.pt").exists():
            logger.info("Creating debug data...")
            create_debug_data(
                debug_dir,
                num_queries=cfg.debug.num_queries,
                num_passages=cfg.debug.num_passages,
                dim=cfg.model.input_dim,
            )

        embeddings_dir = debug_dir
        cfg.training.max_steps = cfg.debug.max_steps
        if cfg.data.num_train_samples == -1:
            cfg.data.num_train_samples = cfg.debug.num_queries * 2

    # Load embedding stores
    try:
        query_store = EmbeddingStore(resolve_embedding_cache_path(embeddings_dir, "query"), mode="dict")
        query_store.load()
    except FileNotFoundError as e:
        logger.error(str(e))
        logger.error(
            "No pre-computed embeddings found. Either:\n"
            "  1. Run 'python training/encode_corpus.py --config configs/base.yaml' first, or\n"
            "  2. Use '--debug' for synthetic data: 'python training/train.py --config configs/base.yaml --debug'"
        )
        sys.exit(1)

    try:
        doc_store = EmbeddingStore(resolve_embedding_cache_path(embeddings_dir, "doc"), mode="dict")
        doc_store.load()
    except FileNotFoundError as e:
        logger.error(str(e))
        logger.error(
            "No pre-computed doc embeddings found. Either:\n"
            "  1. Run 'python training/encode_corpus.py --config configs/base.yaml' first, or\n"
            "  2. Use '--debug' for synthetic data: 'python training/train.py --config configs/base.yaml --debug'"
        )
        sys.exit(1)

    # Load triples
    if "triples_path" in cfg.data:
        triples_path = Path(cfg.data.triples_path)
    else:
        triples_path = embeddings_dir / "triples.tsv"
    
    if not triples_path.exists() and "train_triples" in cfg.data:
        triples_path = Path(cfg.data.train_triples)

    max_samples = cfg.data.get("num_train_samples", -1)
    train_dataset = TriplesDataset(
        triples_path=triples_path,
        query_store=query_store,
        doc_store=doc_store,
        max_query_tokens=cfg.data.max_query_tokens,
        max_doc_tokens=cfg.data.max_doc_tokens,
        max_samples=max_samples,
    )

    # Load qrels for evaluation
    qrels_path = embeddings_dir / "qrels.tsv"
    if not qrels_path.exists():
        qrels_path = Path(cfg.data.get("dev_qrels", ""))

    qrels = load_qrels(qrels_path) if qrels_path.exists() else {}

    # ---------------------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------------------
    model = SubBitModel(
        input_dim=cfg.model.input_dim,
        projected_dim=cfg.model.projected_dim,
        init_method=cfg.model.init,
        orthogonal_constraint=cfg.model.orthogonal_constraint,
        use_scale=cfg.model.get("use_scale", True),
        init_path=cfg.model.get("init_path", None),
        freeze_R=cfg.model.get("freeze_R", False),
    )

    # PCA initialization
    if cfg.model.init == "pca":
        logger.info("Fitting PCA initialization...")
        sample_size = min(cfg.data.pca_sample_size, len(doc_store.get_all_ids()) * 50)
        if cfg.data.get("pca_train_split_only", True):
            train_doc_ids = sorted({pid for _, pid, _ in train_dataset.triples} |
                                   {pid for _, _, pid in train_dataset.triples})
            logger.info(
                "PCA fit restricted to training triples: %d unique docs",
                len(train_doc_ids),
            )
            pca_data = doc_store.sample_embeddings(sample_size, ids=train_doc_ids)
        else:
            pca_data = doc_store.sample_embeddings(sample_size)
        model.fit_pca(pca_data)

    # ---------------------------------------------------------------------------
    # Evaluation function
    # ---------------------------------------------------------------------------
    eval_fn = None
    if qrels:
        doc_ids = doc_store.get_all_ids()
        # In-loop eval defaults to a 1000-query subset for tracking — full
        # 6,980-query eval is run on the best checkpoint after training.
        # Override via `evaluation.in_loop_max_queries: -1` to restore full eval.
        eval_cfg = cfg.get("evaluation", {})
        if cfg.debug.enabled:
            eval_max_q = cfg.debug.num_queries
        else:
            eval_max_q = eval_cfg.get("in_loop_max_queries", 1000)
        eval_fn = make_eval_fn(
            query_store=query_store,
            doc_store=doc_store,
            qrels=qrels,
            doc_ids=doc_ids,
            metrics=list(cfg.evaluation.metrics),
            max_queries=eval_max_q,
        )

    # ---------------------------------------------------------------------------
    # Train
    # ---------------------------------------------------------------------------
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_fn=eval_fn,
        config=config_dict,
    )

    results = trainer.train()

    # Enrich results with reviewer-ready metadata so every checkpoint is
    # self-describing (git SHA, torch/cuda versions, param count, config).
    from src.subbit.measurement import collect_run_metadata

    def _count_params(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    results.update({
        "model_parameters": _count_params(model),
        "projected_dim": model.projected_dim,
        "input_dim": model.input_dim,
        "bits_per_dim": model.bits_per_dim,
        "bytes_per_token": model.projected_dim // 8,
        "run_metadata": collect_run_metadata(config=config_dict),
    })

    # Save results
    results_path = output_dir / "training_results.json"
    import json
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
