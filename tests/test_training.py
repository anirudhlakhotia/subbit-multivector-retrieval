"""Integration tests for Trainer configuration wiring."""

from pathlib import Path
import sys

import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.subbit.model import SubBitModel
from src.subbit.data import collate_triples
from src.subbit.training import Trainer


class DummyTriplesDataset(Dataset):
    """Tiny dataset sufficient for Trainer initialization tests."""

    def __len__(self):
        return 4

    def __getitem__(self, idx):
        emb = torch.randn(2, 128)
        return {
            "query_embs": emb,
            "pos_doc_embs": emb,
            "neg_doc_embs": emb,
            "qid": str(idx),
            "pos_pid": str(idx),
            "neg_pid": str(idx + 1),
        }


class TinyStore:
    def __init__(self, data):
        self.data = data
        self.dim = 128

    def get(self, id_):
        return self.data[id_]

    def get_all_ids(self):
        return list(self.data.keys())


class MiningTriplesDataset(Dataset):
    def __init__(self):
        self.query_store = TinyStore({"q0": torch.ones(2, 128)})
        self.doc_store = TinyStore({
            "p0": torch.ones(2, 128),
            "n0": torch.zeros(2, 128),
            "n1": torch.full((3, 128), 2.0),
        })
        self.triples = [("q0", "p0", "n0")]
        self.max_query_tokens = 2
        self.max_doc_tokens = 3

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        qid, pos_pid, neg_pid = self.triples[idx]
        return {
            "query_embs": self.query_store.get(qid),
            "pos_doc_embs": self.doc_store.get(pos_pid),
            "neg_doc_embs": self.doc_store.get(neg_pid),
            "qid": qid,
            "pos_pid": pos_pid,
            "neg_pid": neg_pid,
        }


def test_trainer_results_include_paper_recording_fields(tmp_path):
    """Trainer.train() must return every field needed for the paper record.

    The fields below are the one-stop resource the paper draws on; if any of
    them goes missing the JSON written to `outputs/<run>/training_results.json`
    stops being self-contained.
    """
    config = {
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "lr": 1e-3,
            "warmup_steps": 1,
            "max_steps": 2,
            "eval_every": 1,
            "save_every": 1,
            "patience": 10,
            "loss": {"boundary_topk_weight": 1.0, "boundary_fp_weight": 1.0,
                     "boundary_k": 2, "ortho_weight": 0.001},
        },
        "hardware": {"device": "cpu", "num_workers": 0, "pin_memory": False},
        "paths": {
            "output_dir": str(tmp_path / "outputs"),
            "checkpoint_dir": str(tmp_path / "checkpoints"),
        },
        "logging": {"backend": "csv", "log_every": 1},
    }

    eval_calls = {"n": 0}

    def stub_eval_fn(model, device):
        eval_calls["n"] += 1
        score = 0.3 + 0.1 * eval_calls["n"]  # monotonically increasing
        return {
            "mrr@10": score,
            "recall@100": score + 0.05,
            "recall@1000": score + 0.1,
            "_measurement": {
                "latency": {"mean_ms": 1.0, "p50_ms": 1.0, "throughput_qps": 1000.0},
                "memory": {"device_type": "cpu", "device_peak_mb": None, "host_rss_mb": 1.0},
                "num_queries": 4,
                "num_docs": 4,
                "top_k": 10,
            },
        }

    trainer = Trainer(
        model=SubBitModel(128, 64, init_method="random_orthogonal"),
        train_dataset=DummyTriplesDataset(),
        eval_fn=stub_eval_fn,
        config=config,
    )

    result = trainer.train()

    assert "best_metric" in result
    assert "best_step" in result
    assert "best_eval_metrics" in result
    assert "eval_history" in result
    assert "final_train_losses" in result
    assert "last_eval_measurement" in result
    assert "training_time_seconds" in result
    assert "best_checkpoint" in result

    assert result["best_eval_metrics"].get("mrr@10") == result["best_metric"]
    assert "recall@100" in result["best_eval_metrics"]
    assert "recall@1000" in result["best_eval_metrics"]

    assert len(result["eval_history"]) >= 1
    for entry in result["eval_history"]:
        assert "step" in entry
        assert "mrr@10" in entry
        assert "_measurement" not in entry  # stripped before history

    assert "total" in result["final_train_losses"]
    assert "lr" in result["final_train_losses"]

    assert "latency" in result["last_eval_measurement"]
    assert "memory" in result["last_eval_measurement"]
    assert result["last_eval_measurement"]["num_queries"] == 4


def test_trainer_uses_active_loss_config_keys(tmp_path):
    """Trainer reads training.loss.* and plumbs them through to SubBitLoss."""
    config = {
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "lr": 1e-3,
            "warmup_steps": 1,
            "ste_query": True,
            "ste_doc": False,
            "loss": {
                "boundary_topk_weight": 0.7,
                "boundary_fp_weight": 0.3,
                "boundary_k": 5,
                "ortho_weight": 0.01,
            },
        },
        "hardware": {"device": "cpu", "num_workers": 0, "pin_memory": False},
        "paths": {
            "output_dir": str(tmp_path / "outputs"),
            "checkpoint_dir": str(tmp_path / "checkpoints"),
        },
        "logging": {"backend": "csv"},
    }

    trainer = Trainer(
        model=SubBitModel(128, 64, init_method="random_orthogonal"),
        train_dataset=DummyTriplesDataset(),
        config=config,
    )

    criterion = trainer.criterion
    assert criterion.boundary_topk_weight == 0.7
    assert criterion.boundary_fp_weight == 0.3
    assert criterion.boundary_k == 5
    assert criterion.ortho_weight == 0.01
    assert criterion.ste_query is True
    assert criterion.ste_doc is False
