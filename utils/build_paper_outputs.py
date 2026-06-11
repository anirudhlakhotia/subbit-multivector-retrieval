#!/usr/bin/env python3
"""Build paper-facing JSON outputs from the raw local result files."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "artifacts" / "results"
PAPER_RESULTS = ROOT / "artifacts" / "paper" / "results"
OUTPUT_LINKS = ROOT / "outputs" / "paper"
PAPER_TEX = "paper/paper_reneuir.tex"


def load(relpath: str) -> Any:
    with (ROOT / relpath).open() as f:
        return json.load(f)


def dump(relpath: str, payload: Any) -> None:
    path = PAPER_RESULTS / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def row_by_label(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    for row in rows:
        if row.get("label") == label or row.get("meta", {}).get("label") == label:
            return row
    raise KeyError(label)


def row_by_prefix(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    for row in rows:
        label = row.get("label") or row.get("meta", {}).get("label") or ""
        if label.startswith(prefix):
            return row
    raise KeyError(prefix)


def metric(row: dict[str, Any], name: str) -> float:
    return float(row["metrics"][name])


def value(exact: Any, paper: str) -> dict[str, Any]:
    return {"exact": exact, "paper": paper}


def f(exact: float, digits: int) -> dict[str, Any]:
    return value(exact, f"{exact:.{digits}f}")


def sf(exact: float, digits: int) -> dict[str, Any]:
    return value(exact, f"{exact:+.{digits}f}")


def pct(frac: float, digits: int = 1) -> dict[str, Any]:
    return value(frac, f"{100.0 * frac:.{digits}f}%")


def pct_from_percent(exact_percent: float, digits: int = 1) -> dict[str, Any]:
    return value(exact_percent, f"{exact_percent:.{digits}f}%")


def common(label: str, source_files: list[str]) -> dict[str, Any]:
    return {
        "paper": PAPER_TEX,
        "paper_reference": label,
        "source_files": source_files,
    }


def qrow(
    group: str,
    method: str,
    source_file: str | None,
    bytes_per_token: int,
    mrr: float | None,
    recall100: float | None,
    recall1000: float | None,
    mrr_digits: int,
    recall_digits: int,
    recall1000_digits: int,
) -> dict[str, Any]:
    return {
        "group": group,
        "method": method,
        "source_file": source_file,
        "bytes_per_token": value(bytes_per_token, str(bytes_per_token)),
        "mrr@10": None if mrr is None else f(mrr, mrr_digits),
        "recall@100": None if recall100 is None else f(recall100, recall_digits),
        "recall@1000": None if recall1000 is None else f(recall1000, recall1000_digits),
    }


def build_storage_quality() -> None:
    baseline_path = "artifacts/results/aug_eval/baseline_100k_aug_r64.json"
    plaid_path = "artifacts/results/aug_eval/table_plaid_noscale_aug.json"
    itq100_path = "artifacts/results/aug_eval/itq_100k_aug.json"
    pq_path = "artifacts/results/aug_eval/pq_opq_100k_aug.json"
    rabitq100_path = "artifacts/results/aug_eval/rabitq_100k_aug.json"
    rerank100_path = "artifacts/results/aug_eval/rerank_aug_fullfp32.json"
    baseline = load(baseline_path)
    plaid = load(plaid_path)
    itq100 = load(itq100_path)
    pq = load(pq_path)
    rabitq100 = load(rabitq100_path)
    rerank100 = load(rerank100_path)
    boot64 = load("artifacts/results/full_msmarco/bootstrap_r64_8m.json")

    def base(label: str) -> dict[str, Any]:
        return row_by_label(baseline["rows"], label)

    def plaid_row(label: str) -> dict[str, Any]:
        return row_by_label(plaid["rows"], label)

    def pq_row(label: str) -> dict[str, Any]:
        return row_by_label(pq["rows"], label)

    def pq_prefix(label: str) -> dict[str, Any]:
        return row_by_prefix(pq["rows"], label)

    full_sources = {
        "rabitq": load("artifacts/results/full_msmarco/rabitq_full_msmarco.json"),
        "itq": load("artifacts/results/full_msmarco/itq_full_msmarco.json"),
        "identity64": load("artifacts/results/full_msmarco/identity_plus_scale_full_msmarco.json"),
        "subbit128": load("artifacts/results/full_msmarco/subbit_r128_full_msmarco.json"),
        "rand128": load("artifacts/results/full_msmarco/rand128_plus_scale_full_msmarco.json"),
        "identity128": load("artifacts/results/full_msmarco/identity128_plus_scale_full_msmarco.json"),
    }

    rows = []
    fp32_oracle = rerank100["metrics"]["fp32_oracle"]
    rows.append(
        qrow(
            "MS MARCO 100k",
            "fp32 ColBERTv2",
            rerank100_path,
            512,
            fp32_oracle["mrr@10"],
            fp32_oracle["recall@100"],
            fp32_oracle["recall@1000"],
            4,
            3,
            4,
        )
    )
    for method, source, row, bytes_per_token in [
        ("PLAID b=4", plaid_path, plaid_row("PLAID (C=32768, b=4)"), 66),
        ("PLAID b=2", plaid_path, plaid_row("PLAID (C=32768, b=2)"), 34),
        ("PLAID b=1", plaid_path, plaid_row("PLAID (C=32768, b=1)"), 18),
    ]:
        rows.append(
            qrow(
                "MS MARCO 100k",
                method,
                source,
                bytes_per_token,
                metric(row, "mrr@10"),
                metric(row, "recall@100"),
                metric(row, "recall@1000"),
                4,
                3,
                4,
            )
        )

    rows.extend(
        [
            qrow(
                "MS MARCO 100k",
                "ITQ r=64 asym",
                itq100_path,
                8,
                itq100["results"]["itq_asym"]["mrr@10"],
                itq100["results"]["itq_asym"]["recall@100"],
                itq100["results"]["itq_asym"]["recall@1000"],
                4,
                4,
                4,
            ),
            qrow(
                "MS MARCO 100k",
                "PQ 8x8 ADC",
                pq_path,
                8,
                metric(pq_prefix("PQ 8"), "mrr@10"),
                metric(pq_prefix("PQ 8"), "recall@100"),
                metric(pq_prefix("PQ 8"), "recall@1000"),
                4,
                4,
                4,
            ),
            qrow(
                "MS MARCO 100k",
                "OPQ 8x8",
                pq_path,
                8,
                metric(pq_prefix("OPQ 8"), "mrr@10"),
                metric(pq_prefix("OPQ 8"), "recall@100"),
                metric(pq_prefix("OPQ 8"), "recall@1000"),
                4,
                4,
                4,
            ),
            qrow(
                "MS MARCO 100k",
                "RaBitQ asym",
                rabitq100_path,
                24,
                rabitq100["metrics"]["mrr@10"],
                rabitq100["metrics"]["recall@100"],
                rabitq100["metrics"]["recall@1000"],
                4,
                4,
                4,
            ),
            qrow(
                "MS MARCO 100k",
                "Trained R r=64 asym",
                baseline_path,
                8,
                metric(base("SubBit r=64"), "mrr@10"),
                metric(base("SubBit r=64"), "recall@100"),
                metric(base("SubBit r=64"), "recall@1000"),
                4,
                3,
                4,
            ),
            qrow(
                "MS MARCO 100k",
                "Random orthogonal R r=64 asym",
                baseline_path,
                8,
                metric(base("random_proj r=64"), "mrr@10"),
                metric(base("random_proj r=64"), "recall@100"),
                metric(base("random_proj r=64"), "recall@1000"),
                4,
                3,
                4,
            ),
            qrow("MS MARCO 8.8M", "fp32 ColBERTv2 external reference", None, 512, 0.397, None, None, 3, 3, 3),
        ]
    )

    for key, method, bytes_per_token in [
        ("rabitq", "RaBitQ", 24),
        ("itq", "ITQ r=64", 8),
    ]:
        src = full_sources[key]
        rows.append(
            qrow(
                "MS MARCO 8.8M",
                method,
                f"artifacts/results/full_msmarco/{src['method']}_full_msmarco.json",
                bytes_per_token,
                src["metrics"]["mrr@10"],
                src["metrics"]["recall@100"],
                src["metrics"]["recall@1000"],
                3,
                3,
                3,
            )
        )

    rows.extend(
        [
            qrow(
                "MS MARCO 8.8M",
                "Trained R r=64 + scale",
                "artifacts/results/full_msmarco/bootstrap_r64_8m.json",
                8,
                boot64["metrics"]["rr@10"]["mean_b"],
                boot64["metrics"]["recall@100"]["mean_b"],
                boot64["metrics"]["recall@1000"]["mean_b"],
                4,
                3,
                3,
            ),
            qrow(
                "MS MARCO 8.8M",
                "Random orthogonal R r=64 + scale",
                "artifacts/results/full_msmarco/bootstrap_r64_8m.json",
                8,
                boot64["metrics"]["rr@10"]["mean_a"],
                boot64["metrics"]["recall@100"]["mean_a"],
                boot64["metrics"]["recall@1000"]["mean_a"],
                4,
                3,
                3,
            ),
            qrow(
                "MS MARCO 8.8M",
                "Identity R r=64 + scale",
                "artifacts/results/full_msmarco/identity_plus_scale_full_msmarco.json",
                8,
                full_sources["identity64"]["metrics"]["mrr@10"],
                full_sources["identity64"]["metrics"]["recall@100"],
                full_sources["identity64"]["metrics"]["recall@1000"],
                4,
                3,
                3,
            ),
        ]
    )

    for key, method in [
        ("subbit128", "Trained R r=128 + scale"),
        ("rand128", "Random orthogonal R r=128 + scale"),
        ("identity128", "Identity R r=128 + scale"),
    ]:
        src = full_sources[key]
        rows.append(
            qrow(
                "MS MARCO 8.8M",
                method,
                f"artifacts/results/full_msmarco/{src['method']}_full_msmarco.json",
                16,
                src["metrics"]["mrr@10"],
                src["metrics"]["recall@100"],
                src["metrics"]["recall@1000"],
                4,
                3,
                3,
            )
        )

    payload = common(
        "tab:pareto and fig:pareto",
        [
            baseline_path,
            plaid_path,
            itq100_path,
            pq_path,
            rabitq100_path,
            rerank100_path,
            "artifacts/results/full_msmarco/bootstrap_r64_8m.json",
            "artifacts/results/full_msmarco/rabitq_full_msmarco.json",
            "artifacts/results/full_msmarco/itq_full_msmarco.json",
            "artifacts/results/full_msmarco/identity_plus_scale_full_msmarco.json",
            "artifacts/results/full_msmarco/subbit_r128_full_msmarco.json",
            "artifacts/results/full_msmarco/rand128_plus_scale_full_msmarco.json",
            "artifacts/results/full_msmarco/identity128_plus_scale_full_msmarco.json",
        ],
    )
    payload["rows"] = rows
    payload["bootstrap"] = {
        "r64_trained_minus_random": {
            "source_file": "artifacts/results/full_msmarco/bootstrap_r64_8m.json",
            "delta_mrr@10": f(boot64["metrics"]["rr@10"]["point"], 4),
            "ci95": [
                f(boot64["metrics"]["rr@10"]["ci_lo"], 4),
                f(boot64["metrics"]["rr@10"]["ci_hi"], 4),
            ],
        }
    }
    dump("table_01_msmarco_storage_quality.json", payload)


def build_beir() -> None:
    ci_path = "artifacts/results/beir_ci/beir9_r64_ci.json"
    ci = load(ci_path)
    rows = []
    sign_ret = []
    rand_ret = []
    for crow in ci["rows"]:
        corpus = crow["corpus"]
        detail_path = f"artifacts/results/beir_ci/beir_{corpus}_sign_vs_random_r64.json"
        detail = load(detail_path)
        methods = detail["methods"]
        fp32 = methods.get("fp32_colbertv2", {}).get("ndcg@10")
        sign = methods["sign_d_r64"]["ndcg@10"]
        rand = methods["random_R_r64_sign"]["ndcg@10"]
        if fp32 is not None:
            sign_ret.append(sign / fp32)
            rand_ret.append(rand / fp32)
        rows.append(
            {
                "corpus": corpus,
                "source_file": detail_path,
                "n_queries": value(crow["n_q"], str(crow["n_q"])),
                "fp32_ndcg@10": None if fp32 is None else f(fp32, 4),
                "sign_ndcg@10": f(sign, 4),
                "random_ndcg@10": f(rand, 4),
                "delta_sign_minus_random": sf(crow["delta"], 3),
                "retention_vs_fp32": None if fp32 is None else pct(sign / fp32, 1),
                "ci95": [f(crow["ci_lo"], 4), f(crow["ci_hi"], 4)],
                "p_holm": f(crow["p_holm"], 3) if crow["p_holm"] < 0.01 else f(crow["p_holm"], 1),
            }
        )
    payload = common("tab:beir", [ci_path])
    payload.update(
        {
            "metric": "ndcg@10",
            "comparison": ci["comparison"],
            "rows": rows,
            "summary": {
                "median_abs_delta": f(ci["median_abs_delta"], 3),
                "mean_sign_retention_vs_fp32": pct(sum(sign_ret) / len(sign_ret), 1),
                "mean_random_retention_vs_fp32": pct(sum(rand_ret) / len(rand_ret), 1),
                "holm_significant_corpora": value(ci["n_sig_holm_05"], str(ci["n_sig_holm_05"])),
            },
        }
    )
    dump("table_02_beir_ndcg.json", payload)


def build_preservation_and_levels() -> None:
    paths = [
        "artifacts/results/aug_eval/preservation_aug.json",
        "artifacts/results/aug_eval/preservation_aug_pca_identity.json",
        "artifacts/results/aug_eval/preservation_aug_rest.json",
    ]
    rows_by_label: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in load(path)["rows"]:
            rows_by_label[row["label"]] = row

    def pres_row(method: str, label: str, source_file: str | None = None) -> dict[str, Any]:
        if label == "fp32":
            return {
                "method": method,
                "source_file": None,
                "pearson": f(1.0, 3),
                "spearman_rho": f(1.0, 3),
                "kendall_tau": f(1.0, 3),
                "overlap@10": f(1.0, 3),
                "overlap@100": f(1.0, 3),
                "overlap@1000": f(1.0, 3),
            }
        row = rows_by_label[label]
        return {
            "method": method,
            "source_file": source_file,
            "pearson": f(row["pearson_flat"], 3),
            "spearman_rho": f(row["spearman_mean"], 3),
            "kendall_tau": f(row["kendall_mean"], 3),
            "overlap@10": f(row["overlap_at_k"]["10"], 3),
            "overlap@100": f(row["overlap_at_k"]["100"], 3),
            "overlap@1000": f(row["overlap_at_k"]["1000"], 3),
        }

    payload = common("tab:preservation", paths)
    payload["rows"] = [
        pres_row("fp32", "fp32"),
        pres_row("Trained asym", "learned_asymmetric", paths[0]),
        pres_row("PCA", "pca_projection", paths[1]),
        pres_row("Random orthogonal", "random_projection", paths[2]),
        pres_row("Identity", "identity_truncation", paths[1]),
    ]
    dump("table_03_rank_preservation.json", payload)

    geometry_path = "artifacts/results/diagnostics/geometry_fidelity_vs_argmax.json"
    rec_path = "artifacts/results/aug_eval/rec_at_10_trained_aug.json"
    worst_path = "artifacts/results/aug_eval/worst5_fp32_aug100k.json"
    geometry = load(geometry_path)
    rec = load(rec_path)
    worst = load(worst_path)
    levels = common("fig:levels", [geometry_path, rec_path, worst_path] + paths)
    levels.update(
        {
            "token_neighbor_overlap": {
                "k10": pct(geometry["neighbor_overlap"]["10"]["mean"], 1),
                "k1000": pct(geometry["neighbor_overlap"]["1000"]["mean"], 1),
                "random_baseline_k1000": pct(geometry["neighbor_overlap"]["1000"]["random_baseline"], 1),
            },
            "score_correlation": {
                "pearson": f(geometry["correlation"]["pearson_r"], 3),
                "spearman_rho": f(geometry["correlation"]["spearman_rho"], 3),
            },
            "relevant_document_recall_at_10": pct(rec["value"], 1),
            "worst_5_percent_queries": {
                "n_queries": value(worst["n_worst"], str(worst["n_worst"])),
                "mean_rank_shift": f(worst["worst_mean_shift_subbit_minus_fp32"], 3),
                "still_top10": value(worst["worst_still_in_top10"], str(worst["worst_still_in_top10"])),
                "still_top10_percent": pct_from_percent(worst["worst_still_in_top10_pct"], 1),
                "dropped": value(worst["worst_dropped_out"], str(worst["worst_dropped_out"])),
            },
        }
    )
    dump("figure_01_three_levels.json", levels)


def build_mechanism() -> None:
    decomp_path = "artifacts/results/diagnostics/c6_decomposition_aug100k_r64.json"
    rank_path = "artifacts/results/diagnostics/c6_rank_landing_aug100k_r64.json"
    decomp = load(decomp_path)
    rank = load(rank_path)
    low_survive = decomp["sign_preservation_at_r64"]["low_margin_preservation_rate"]
    rel_loss = decomp["flip_residuals"]["per_query_aggregate"]["relative_to_fp32_sum"]
    payload = common("fig:argmax, tab:ranklanding, and fig:residual", [decomp_path, rank_path])
    payload.update(
        {
            "argmax_flip": {
                "low_margin_flip_rate": pct(1.0 - low_survive, 2),
                "low_margin_survival_rate": pct(low_survive, 2),
                "high_margin_tail": pct(decomp["population"]["pooled_n_high_frac"], 2),
            },
            "rank_landing": {
                "n_low_margin_flipped": value(rank["n_low_margin_flipped"], f"{rank['n_low_margin_flipped']:,}"),
                "rank_2": pct_from_percent(rank["rank_landing"]["rank_2_pct"], 1),
                "rank_3_5": pct_from_percent(rank["rank_landing"]["rank_3_5_pct"], 1),
                "rank_6_10": pct_from_percent(rank["rank_landing"]["rank_6_10_pct"], 1),
                "rank_gt10": pct_from_percent(rank["rank_landing"]["rank_gt10_pct"], 1),
                "median_rank": value(rank["rank_landing"]["median_rank"], str(rank["rank_landing"]["median_rank"])),
            },
            "residual": {
                "median_delta_fp32": f(rank["residual_among_same"]["median_delta_fp32"], 4),
                "frac_within_0_05": pct(rank["residual_among_same"]["frac_within_0_05"], 1),
                "per_query_median_aggregate_loss": pct(abs(rel_loss["median"]), 1),
                "per_query_losing_at_most_10_percent": pct(rel_loss["frac_ge_minus_0_10"], 1),
            },
        }
    )
    dump("figure_02_argmax_mechanism.json", payload)


def build_training() -> None:
    sec8_path = "artifacts/results/aug_eval/sec8_figure_data.json"
    drho_path = "artifacts/results/aug_eval/drho_vs_dmrr_q1500.json"
    rdeep_path = "artifacts/results/diagnostics/r_deep_dive.json"
    sec8 = load(sec8_path)
    drho = load(drho_path)
    rdeep = load(rdeep_path)
    payload = common("fig:training", [sec8_path, drho_path, rdeep_path])
    payload.update(
        {
            "fidelity_bars": {
                "spearman_rho_random": f(sec8["fidelity_bars"]["metric_rho"]["random"], 3),
                "spearman_rho_trained": f(sec8["fidelity_bars"]["metric_rho"]["trained"], 3),
                "mrr_random": f(sec8["fidelity_bars"]["metric_mrr"]["random"], 4),
                "mrr_trained": f(sec8["fidelity_bars"]["metric_mrr"]["trained"], 4),
            },
            "training_budget": {
                point["triples"]: {
                    "spearman_rho": f(point["rho"], 4),
                    "mrr@10": f(point["mrr"], 4),
                }
                for point in sec8["budget"]["points"]
            },
            "scatter": {
                "n_queries": value(drho["n"], str(drho["n"])),
                "pearson_r": f(drho["pearson_drho_dmrr"], 3),
                "pearson_p": f(drho["pearson_p"], 2),
                "gain_queries_mean_abs_delta_mrr": f(drho["mean_drho_mrr_gain"], 3),
                "loss_queries_mean_abs_delta_mrr": f(drho["mean_drho_mrr_lose"], 3),
            },
            "scale_gain": {
                item["corpus"]: {
                    "delta_mrr": f(item["dmrr"], 4),
                    "ci95": [f(item["ci"][0], 4), f(item["ci"][1], 4)],
                }
                for item in sec8["scale_gain"]
            },
            "geometry_checks": {
                "trained_R_mean_angle_to_pca_deg": f(
                    rdeep["diag1_cross_checkpoint_colbert"]["pair_angles"]["canonical_50k_topk__init_pca"]["mean_deg"],
                    0,
                ),
                "relevant_mean_hyperplane_distance": f(
                    rdeep["diag5_hyperplane_saturation"]["canonical"]["relevant"]["global_mean_margin"],
                    5,
                ),
                "non_relevant_mean_hyperplane_distance": f(
                    rdeep["diag5_hyperplane_saturation"]["canonical"]["non_relevant"]["global_mean_margin"],
                    5,
                ),
            },
        }
    )
    dump("figure_03_training_anatomy.json", payload)


def build_rerank_and_latency() -> None:
    baseline_path = "artifacts/results/aug_eval/baseline_100k_aug_r64.json"
    rerank_path = "artifacts/results/aug_eval/rerank_aug_fullfp32.json"
    rand_rerank_path = "artifacts/results/aug_eval/rerank_aug_fullfp32_randomR.json"
    baseline = load(baseline_path)
    rerank = load(rerank_path)
    rand_rerank = load(rand_rerank_path)
    trained = row_by_label(baseline["rows"], "SubBit r=64")
    fp32 = rerank["metrics"]["fp32_oracle"]
    k100 = rerank["metrics"]["two_stage"]["100"]
    payload = common("tab:rerank", [baseline_path, rerank_path, rand_rerank_path])
    payload["rows"] = [
        {
            "method": "fp32 MaxSim only",
            "mrr@10": f(fp32["mrr@10"], 4),
            "recall@100": f(fp32["recall@100"], 4),
            "recall@1000": f(fp32["recall@1000"], 4),
        },
        {
            "method": "Sign-coded single stage",
            "mrr@10": f(metric(trained, "mrr@10"), 4),
            "recall@100": f(metric(trained, "recall@100"), 4),
            "recall@1000": f(metric(trained, "recall@1000"), 4),
        },
        {
            "method": "Sign-coded -> fp32 rerank K=100",
            "mrr@10": f(k100["mrr@10"], 4),
            "recall@100": f(k100["recall@100"], 4),
            "recall@1000": f(k100["recall@1000"], 4),
        },
    ]
    payload["random_R_k100_mrr@10"] = f(rand_rerank["metrics"]["two_stage"]["100"]["mrr@10"], 4)
    payload["gap_to_fp32_after_k100"] = f(k100["mrr@10"] - fp32["mrr@10"], 4)
    dump("table_04_two_stage_retrieval.json", payload)

    latency_path = "artifacts/results/latency/latency_interleaved_rerank.json"
    lat = load(latency_path)["summary"]
    payload = common("tab:rerank-latency", [latency_path])
    payload["rows"] = [
        {
            "K": value(100, "100"),
            "median_ms": f(lat["stage2 fp32 rerank K=100"]["median_of_round_medians_ms"], 1),
            "iqr_ms": f(lat["stage2 fp32 rerank K=100"]["iqr_ms"], 2),
        },
        {
            "K": value(256, "256"),
            "median_ms": f(lat["stage2 fp32 rerank K=256"]["median_of_round_medians_ms"], 1),
            "iqr_ms": f(lat["stage2 fp32 rerank K=256"]["iqr_ms"], 2),
        },
        {
            "K": value(1024, "1024"),
            "median_ms": f(lat["stage2 fp32 rerank K=1024"]["median_of_round_medians_ms"], 1),
            "iqr_ms": f(lat["stage2 fp32 rerank K=1024"]["iqr_ms"], 2),
        },
    ]
    dump("table_05_rerank_latency.json", payload)


def build_plaid_and_e2e_latency() -> None:
    plaid_path = "artifacts/results/aug_eval/table_plaid_noscale_aug.json"
    random_rerank_path = "artifacts/results/aug_eval/rerank_aug_fullfp32_randomR.json"
    trained_rerank_path = "artifacts/results/aug_eval/rerank_aug_fullfp32.json"
    plaid = load(plaid_path)
    random_rerank = load(random_rerank_path)
    trained_rerank = load(trained_rerank_path)

    def prow(label: str) -> dict[str, Any]:
        return row_by_label(plaid["rows"], label)

    rows = []
    fp32 = trained_rerank["metrics"]["fp32_oracle"]
    rows.append(
        {
            "method": "fp32",
            "bytes_per_token": value(512, "512"),
            "mrr@10": f(fp32["mrr@10"], 4),
            "ndcg@10": f(fp32["ndcg@10"], 4),
        }
    )
    for method, label, bytes_per_token in [
        ("PLAID b=4", "PLAID (C=32768, b=4)", 66),
        ("PLAID b=2", "PLAID (C=32768, b=2)", 34),
        ("PLAID b=1", "PLAID (C=32768, b=1)", 18),
        ("Random R r=64", "random_proj r=64", 8),
    ]:
        row = prow(label)
        rows.append(
            {
                "method": method,
                "bytes_per_token": value(bytes_per_token, str(bytes_per_token)),
                "mrr@10": f(metric(row, "mrr@10"), 4),
                "ndcg@10": f(metric(row, "ndcg@10"), 4),
            }
        )
    rows.append(
        {
            "method": "Random R r=64 + fp32 rerank K=100",
            "bytes_per_token": value(8, "8"),
            "mrr@10": f(random_rerank["metrics"]["two_stage"]["100"]["mrr@10"], 4),
            "ndcg@10": None,
        }
    )
    for method, label in [
        ("Trained R no scale", "SubBit r=64 (no scale)"),
        ("Identity sign", "identity r=64"),
    ]:
        row = prow(label)
        rows.append(
            {
                "method": method,
                "bytes_per_token": value(8, "8"),
                "mrr@10": f(metric(row, "mrr@10"), 4),
                "ndcg@10": f(metric(row, "ndcg@10"), 4),
            }
        )
    payload = common("tab:plaid", [plaid_path, random_rerank_path, trained_rerank_path])
    payload["rows"] = rows
    payload["trained_R_k100_mrr@10"] = f(trained_rerank["metrics"]["two_stage"]["100"]["mrr@10"], 4)
    dump("table_06_plaid_storage_quality.json", payload)

    latency_path = "artifacts/results/latency/latency_interleaved_table.json"
    engine_path = "artifacts/results/latency/plaid_official_engine.json"
    lat = load(latency_path)["summary"]
    engine = load(engine_path)
    payload = common("tab:e2e-latency", [latency_path, engine_path])
    payload["rows"] = [
        {
            "method": "Sign-coded exhaustive",
            "bytes_per_token": value(8, "8"),
            "median_ms": f(lat["sign-coded r=64"]["median_of_round_medians_ms"], 1),
            "iqr_ms": f(lat["sign-coded r=64"]["iqr_ms"], 1),
        },
        {
            "method": "fp32 exhaustive",
            "bytes_per_token": value(512, "512"),
            "median_ms": f(lat["FP128"]["median_of_round_medians_ms"], 1),
            "iqr_ms": f(lat["FP128"]["iqr_ms"], 1),
        },
        {
            "method": "PLAID engine official b=2",
            "bytes_per_token": value(34, "approx 34"),
            "median_ms": f(engine["summary"]["engine_median_of_round_medians_ms"], 1),
            "iqr_ms": f(engine["summary"]["engine_iqr_ms"], 1),
            "mrr@10": f(engine["quality"]["mrr@10"], 4),
            "max_docs": value(4096, "4096"),
        },
    ]
    payload["sign_coded_vs_fp32_latency_ratio"] = f(
        lat["FP128"]["median_of_round_medians_ms"] / lat["sign-coded r=64"]["median_of_round_medians_ms"],
        2,
    )
    dump("table_07_retrieval_latency.json", payload)


def build_prose_claims() -> None:
    two16_path = "artifacts/results/diagnostics/two_stage_r16_fast.json"
    two32_path = "artifacts/results/diagnostics/two_stage_r32_fast.json"
    colbert_path = "artifacts/results/diagnostics/sign_only_canonical_colbertv2.json"
    const_path = "artifacts/results/diagnostics/bootstrap_per_config_effects_constbert_v2.json"
    two16 = load(two16_path)
    two32 = load(two32_path)
    colbert = load(colbert_path)
    const = load(const_path)
    colbert_fp32 = row_by_label(colbert["rows"], "FP128")["metrics"]["mrr@10"]
    const_fp32 = const["per_config_mrr10"]["fp32"]
    payload = common("low-rank ConstBERT control prose", [two16_path, two32_path, colbert_path, const_path])
    payload["colbertv2"] = {
        "fp32_mrr@10": f(colbert_fp32, 4),
        "r32_mrr@10": f(two32["single_stage"]["mrr@10"], 3),
        "r32_retention": pct(two32["single_stage"]["mrr@10"] / colbert_fp32, 1),
        "r16_mrr@10": f(two16["single_stage"]["mrr@10"], 3),
        "r16_retention": pct(two16["single_stage"]["mrr@10"] / colbert_fp32, 1),
    }
    payload["constbert32"] = {
        "fp32_mrr@10": f(const_fp32, 4),
        "r32_mrr@10": f(const["per_config_mrr10"]["r32_random_sign"], 3),
        "r32_retention": pct(const["per_config_mrr10"]["r32_random_sign"] / const_fp32, 1),
        "r16_mrr@10": f(const["per_config_mrr10"]["r16_random_sign"], 3),
        "r16_retention": pct(const["per_config_mrr10"]["r16_random_sign"] / const_fp32, 1),
    }
    dump("prose_01_low_rank_controls.json", payload)

    full_rand_path = "artifacts/results/full_msmarco/rerank_8m_rand_plus_scale_K_sweep.json"
    full_trained_path = "artifacts/results/full_msmarco/rerank_8m_subbit_K_sweep.json"
    boot_path = "artifacts/results/full_msmarco/bootstrap_rerank_8m.json"
    miss_path = "artifacts/results/aug_eval/missed_resolved_margin_aug100k.json"
    full_rand = load(full_rand_path)
    full_trained = load(full_trained_path)
    boot = load(boot_path)
    missed = load(miss_path)
    payload = common("full-scale rerank and residual-miss prose", [full_rand_path, full_trained_path, boot_path, miss_path])
    payload.update(
        {
            "full_scale_k100": {
                "random_R_mrr@10": f(full_rand["K_sweep"]["100"]["mrr@10"], 4),
                "trained_R_mrr@10": f(full_trained["K_sweep"]["100"]["mrr@10"], 4),
                "delta_trained_minus_random": f(boot["by_K"]["100"]["point"], 4),
                "ci95": [f(boot["by_K"]["100"]["ci_lo"], 4), f(boot["by_K"]["100"]["ci_hi"], 4)],
                "n_queries": value(boot["by_K"]["100"]["n_queries"], f"{boot['by_K']['100']['n_queries']:,}"),
            },
            "full_scale_k1000": {
                "random_R_mrr@10": f(full_rand["K_sweep"]["1000"]["mrr@10"], 4),
                "trained_R_mrr@10": f(full_trained["K_sweep"]["1000"]["mrr@10"], 4),
                "delta_trained_minus_random": f(boot["by_K"]["1000"]["point"], 4),
                "ci95": [f(boot["by_K"]["1000"]["ci_lo"], 4), f(boot["by_K"]["1000"]["ci_hi"], 4)],
            },
            "k100_residual_misses": {
                "n_missed": value(missed["n_missed"], str(missed["n_missed"])),
                "n_total": value(6980, "6,980"),
                "missed_median_margin": f(missed["missed_median_margin"], 2),
                "resolved_median_margin": f(missed["resolved_median_margin"], 2),
                "resolved_to_missed_margin_ratio": f(
                    missed["resolved_median_margin"] / missed["missed_median_margin"], 1
                ),
            },
        }
    )
    dump("prose_02_full_scale_rerank.json", payload)

    rot_path = "artifacts/results/full_msmarco/bootstrap_r128_8m.json"
    sub128_path = "artifacts/results/full_msmarco/subbit_r128_full_msmarco.json"
    rand128_path = "artifacts/results/full_msmarco/rand128_plus_scale_full_msmarco.json"
    ident128_path = "artifacts/results/full_msmarco/identity128_plus_scale_full_msmarco.json"
    rot = load(rot_path)
    sub128 = load(sub128_path)
    rand128 = load(rand128_path)
    ident128 = load(ident128_path)
    payload = common("r=128 full-scale rotation prose", [rot_path, sub128_path, rand128_path, ident128_path])
    payload.update(
        {
            "trained_vs_random_plus_scale_point_estimate": {
                "trained_R_mrr@10": f(sub128["metrics"]["mrr@10"], 4),
                "random_R_mrr@10": f(rand128["metrics"]["mrr@10"], 4),
                "delta_trained_minus_random": f(
                    sub128["metrics"]["mrr@10"] - rand128["metrics"]["mrr@10"],
                    4,
                ),
            },
            "identity_vs_random_rotation_bootstrap": {
                "identity_plus_scale_table_mrr@10": f(ident128["metrics"]["mrr@10"], 4),
                "random_plus_scale_table_mrr@10": f(rand128["metrics"]["mrr@10"], 4),
                "sign_d_no_rotation_bootstrap_mean_mrr@10": f(rot["metrics"]["rr@10"]["mean_b"], 4),
                "random_R_bootstrap_mean_mrr@10": f(rot["metrics"]["rr@10"]["mean_a"], 4),
                "bootstrap_delta_sign_d_minus_random": sf(rot["metrics"]["rr@10"]["point"], 4),
                "ci95": [
                    f(rot["metrics"]["rr@10"]["ci_lo"], 4),
                    sf(rot["metrics"]["rr@10"]["ci_hi"], 4),
                ],
            },
        }
    )
    dump("prose_03_r128_rotation_control.json", payload)

    payload = common("PLAID storage-accounting prose", ["paper/paper_reneuir.tex"])
    payload["values"] = {
        "plaid_codebooks_mib": value(16, "16"),
        "msmarco_100k_tokens_million": value(6.73, "6.73"),
        "codebook_bytes_per_token_100k": f(16 * 1024 * 1024 / (6.73 * 1_000_000), 1),
        "plaid_b2_bare_bytes_per_token": value(34, "34"),
        "plaid_b2_amortized_bytes_per_token_100k": f(34 + 16 * 1024 * 1024 / (6.73 * 1_000_000), 1),
        "ratio_vs_8_btok_at_100k": f((34 + 16 * 1024 * 1024 / (6.73 * 1_000_000)) / 8, 1),
        "msmarco_8m_tokens_million": value(597, "597"),
        "codebook_bytes_per_token_8m": f(16 * 1024 * 1024 / (597 * 1_000_000), 2),
        "bare_ratio_vs_8_btok": f(34 / 8, 2),
    }
    dump("prose_04_storage_accounting.json", payload)


def build_all() -> None:
    build_storage_quality()
    build_beir()
    build_preservation_and_levels()
    build_mechanism()
    build_training()
    build_rerank_and_latency()
    build_plaid_and_e2e_latency()
    build_prose_claims()


def sync_output_links() -> None:
    OUTPUT_LINKS.mkdir(parents=True, exist_ok=True)
    for target in sorted(PAPER_RESULTS.glob("*.json")):
        link = OUTPUT_LINKS / target.name
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(os.path.relpath(target, link.parent))


def verify() -> None:
    files = sorted(PAPER_RESULTS.glob("*.json"))
    if not files:
        raise SystemExit("no paper JSON files found")
    for path in files:
        with path.open() as f:
            json.load(f)
    missing_links = [path.name for path in files if not (OUTPUT_LINKS / path.name).exists()]
    if missing_links:
        raise SystemExit(f"missing output links: {missing_links}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if not args.verify_only:
        build_all()
        sync_output_links()
    verify()
    print(f"paper JSON outputs ready: {len(list(PAPER_RESULTS.glob('*.json')))} files")


if __name__ == "__main__":
    main()
