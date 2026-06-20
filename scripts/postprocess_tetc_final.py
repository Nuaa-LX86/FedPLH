from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulator.acf_simulator import ACFSimulator


CORE_SCENARIOS = (
    "FP32_noDP",
    "FP32_softDP",
    "FedBN",
    "FedPAQ",
    "Mao_etal",
    "BitFusion",
    "HMPE-ACF_noDP",
    "HMPE-ACF",
)
ABLATION_SCENARIOS = (
    "ACF_static_FP8",
    "ACF_progress_only",
    "ACF_entropy_only",
)
STRESS_SCENARIOS = ("ACF_FedBN", "ACF_progress_only", "ACF_full")
PAPER_METRICS = (
    "test_dice",
    "avg_latency_ms",
    "avg_local_training_energy_mJ",
    "avg_compute_latency_ms",
    "avg_dp_overhead_ms",
    "avg_dp_total_ms",
    "avg_dp_background_ms",
    "avg_comm_latency_ms",
    "avg_agg_latency_ms",
    "avg_misc_latency_ms",
    "avg_dp_hidden_ratio",
    "avg_delta_c_cycles",
    "avg_c_priv_cycles",
    "final_epsilon",
)
EVIDENCE_METRICS = (
    "test_dice",
    "avg_latency_ms",
    "avg_local_training_energy_mJ",
    "final_epsilon",
    "high_precision_assignment_rate",
)
T_CRITICAL_975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
}
UPDATE_BYTES = 1_402_612 * 4
UPDATE_MIB = UPDATE_BYTES / (1024.0**2)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sample_stats(values: Iterable[float], use_t: bool = False) -> Dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "ci95_half_width": None,
            "values": [],
        }
    std = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    critical = (
        T_CRITICAL_975.get(int(array.size) - 1, 1.96)
        if use_t
        else 1.96
    )
    ci95 = (
        float(critical * std / math.sqrt(array.size))
        if array.size > 1
        else 0.0
    )
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "std": std,
        "ci95_half_width": ci95,
        "values": [float(value) for value in array],
    }


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def spearman(first: Iterable[float], second: Iterable[float | None]) -> float | None:
    first_array = np.asarray(list(first), dtype=np.float64)
    second_array = np.asarray(
        [np.nan if value is None else value for value in second],
        dtype=np.float64,
    )
    mask = np.isfinite(first_array) & np.isfinite(second_array)
    if int(mask.sum()) < 2:
        return None
    first_ranks = rankdata(first_array[mask])
    second_ranks = rankdata(second_array[mask])
    if float(first_ranks.std()) == 0.0 or float(second_ranks.std()) == 0.0:
        return None
    value = float(np.corrcoef(first_ranks, second_ranks)[0, 1])
    return value if np.isfinite(value) else None


def discover_runs(source_root: Path) -> List[Dict[str, Any]]:
    specifications: List[tuple[str, int, str, int, Path]] = []
    for scenario in CORE_SCENARIOS:
        for seed in range(5):
            specifications.append(
                (
                    "core",
                    0,
                    scenario,
                    seed,
                    source_root / "core" / "unet" / scenario / f"seed{seed}",
                )
            )
    for scenario in ABLATION_SCENARIOS:
        for seed in range(3):
            specifications.append(
                (
                    "ablation",
                    0,
                    scenario,
                    seed,
                    source_root / "ablation" / "unet" / scenario / f"seed{seed}",
                )
            )
    for partition_id in range(3):
        for scenario in STRESS_SCENARIOS:
            for seed in range(3):
                specifications.append(
                    (
                        "stress",
                        partition_id,
                        scenario,
                        seed,
                        source_root
                        / "stress"
                        / f"partition{partition_id}"
                        / "unet"
                        / scenario
                        / f"seed{seed}",
                    )
                )

    if len(specifications) != 76:
        raise AssertionError(f"Expected 76 run specifications, found {len(specifications)}")

    records = []
    for stage, partition_id, scenario, seed, run_dir in specifications:
        history_path = run_dir / "training_history.json"
        manifest_path = run_dir / "run_manifest.json"
        history = load_json(history_path)
        manifest = load_json(manifest_path)
        if manifest.get("status") != "completed":
            raise ValueError(f"Incomplete run: {run_dir}")
        if len(history.get("round", [])) != 80:
            raise ValueError(f"Expected 80 rounds: {history_path}")
        records.append(
            {
                "stage": stage,
                "partition_id": partition_id,
                "scenario": scenario,
                "seed": seed,
                "run_dir": run_dir,
                "history_path": history_path,
                "manifest_path": manifest_path,
                "history": history,
                "manifest": manifest,
            }
        )
    return records


def fingerprint_inputs(records: List[Dict[str, Any]], source_root: Path) -> Dict[str, str]:
    paths = sorted(
        {
            record[key]
            for record in records
            for key in ("history_path", "manifest_path")
        }
    )
    return {
        path.relative_to(source_root).as_posix(): sha256_file(path)
        for path in paths
    }


def payload_ratio(history: Dict[str, Any]) -> float:
    levels = history.get("metrics", {}).get("fedpaq_quantization_levels")
    if levels is None:
        return 1.0
    payload_bits = math.ceil(math.log2(int(levels) + 1)) + 1
    return payload_bits / 32.0


def clean_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = copy.deepcopy(metrics)
    cleaned.pop("avg_energy_mJ", None)
    cleaned.pop("total_energy_mJ", None)
    return cleaned


def derive_run(
    record: Dict[str, Any],
    simulator: ACFSimulator,
) -> Dict[str, Any]:
    history = record["history"]
    manifest = record["manifest"]
    rounds = len(history["round"])
    required = (
        "compute_latency_ms",
        "dp_overhead_ms",
        "comm_latency_ms",
        "misc_latency_ms",
        "participating_clients",
        "is_aggregation_round",
    )
    for key in required:
        if len(history.get(key, [])) != rounds:
            raise ValueError(f"{record['history_path']}: invalid {key}")

    use_sac = bool(
        manifest.get("scenario_config", {})
        .get("dp", {})
        .get("simulate_hardware_pec", False)
    )
    method = "PEC" if use_sac else "Software"
    ratio = payload_ratio(history)
    model_size_mib = UPDATE_MIB * ratio
    aggregation = []
    latency = []
    residuals = []
    for index in range(rounds):
        is_aggregation_round = bool(history["is_aggregation_round"][index])
        participants = len(history["participating_clients"][index])
        agg_ms = (
            simulator.simulate_aggregation(
                participants,
                model_size_mib,
                method,
            )
            if is_aggregation_round
            else 0.0
        )
        round_ms = (
            float(history["compute_latency_ms"][index])
            + float(history["dp_overhead_ms"][index])
            + float(history["comm_latency_ms"][index])
            + agg_ms
            + float(history["misc_latency_ms"][index])
        )
        aggregation.append(float(agg_ms))
        latency.append(float(round_ms))
        residuals.append(
            float(
                round_ms
                - (
                    float(history["compute_latency_ms"][index])
                    + float(history["dp_overhead_ms"][index])
                    + float(history["comm_latency_ms"][index])
                    + agg_ms
                    + float(history["misc_latency_ms"][index])
                )
            )
        )

    max_residual = float(np.max(np.abs(residuals)))
    if max_residual > 1e-6:
        raise ValueError(f"Latency closure failed: {record['history_path']}")

    metrics = clean_metrics(history["metrics"])
    metrics["avg_agg_latency_ms"] = float(np.mean(aggregation))
    metrics["avg_latency_ms"] = float(np.mean(latency))
    metrics["total_time_ms"] = float(np.sum(latency))
    metrics["max_abs_latency_residual_ms"] = max_residual
    if metrics.get("t2a_round") is not None:
        target_round = int(metrics["t2a_round"])
        metrics["t2a_ms"] = float(np.sum(latency[: target_round + 1]))
    if "avg_local_training_energy_mJ" not in metrics:
        legacy = history["metrics"].get("avg_energy_mJ")
        if legacy is None:
            raise ValueError(f"Missing local-training energy: {record['history_path']}")
        metrics["avg_local_training_energy_mJ"] = float(legacy)

    return {
        **record,
        "derived": {
            "schema_version": 1,
            "source": {
                "training_history": str(record["history_path"].resolve()),
                "training_history_sha256": sha256_file(record["history_path"]),
                "run_manifest": str(record["manifest_path"].resolve()),
                "run_manifest_sha256": sha256_file(record["manifest_path"]),
            },
            "stage": record["stage"],
            "partition_id": record["partition_id"],
            "scenario": record["scenario"],
            "seed": record["seed"],
            "aggregation_model": {
                "method": "SAC" if use_sac else "CPU",
                "participants_symbol": "M_r",
                "update_bytes_fp32": UPDATE_BYTES,
                "update_size_mib_fp32": UPDATE_MIB,
                "payload_ratio": ratio,
                "modeled_payload_mib": model_size_mib,
                "communication_excluded": True,
            },
            "round": copy.deepcopy(history["round"]),
            "agg_latency_ms": aggregation,
            "latency_ms": latency,
            "latency_residual_ms": residuals,
            "metrics": metrics,
        },
    }


def derived_output_path(output_root: Path, record: Dict[str, Any]) -> Path:
    if record["stage"] == "stress":
        relative = (
            Path("derived_runs")
            / "stress"
            / f"partition{record['partition_id']}"
            / record["scenario"]
            / f"seed{record['seed']}"
        )
    else:
        relative = (
            Path("derived_runs")
            / record["stage"]
            / record["scenario"]
            / f"seed{record['seed']}"
        )
    return output_root / relative / "derived_metrics.json"


def result_metrics(record: Dict[str, Any]) -> Dict[str, Any]:
    return record["derived"]["metrics"]


def build_paper_results(records: List[Dict[str, Any]], metadata: Dict[str, Any]) -> Dict[str, Any]:
    core = [record for record in records if record["stage"] == "core"]
    indexed = {
        (record["scenario"], record["seed"]): record
        for record in core
    }
    baseline = "FedBN"
    results: Dict[str, Any] = {}
    for scenario in CORE_SCENARIOS:
        selected = [
            indexed[(scenario, seed)]
            for seed in range(5)
        ]
        metrics: Dict[str, float] = {}
        metrics_std: Dict[str, float] = {}
        metrics_ci95: Dict[str, float] = {}
        for metric in PAPER_METRICS:
            values = [
                float(result_metrics(record)[metric])
                for record in selected
                if result_metrics(record).get(metric) is not None
            ]
            if values:
                stats = sample_stats(values)
                metrics[metric] = stats["mean"]
                metrics_std[metric] = stats["std"]
                metrics_ci95[metric] = stats["ci95_half_width"]
        normalized = {}
        for metric in ("avg_latency_ms", "avg_local_training_energy_mJ"):
            ratios = [
                float(result_metrics(indexed[(scenario, seed)])[metric])
                / float(result_metrics(indexed[(baseline, seed)])[metric])
                for seed in range(5)
            ]
            normalized[metric] = sample_stats(ratios)
        results[scenario] = {
            "metrics": metrics,
            "metrics_std": metrics_std,
            "metrics_ci95": metrics_ci95,
            "normalized": normalized,
            "seeds": {
                str(record["seed"]): {
                    metric: result_metrics(record).get(metric)
                    for metric in PAPER_METRICS
                }
                for record in selected
            },
            "scenario_config": copy.deepcopy(selected[0]["manifest"]["scenario_config"]),
        }
    return {
        "schema_version": 2,
        "postprocessing": metadata,
        "baseline": baseline,
        "normalization": "paired by training seed, then summarized",
        "error_bars": "sample standard deviation across five training seeds",
        "seeds": list(range(5)),
        "energy_scope": (
            "modeled local-training compute/memory energy; SoftDP, BEU "
            "auxiliary logic, communication, and server aggregation energy excluded"
        ),
        "validation": {
            "validated_run_count": 76,
            "latency_closure_tolerance_ms": 1e-6,
            "source_training_outputs_modified": False,
        },
        "results": results,
    }


def paired(
    records: List[Dict[str, Any]],
    first_scenario: str,
    second_scenario: str,
    metric: str,
) -> Dict[str, Any]:
    indexed = {
        (
            record["stage"],
            record["partition_id"],
            record["seed"],
            record["scenario"],
        ): record
        for record in records
    }
    differences = []
    pairs = []
    for first in records:
        if first["scenario"] != first_scenario:
            continue
        key = (
            first["stage"],
            first["partition_id"],
            first["seed"],
            second_scenario,
        )
        if key not in indexed:
            continue
        second = indexed[key]
        first_value = float(result_metrics(first)[metric])
        second_value = float(result_metrics(second)[metric])
        difference = first_value - second_value
        differences.append(difference)
        pairs.append(
            {
                "stage": first["stage"],
                "partition_id": first["partition_id"],
                "training_seed": first["seed"],
                "first_value": first_value,
                "second_value": second_value,
                "difference": difference,
            }
        )
    return {
        "summary": sample_stats(differences, use_t=True),
        "positive_direction_count": int(sum(value > 0 for value in differences)),
        "negative_direction_count": int(sum(value < 0 for value in differences)),
        "pairs": pairs,
    }


def build_semantic_evidence(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    scenario_metrics = {}
    for stage in ("core", "ablation", "stress"):
        scenarios = sorted(
            {
                record["scenario"]
                for record in records
                if record["stage"] == stage
            }
        )
        for scenario in scenarios:
            selected = [
                record
                for record in records
                if record["stage"] == stage and record["scenario"] == scenario
            ]
            scenario_metrics[f"{stage}/{scenario}"] = {
                metric: sample_stats(
                    [
                        float(result_metrics(record)[metric])
                        for record in selected
                    ],
                    use_t=True,
                )
                for metric in EVIDENCE_METRICS
            }

    core_comparisons = {}
    for baseline in ("FP32_softDP", "FedBN", "FedPAQ", "Mao_etal", "BitFusion"):
        core_comparisons[f"HMPE-ACF_minus_{baseline}"] = {
            metric: paired(records, "HMPE-ACF", baseline, metric)
            for metric in (
                "test_dice",
                "avg_latency_ms",
                "avg_local_training_energy_mJ",
            )
        }

    core_full = [
        {**record, "stage": "mechanism", "scenario": "ACF_full"}
        for record in records
        if record["stage"] == "core"
        and record["scenario"] == "HMPE-ACF"
        and record["seed"] in (0, 1, 2)
    ]
    ablation = [
        {**record, "stage": "mechanism"}
        for record in records
        if record["stage"] == "ablation"
    ]
    mechanism_records = core_full + ablation
    mechanism_comparisons = {}
    for reference in ABLATION_SCENARIOS:
        mechanism_comparisons[f"ACF_full_minus_{reference}"] = {
            metric: paired(mechanism_records, "ACF_full", reference, metric)
            for metric in (
                "test_dice",
                "avg_latency_ms",
                "avg_local_training_energy_mJ",
            )
        }

    stress_records = [record for record in records if record["stage"] == "stress"]
    stress_comparisons = {}
    for reference in ("ACF_progress_only", "ACF_FedBN"):
        stress_comparisons[f"ACF_full_minus_{reference}"] = {
            metric: paired(stress_records, "ACF_full", reference, metric)
            for metric in (
                "test_dice",
                "avg_latency_ms",
                "avg_local_training_energy_mJ",
            )
        }

    correlations = []
    for record in records:
        if not (
            (record["stage"] == "core" and record["scenario"] == "HMPE-ACF")
            or (record["stage"] == "stress" and record["scenario"] == "ACF_full")
        ):
            continue
        correlations.append(
            {
                "stage": record["stage"],
                "partition_id": record["partition_id"],
                "training_seed": record["seed"],
                "spearman": spearman(
                    record["history"]["client_entropies"],
                    result_metrics(record)["high_precision_assignment_rate_by_client"],
                ),
                "high_precision_assignment_rate": result_metrics(record)[
                    "high_precision_assignment_rate"
                ],
            }
        )

    return {
        "schema_version": 2,
        "validated_run_count": len(records),
        "expected_run_count": 76,
        "validation": {
            "rounds": 80,
            "local_epochs": 2,
            "latency_breakdown_closed": True,
            "energy_scope": "modeled local-training compute/memory energy",
        },
        "scenario_metrics": scenario_metrics,
        "core_paired_comparisons": core_comparisons,
        "acf_mechanism_paired_comparisons": mechanism_comparisons,
        "strong_heterogeneity_paired_comparisons": stress_comparisons,
        "entropy_assignment_correlations": correlations,
    }


def write_paper_csv(path: Path, paper_results: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "method",
                "dice_mean_percent",
                "dice_sd_percent",
                "latency_mean_ms",
                "latency_sd_ms",
                "normalized_latency_mean",
                "normalized_latency_sd",
                "local_training_compute_memory_energy_mean_mJ",
                "local_training_compute_memory_energy_sd_mJ",
                "normalized_local_training_compute_memory_energy_mean",
                "normalized_local_training_compute_memory_energy_sd",
                "final_epsilon_mean",
                "final_epsilon_sd",
            ]
        )
        for method, entry in paper_results["results"].items():
            metrics = entry["metrics"]
            std = entry["metrics_std"]
            latency = entry["normalized"]["avg_latency_ms"]
            energy = entry["normalized"]["avg_local_training_energy_mJ"]
            writer.writerow(
                [
                    method,
                    metrics["test_dice"],
                    std["test_dice"],
                    metrics["avg_latency_ms"],
                    std["avg_latency_ms"],
                    latency["mean"],
                    latency["std"],
                    metrics["avg_local_training_energy_mJ"],
                    std["avg_local_training_energy_mJ"],
                    energy["mean"],
                    energy["std"],
                    metrics["final_epsilon"],
                    std["final_epsilon"],
                ]
            )


def write_main_table(path: Path, paper_results: Dict[str, Any]) -> None:
    display = {
        "FP32_noDP": "FP32 (NoDP)",
        "FP32_softDP": "FP32 (SoftDP)",
        "FedBN": "FedBN",
        "FedPAQ": "FedPAQ",
        "Mao_etal": "Mao et al.",
        "BitFusion": "BitFusion",
        "HMPE-ACF_noDP": "FedPLH-noDP",
        "HMPE-ACF": "FedPLH",
    }
    rows = []
    for method, entry in paper_results["results"].items():
        metrics = entry["metrics"]
        std = entry["metrics_std"]
        latency = entry["normalized"]["avg_latency_ms"]
        energy = entry["normalized"]["avg_local_training_energy_mJ"]
        rows.append(
            f"{display[method]} & "
            f"{metrics['test_dice']:.2f}$\\pm${std['test_dice']:.2f} & "
            f"{latency['mean']:.3f}$\\pm${latency['std']:.3f} & "
            f"{energy['mean']:.3f}$\\pm${energy['std']:.3f} \\\\"
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_acf_table(path: Path, evidence: Dict[str, Any]) -> None:
    rows = []
    sections = (
        (
            "Main partition",
            evidence["acf_mechanism_paired_comparisons"],
            (
                ("ACF_full_minus_ACF_static_FP8", "Static FP8"),
                ("ACF_full_minus_ACF_progress_only", "Progress-only"),
                ("ACF_full_minus_ACF_entropy_only", "Entropy-only"),
            ),
        ),
        (
            "Strong skew",
            evidence["strong_heterogeneity_paired_comparisons"],
            (
                ("ACF_full_minus_ACF_progress_only", "Progress-only"),
                ("ACF_full_minus_ACF_FedBN", "FedBN"),
            ),
        ),
    )
    for scope, comparisons, entries in sections:
        for key, reference in entries:
            comparison = comparisons[key]
            dice = comparison["test_dice"]["summary"]
            latency = comparison["avg_latency_ms"]["summary"]
            energy = comparison["avg_local_training_energy_mJ"]["summary"]
            rows.append(
                f"{scope} & {reference} & {dice['n']} & "
                f"{dice['mean']:+.3f}$\\pm${dice['ci95_half_width']:.3f} & "
                f"{latency['mean']:+.3f}$\\pm${latency['ci95_half_width']:.3f} & "
                f"{energy['mean']:+.1f}$\\pm${energy['ci95_half_width']:.1f} \\\\"
            )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def build_scalability(simulator: ACFSimulator) -> Dict[str, Any]:
    clients = (2, 5, 10, 20, 50, 100, 200, 500, 1000)
    sizes = (10, 50, 100)
    values = {}
    for client_count in clients:
        entry = {}
        for size_mib in sizes:
            sac = simulator.simulate_aggregation(client_count, size_mib, "PEC")
            cpu = simulator.simulate_aggregation(client_count, size_mib, "Software")
            entry[f"sac_{size_mib}mib"] = sac
            entry[f"cpu_{size_mib}mib"] = cpu
            entry[f"speedup_{size_mib}mib"] = cpu / sac
        values[str(client_count)] = entry
    profile = simulator.hw_profile
    clock_mhz = float(profile["design_parameters"]["clock_frequency_MHz"])
    throughput = float(
        profile["federation_costs"]["PEC_hardware"]["throughput_bytes_per_cycle"]
    )
    lanes = int(profile["federation_costs"]["PEC_hardware"].get("parallel_lanes", 1))
    memory_bw = float(profile["design_parameters"]["memory_bandwidth_GBps"])
    lane_bw = throughput * lanes * clock_mhz / 1000.0
    return {
        "schema_version": 2,
        "model": {
            "clock_frequency_ghz": clock_mhz / 1000.0,
            "memory_bandwidth_gbps": memory_bw,
            "streaming_bandwidth_gbps": lane_bw,
            "effective_bandwidth_gbps": min(memory_bw, lane_bw),
            "pipeline_depth_cycles": int(
                profile["federation_costs"]["PEC_hardware"]["pipeline_depth"]
            ),
            "software_conversion_factor": float(
                profile["federation_costs"]["software_baseline"][
                    "software_conversion_factor"
                ]
            ),
            "communication_excluded": True,
            "display_size_unit": "MiB",
        },
        "clients": values,
    }


def representative_history(record: Dict[str, Any]) -> Dict[str, Any]:
    source = record["history"]
    output = {
        key: copy.deepcopy(source[key])
        for key in (
            "round",
            "train_loss",
            "val_dice",
            "epsilon",
            "compute_latency_ms",
            "dp_overhead_ms",
            "dp_total_ms",
            "dp_background_ms",
            "comm_latency_ms",
            "misc_latency_ms",
            "local_training_energy_mJ",
            "energy_scope",
            "client_entropies",
            "client_precisions",
            "participating_clients",
        )
        if key in source
    }
    output["scenario"] = record["scenario"]
    output["seed"] = record["seed"]
    output["selection"] = "representative FedPLH seed 0; not a five-seed mean curve"
    output["agg_latency_ms"] = copy.deepcopy(record["derived"]["agg_latency_ms"])
    output["latency_ms"] = copy.deepcopy(record["derived"]["latency_ms"])
    output["latency_residual_ms"] = copy.deepcopy(
        record["derived"]["latency_residual_ms"]
    )
    output["metrics"] = copy.deepcopy(result_metrics(record))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("outputs/tetc_semantic_final_20260615"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/tetc_semantic_final_20260615_postprocessed"
        ),
    )
    parser.add_argument("--profile", type=Path, default=Path("hardware_profile.json"))
    args = parser.parse_args()

    source_root = args.source.resolve()
    output_root = args.output.resolve()
    if output_root == source_root or source_root in output_root.parents:
        raise ValueError("Output must be a sibling of the immutable source directory")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite existing postprocessed output: {output_root}"
        )

    records = discover_runs(source_root)
    before_hashes = fingerprint_inputs(records, source_root)
    if len(before_hashes) != 152:
        raise ValueError(f"Expected 152 source files, found {len(before_hashes)}")

    simulator = ACFSimulator(str(args.profile))
    derived_records = [derive_run(record, simulator) for record in records]
    output_root.mkdir(parents=True, exist_ok=True)
    for record in derived_records:
        write_json_atomic(
            derived_output_path(output_root, record),
            record["derived"],
        )

    generated_at = datetime.now(timezone.utc).isoformat()
    code_paths = (
        Path(__file__).resolve(),
        Path("hardware_profile.json").resolve(),
        Path("simulator/acf_simulator.py").resolve(),
        Path("visualization/plot_generator.py").resolve(),
        Path("plot_beu_boundary.py").resolve(),
    )
    metadata = {
        "generated_at_utc": generated_at,
        "source_directory": str(source_root),
        "output_directory": str(output_root),
        "source_file_count": len(before_hashes),
        "code_sha256": {
            path.relative_to(Path.cwd()).as_posix(): sha256_file(path)
            for path in code_paths
        },
        "model_assumptions": {
            "participant_count": "M_r = |S^(r)|",
            "aggregation_input": "S_r = M_r U",
            "fp32_update_bytes": UPDATE_BYTES,
            "fp32_update_mib": UPDATE_MIB,
            "effective_bandwidth": "min(BW_mem, 32 Bytes/cycle * f)",
            "clock_frequency_ghz": 1.408,
            "memory_bandwidth_gbps": 32.0,
            "pipeline_depth_cycles": 14,
            "software_conversion_factor": 12.0,
            "communication_in_aggregation_formula": False,
        },
    }

    paper_results = build_paper_results(derived_records, metadata)
    summaries_dir = output_root / "core" / "unet" / "summaries"
    write_json_atomic(summaries_dir / "paper_results.json", paper_results)
    write_paper_csv(summaries_dir / "paper_results.csv", paper_results)
    write_main_table(summaries_dir / "main_table_rows.tex", paper_results)
    representative = next(
        record
        for record in derived_records
        if record["stage"] == "core"
        and record["scenario"] == "HMPE-ACF"
        and record["seed"] == 0
    )
    write_json_atomic(
        summaries_dir / "representative_history.json",
        representative_history(representative),
    )

    scalability = build_scalability(simulator)
    write_json_atomic(
        output_root / "core" / "unet" / "scalability" / "scalability_results.json",
        scalability,
    )

    evidence = build_semantic_evidence(derived_records)
    write_json_atomic(output_root / "semantic_evidence_summary.json", evidence)
    write_acf_table(output_root / "acf_table_rows.tex", evidence)

    after_hashes = fingerprint_inputs(records, source_root)
    if before_hashes != after_hashes:
        raise RuntimeError("Immutable source inputs changed during postprocessing")
    input_manifest = {
        "schema_version": 1,
        "source_directory": str(source_root),
        "file_count": len(before_hashes),
        "sha256": before_hashes,
        "verified_unchanged_after_postprocessing": True,
    }
    write_json_atomic(output_root / "input_sha256_manifest.json", input_manifest)

    max_residual = max(
        max(abs(value) for value in record["derived"]["latency_residual_ms"])
        for record in derived_records
    )
    postprocess_manifest = {
        "schema_version": 1,
        **metadata,
        "validated_run_count": len(derived_records),
        "source_inputs_unchanged": True,
        "max_abs_latency_closure_residual_ms": max_residual,
        "energy_export_field": "avg_local_training_energy_mJ",
        "legacy_avg_energy_exported": False,
    }
    write_json_atomic(output_root / "postprocess_manifest.json", postprocess_manifest)

    print(
        json.dumps(
            {
                "output": str(output_root),
                "validated_runs": len(derived_records),
                "source_files_hashed": len(before_hashes),
                "max_abs_latency_residual_ms": max_residual,
                "fp32_update_mib": UPDATE_MIB,
                "sac_10mib_1000clients_ms": scalability["clients"]["1000"][
                    "sac_10mib"
                ],
                "speedup_10mib_1000clients": scalability["clients"]["1000"][
                    "speedup_10mib"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
