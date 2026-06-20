import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


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
METRICS = (
    "test_dice",
    "avg_latency_ms",
    "avg_local_training_energy_mJ",
    "final_epsilon",
    "high_precision_assignment_rate",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def summary(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "ci95_half_width": None,
            "values": [],
        }
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    critical = T_CRITICAL_975.get(len(values) - 1, 1.96)
    ci95 = (
        float(critical * std / math.sqrt(len(values)))
        if len(values) > 1
        else 0.0
    )
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "std": std,
        "ci95_half_width": ci95,
        "values": [float(value) for value in values],
    }


def rankdata(values):
    values = np.asarray(values, dtype=np.float64)
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


def spearman(first, second):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(
        [np.nan if value is None else value for value in second],
        dtype=np.float64,
    )
    mask = np.isfinite(first) & np.isfinite(second)
    if int(mask.sum()) < 2:
        return None
    first_ranks = rankdata(first[mask])
    second_ranks = rankdata(second[mask])
    if float(first_ranks.std()) == 0.0 or float(second_ranks.std()) == 0.0:
        return None
    value = float(np.corrcoef(first_ranks, second_ranks)[0, 1])
    return value if np.isfinite(value) else None


def validate_history(history, scenario, seed, expected_rounds, dp_enabled):
    if len(history.get("round", [])) != expected_rounds:
        raise ValueError(
            f"{scenario} seed {seed} does not have {expected_rounds} rounds"
        )
    if int(history.get("local_epochs", -1)) != 2:
        raise ValueError(f"{scenario} seed {seed} does not use two epochs")
    residuals = np.asarray(
        history.get("latency_residual_ms", []),
        dtype=np.float64,
    )
    if len(residuals) != expected_rounds or float(
        np.max(np.abs(residuals))
    ) > 1e-6:
        raise ValueError(f"Latency does not close for {scenario} seed {seed}")
    epsilon = np.asarray(history.get("epsilon", []), dtype=np.float64)
    if len(epsilon) > 1 and float(np.min(np.diff(epsilon))) < -1e-9:
        raise ValueError(f"Epsilon decreases for {scenario} seed {seed}")
    clients = history.get("participating_clients", [])
    sizes = history.get("client_num_samples", [])
    weights = history.get("aggregation_weights", [])
    steps = history.get("client_optimizer_steps", [])
    events = history.get("client_privacy_events", [])
    if not (
        len(clients)
        == len(weights)
        == len(steps)
        == len(events)
        == expected_rounds
    ):
        raise ValueError(f"Audit arrays differ for {scenario} seed {seed}")
    for round_index, (client_ids, round_weights, round_steps, round_events) in enumerate(
        zip(clients, weights, steps, events)
    ):
        counts = [float(sizes[int(client_id)]) for client_id in client_ids]
        denominator = sum(counts)
        expected_weights = [count / denominator for count in counts]
        if not np.allclose(
            round_weights,
            expected_weights,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                f"Weight mismatch in {scenario} seed {seed} round {round_index}"
            )
        event_counts = [len(client_events) for client_events in round_events]
        step_counts = [int(value) for value in round_steps]
        if dp_enabled and event_counts != step_counts:
            raise ValueError(
                f"Privacy-event mismatch in {scenario} seed {seed} "
                f"round {round_index}"
            )
        if (not dp_enabled) and any(event_counts):
            raise ValueError(
                f"NoDP run records privacy events in {scenario} seed {seed} "
                f"round {round_index}"
            )
    if "local-training" not in history.get("energy_scope", ""):
        raise ValueError(f"Energy scope mismatch for {scenario} seed {seed}")


def load_record(stage, root, scenario, seed, partition_id, expected_rounds):
    seed_dir = root / "unet" / scenario / f"seed{seed}"
    with (seed_dir / "training_history.json").open(
        "r",
        encoding="utf-8",
    ) as handle:
        history = json.load(handle)
    with (seed_dir / "run_manifest.json").open(
        "r",
        encoding="utf-8",
    ) as handle:
        manifest = json.load(handle)
    if manifest.get("status") != "completed":
        raise ValueError(f"Incomplete run: {seed_dir}")
    dp_enabled = bool(
        manifest.get("scenario_config", {})
        .get("dp", {})
        .get("enable")
    )
    validate_history(history, scenario, seed, expected_rounds, dp_enabled)
    partition_file = Path(manifest["partition_file"])
    if not partition_file.is_file():
        raise FileNotFoundError(partition_file)
    return {
        "stage": stage,
        "scenario": scenario,
        "partition_id": int(partition_id),
        "training_seed": int(seed),
        "history": history,
        "manifest": manifest,
        "partition_file": str(partition_file.resolve()),
        "partition_sha256": sha256_file(partition_file),
        "path": str(seed_dir.resolve()),
    }


def paired(records, first_scenario, second_scenario, metric):
    indexed = {
        (
            record["stage"],
            record["partition_id"],
            record["training_seed"],
            record["scenario"],
        ): record
        for record in records
    }
    first_records = [
        record
        for record in records
        if record["scenario"] == first_scenario
    ]
    differences = []
    rows = []
    for first in first_records:
        key = (
            first["stage"],
            first["partition_id"],
            first["training_seed"],
            second_scenario,
        )
        if key not in indexed:
            continue
        second = indexed[key]
        first_value = float(first["history"]["metrics"][metric])
        second_value = float(second["history"]["metrics"][metric])
        difference = first_value - second_value
        differences.append(difference)
        rows.append({
            "stage": first["stage"],
            "partition_id": first["partition_id"],
            "training_seed": first["training_seed"],
            "first_value": first_value,
            "second_value": second_value,
            "difference": difference,
        })
    return {
        "summary": summary(differences),
        "positive_direction_count": int(
            sum(value > 0 for value in differences)
        ),
        "negative_direction_count": int(
            sum(value < 0 for value in differences)
        ),
        "pairs": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--core_root", type=Path, required=True)
    parser.add_argument("--ablation_root", type=Path, required=True)
    parser.add_argument("--stress_root", type=Path, required=True)
    parser.add_argument("--protocol_manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=80)
    args = parser.parse_args()

    with args.protocol_manifest.open("r", encoding="utf-8") as handle:
        protocol = json.load(handle)
    expected_main_hash = protocol["main_partition"]["sha256"]
    expected_stress_hashes = {
        int(item["partition_seed"]): item["sha256"]
        for item in protocol["stress_partitions"]
    }

    core_scenarios = (
        "FP32_noDP",
        "FP32_softDP",
        "FedBN",
        "FedPAQ",
        "Mao_etal",
        "BitFusion",
        "HMPE-ACF_noDP",
        "HMPE-ACF",
    )
    records = []
    for scenario in core_scenarios:
        for seed in range(5):
            records.append(
                load_record(
                    "core",
                    args.core_root,
                    scenario,
                    seed,
                    0,
                    args.rounds,
                )
            )
    for scenario in (
        "ACF_static_FP8",
        "ACF_progress_only",
        "ACF_entropy_only",
    ):
        for seed in range(3):
            records.append(
                load_record(
                    "ablation",
                    args.ablation_root,
                    scenario,
                    seed,
                    0,
                    args.rounds,
                )
            )
    for partition_id in range(3):
        partition_root = args.stress_root / f"partition{partition_id}"
        for scenario in ("ACF_FedBN", "ACF_progress_only", "ACF_full"):
            for seed in range(3):
                records.append(
                    load_record(
                        "stress",
                        partition_root,
                        scenario,
                        seed,
                        partition_id,
                        args.rounds,
                    )
                )

    for record in records:
        expected_hash = (
            expected_main_hash
            if record["stage"] in ("core", "ablation")
            else expected_stress_hashes[record["partition_id"]]
        )
        if record["partition_sha256"] != expected_hash:
            raise ValueError(
                f"Partition hash mismatch for {record['path']}"
            )

    scenario_metrics = {}
    for stage in ("core", "ablation", "stress"):
        for scenario in sorted({
            record["scenario"]
            for record in records
            if record["stage"] == stage
        }):
            selected = [
                record
                for record in records
                if record["stage"] == stage
                and record["scenario"] == scenario
            ]
            scenario_metrics[f"{stage}/{scenario}"] = {
                metric: summary([
                    record["history"]["metrics"][metric]
                    for record in selected
                ])
                for metric in METRICS
            }

    core_comparisons = {}
    for baseline in (
        "FP32_softDP",
        "FedBN",
        "FedPAQ",
        "Mao_etal",
        "BitFusion",
    ):
        core_comparisons[f"HMPE-ACF_minus_{baseline}"] = {
            metric: paired(records, "HMPE-ACF", baseline, metric)
            for metric in (
                "test_dice",
                "avg_latency_ms",
                "avg_local_training_energy_mJ",
            )
        }

    mechanism_comparisons = {}
    core_full = [
        record
        for record in records
        if record["stage"] == "core"
        and record["scenario"] == "HMPE-ACF"
        and record["training_seed"] in (0, 1, 2)
    ]
    ablation_records = [
        record for record in records if record["stage"] == "ablation"
    ]
    mechanism_records = [
        {**record, "stage": "mechanism", "scenario": "ACF_full"}
        for record in core_full
    ] + [
        {**record, "stage": "mechanism"}
        for record in ablation_records
    ]
    for reference in (
        "ACF_static_FP8",
        "ACF_progress_only",
        "ACF_entropy_only",
    ):
        mechanism_comparisons[f"ACF_full_minus_{reference}"] = {
            metric: paired(
                mechanism_records,
                "ACF_full",
                reference,
                metric,
            )
            for metric in (
                "test_dice",
                "avg_latency_ms",
                "avg_local_training_energy_mJ",
            )
        }

    stress_records = [
        record for record in records if record["stage"] == "stress"
    ]
    stress_comparisons = {}
    for reference in ("ACF_progress_only", "ACF_FedBN"):
        stress_comparisons[f"ACF_full_minus_{reference}"] = {
            metric: paired(
                stress_records,
                "ACF_full",
                reference,
                metric,
            )
            for metric in (
                "test_dice",
                "avg_latency_ms",
                "avg_local_training_energy_mJ",
            )
        }

    correlations = []
    for record in records:
        if record["scenario"] not in ("HMPE-ACF", "ACF_full"):
            continue
        metrics = record["history"]["metrics"]
        correlations.append({
            "stage": record["stage"],
            "partition_id": record["partition_id"],
            "training_seed": record["training_seed"],
            "spearman": spearman(
                record["history"]["client_entropies"],
                metrics["high_precision_assignment_rate_by_client"],
            ),
            "high_precision_assignment_rate": metrics[
                "high_precision_assignment_rate"
            ],
        })

    output = {
        "schema_version": 1,
        "validated_run_count": len(records),
        "expected_run_count": 76,
        "validation": {
            "rounds": args.rounds,
            "local_epochs": 2,
            "sample_weighted_reduction_verified": True,
            "privacy_events_match_optimizer_steps": True,
            "latency_breakdown_closed": True,
            "cumulative_epsilon_nondecreasing": True,
            "frozen_partition_hashes_verified": True,
            "energy_scope": "modeled local-training compute/memory energy",
        },
        "scenario_metrics": scenario_metrics,
        "core_paired_comparisons": core_comparisons,
        "acf_mechanism_paired_comparisons": mechanism_comparisons,
        "strong_heterogeneity_paired_comparisons": stress_comparisons,
        "entropy_assignment_correlations": correlations,
    }
    if len(records) != 76:
        raise ValueError(f"Expected 76 runs, found {len(records)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)

    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "stage",
                "scenario",
                "partition_id",
                "training_seed",
                *METRICS,
                "partition_sha256",
                "path",
            ],
        )
        writer.writeheader()
        for record in records:
            row = {
                "stage": record["stage"],
                "scenario": record["scenario"],
                "partition_id": record["partition_id"],
                "training_seed": record["training_seed"],
                "partition_sha256": record["partition_sha256"],
                "path": record["path"],
            }
            for metric in METRICS:
                row[metric] = record["history"]["metrics"][metric]
            writer.writerow(row)
    print(f"Validated {len(records)} runs and wrote {args.output}")


if __name__ == "__main__":
    main()
