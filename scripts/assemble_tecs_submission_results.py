#!/usr/bin/env python3
"""Assemble the audited TECS result set without modifying frozen runs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE_RESULTS = (
    ROOT
    / "audited_runs"
    / "tpds_operand_complete_five_seed_20260902"
    / "unet"
    / "summaries"
    / "paper_results.json"
)
REFERENCE_ROOT = ROOT / "audited_runs" / "tpds_operand_complete_five_seed_20260902"
PRIMARY_ROOT = (
    ROOT
    / "audited_runs"
    / "tecs_precision_policy_ablation_operand_complete_20260904"
)
OUTPUT = ROOT / "postprocessed_summaries" / "tecs_submission_results.json"
MANIFEST = (
    ROOT
    / "validated_aggregate_evidence"
    / "tecs_primary_result_manifest.json"
)

EXPECTED_PARTITION_INPUT_SHA256 = (
    "a193f2783c84d38233bd85bf08b442fb75c85d5c2515e2fd4fbc67786c39902f"
)
EXPECTED_PARTITION_REFERENCE_SHA256 = (
    "fe064da7da3259b7bf2e3d83acea2ed7e75c58a70e8ba2bb2bf40fa381e5f945"
)
EXPECTED_PROFILE_SHA256 = (
    "bd1a3bc2f3eb512af8718fec8dc4897963c27dd5ab14727aa309abd2e51d76ce"
)
OPERATOR_CONFIG_KEYS = (
    "enable",
    "simulate_hardware_beu",
    "simulate_hardware_pec",
    "dp_cost_model",
    "clip_norm",
    "noise_multiplier",
    "batch_size",
)
PRIMARY_SCENARIO = "ACF_progress_only"
REFERENCE_SCENARIO = "HMPE-ACF"
SEEDS = list(range(5))
ROUNDS = 80
T_CRITICAL_DF4 = 2.7764451051977987


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Required JSON file is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def finite(value: object, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} is not finite: {value}")
    return number


def canonical_partition(payload: dict[str, Any]) -> dict[str, Any]:
    canonical = copy.deepcopy(payload)
    canonical.get("partition", {}).pop("frozen_partition_file", None)
    return canonical


def summarize(values: list[float]) -> dict[str, Any]:
    if len(values) != len(SEEDS):
        raise ValueError("The fixed protocol requires exactly five values")
    mean = statistics.mean(values)
    sample_sd = statistics.stdev(values)
    ci95 = T_CRITICAL_DF4 * sample_sd / math.sqrt(len(values))
    return {
        "mean": mean,
        "std": sample_sd,
        "sample_sd": sample_sd,
        "ci95": ci95,
        "ci95_half_width": ci95,
        "values": values,
        "n": len(values),
        "ci_method": "two-sided Student-t, df=4",
    }


def operator_config(manifest: dict[str, Any]) -> dict[str, Any]:
    config = manifest.get("scenario_config", {}).get("dp", {})
    return {key: config.get(key) for key in OPERATOR_CONFIG_KEYS}


def validate_primary_policy(manifest: dict[str, Any], seed: int) -> None:
    resolved = manifest.get("resolved_acf", {})
    expected = {
        "mode": "time_decay",
        "lamda": 0.0,
        "budget_threshold": 0.0,
        "deterministic_decision": False,
        "decision_rule": "seeded_bernoulli",
        "low_precision": "FP8_E5M2",
        "high_precision": "BF16",
        "scheduler_seed_namespace": "acf_scheduler:progress_only",
    }
    observed = {key: resolved.get(key) for key in expected}
    if observed != expected:
        raise ValueError(
            f"Progress-only policy mismatch for seed {seed}: {observed}"
        )


def primary_seed_record(history: dict[str, Any], seed: int) -> dict[str, Any]:
    metrics = copy.deepcopy(history.get("metrics", {}))
    required = (
        "test_dice",
        "avg_latency_ms",
        "avg_compute_latency_ms",
        "avg_dp_overhead_ms",
        "avg_dp_total_ms",
        "avg_dp_background_ms",
        "avg_comm_latency_ms",
        "avg_agg_latency_ms",
        "avg_misc_latency_ms",
        "avg_local_training_energy_mJ",
        "avg_delta_c_cycles",
        "avg_c_priv_cycles",
        "high_precision_assignment_rate",
    )
    values = {name: finite(metrics.get(name), f"seed {seed} {name}") for name in required}
    operator_total = values["avg_dp_total_ms"]
    admitted = values["avg_dp_background_ms"]
    visible_bound = values["avg_dp_overhead_ms"]
    if abs(operator_total - admitted - visible_bound) > 1e-6:
        raise ValueError(f"Operator accounting does not close for seed {seed}")
    actual_serial = values["avg_latency_ms"] + admitted
    reconstructed = (
        values["avg_compute_latency_ms"]
        + operator_total
        + values["avg_comm_latency_ms"]
        + values["avg_agg_latency_ms"]
        + values["avg_misc_latency_ms"]
    )
    if abs(actual_serial - reconstructed) > 1e-6:
        raise ValueError(f"Serial latency decomposition does not close for seed {seed}")

    metrics.update(
        {
            "avg_operator_total_ms": operator_total,
            "avg_operator_admitted_ms": admitted,
            "avg_operator_visible_bound_ms": visible_bound,
            "avg_actual_serial_latency_ms": actual_serial,
            "avg_profiled_timing_slack_cycles": values["avg_delta_c_cycles"],
            "avg_operator_cost_cycles": values["avg_c_priv_cycles"],
        }
    )
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-results", type=Path, default=BASE_RESULTS)
    parser.add_argument("--reference-root", type=Path, default=REFERENCE_ROOT)
    parser.add_argument("--primary-root", type=Path, default=PRIMARY_ROOT)
    parser.add_argument("--primary-scenario", default=PRIMARY_SCENARIO)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()

    if args.primary_scenario != PRIMARY_SCENARIO:
        raise ValueError(
            f"The approved TECS configuration is {PRIMARY_SCENARIO}, not "
            f"{args.primary_scenario}"
        )

    base = load_json(args.base_results)
    if base.get("validation", {}).get("status") != "passed":
        raise ValueError("Matched baseline result validation is not passing")
    if base.get("seeds") != SEEDS:
        raise ValueError("Matched baseline results do not use seeds 0..4")
    fedbn = base.get("results", {}).get("FedBN")
    if not fedbn or sorted(int(seed) for seed in fedbn.get("seeds", {})) != SEEDS:
        raise ValueError("FedBN does not contain the five matched seed records")

    primary_parent_path = args.primary_root / "run_manifest.json"
    primary_parent = load_json(primary_parent_path)
    if primary_parent.get("status") != "completed":
        raise ValueError("Primary ablation suite is not complete")
    if primary_parent.get("arguments", {}).get("hmpe_operand_model") != "quantized_operands":
        raise ValueError("Primary runs are not operand complete")
    if primary_parent.get("seeds") != SEEDS:
        raise ValueError("Primary suite does not use seeds 0..4")
    if primary_parent.get("hardware_profile", {}).get("sha256", "").lower() != EXPECTED_PROFILE_SHA256:
        raise ValueError("Primary suite hardware profile hash has changed")
    partition_input = primary_parent.get("methodology", {}).get("partition_input", {})
    if partition_input.get("sha256", "").lower() != EXPECTED_PARTITION_INPUT_SHA256:
        raise ValueError("Primary suite partition input hash has changed")

    primary_partition_path = args.primary_root / "partition_evidence.json"
    reference_partition_path = args.reference_root / "partition_evidence.json"
    if sha256(reference_partition_path).lower() != EXPECTED_PARTITION_REFERENCE_SHA256:
        raise ValueError("Reference partition evidence hash has changed")
    if canonical_partition(load_json(primary_partition_path)) != canonical_partition(
        load_json(reference_partition_path)
    ):
        raise ValueError("Primary and reference partition content differ")

    input_paths = [
        args.base_results,
        primary_parent_path,
        primary_partition_path,
        reference_partition_path,
    ]
    seed_records: dict[str, dict[str, Any]] = {}
    source_hashes_before: dict[str, str] = {}
    primary_operator_config: dict[str, Any] | None = None
    primary_scenario_config: dict[str, Any] | None = None

    for seed in SEEDS:
        primary_dir = args.primary_root / "unet" / args.primary_scenario / f"seed{seed}"
        reference_dir = args.reference_root / "unet" / REFERENCE_SCENARIO / f"seed{seed}"
        primary_manifest_path = primary_dir / "run_manifest.json"
        primary_history_path = primary_dir / "training_history.json"
        reference_manifest_path = reference_dir / "run_manifest.json"
        primary_manifest = load_json(primary_manifest_path)
        primary_history = load_json(primary_history_path)
        reference_manifest = load_json(reference_manifest_path)
        input_paths.extend(
            [primary_manifest_path, primary_history_path, reference_manifest_path]
        )
        source_hashes_before[str(primary_history_path.resolve())] = sha256(
            primary_history_path
        )

        if primary_manifest.get("status") != "completed":
            raise ValueError(f"Primary seed {seed} is incomplete")
        validate_primary_policy(primary_manifest, seed)
        if primary_history.get("round") != list(range(ROUNDS)):
            raise ValueError(f"Primary seed {seed} is not an 80-round history")
        for field in ("val_dice", "train_loss", "latency_ms", "client_precisions"):
            if len(primary_history.get(field, [])) != ROUNDS:
                raise ValueError(f"Primary seed {seed} has incomplete {field}")
        if primary_manifest.get("client_schedule") != reference_manifest.get("client_schedule")[:ROUNDS]:
            raise ValueError(f"Client schedule differs from the matched seed {seed}")
        if primary_manifest.get("partition_evidence_sha256", "").lower() != sha256(primary_partition_path).lower():
            raise ValueError(f"Partition export hash differs for primary seed {seed}")

        current_operator = operator_config(primary_manifest)
        reference_operator = operator_config(reference_manifest)
        if current_operator != reference_operator:
            raise ValueError(f"Operator configuration differs for seed {seed}")
        if primary_operator_config is None:
            primary_operator_config = current_operator
            primary_scenario_config = copy.deepcopy(primary_manifest.get("scenario_config", {}))
        elif current_operator != primary_operator_config:
            raise ValueError("Operator configuration differs across primary seeds")

        seed_records[str(seed)] = primary_seed_record(primary_history, seed)

    metric_names = (
        "test_dice",
        "avg_latency_ms",
        "avg_actual_serial_latency_ms",
        "avg_local_training_energy_mJ",
        "avg_compute_latency_ms",
        "avg_operator_visible_bound_ms",
        "avg_operator_total_ms",
        "avg_operator_admitted_ms",
        "avg_comm_latency_ms",
        "avg_agg_latency_ms",
        "avg_misc_latency_ms",
        "avg_profiled_timing_slack_cycles",
        "avg_operator_cost_cycles",
        "high_precision_assignment_rate",
    )
    summaries = {
        name: summarize([finite(seed_records[str(seed)][name], name) for seed in SEEDS])
        for name in metric_names
    }
    normalized_actual_latency = summarize(
        [
            seed_records[str(seed)]["avg_actual_serial_latency_ms"]
            / finite(fedbn["seeds"][str(seed)]["avg_latency_ms"], "FedBN latency")
            for seed in SEEDS
        ]
    )
    normalized_bound_latency = summarize(
        [
            seed_records[str(seed)]["avg_latency_ms"]
            / finite(fedbn["seeds"][str(seed)]["avg_latency_ms"], "FedBN latency")
            for seed in SEEDS
        ]
    )
    normalized_energy = summarize(
        [
            seed_records[str(seed)]["avg_local_training_energy_mJ"]
            / finite(
                fedbn["seeds"][str(seed)]["avg_local_training_energy_mJ"],
                "FedBN energy",
            )
            for seed in SEEDS
        ]
    )

    output = copy.deepcopy(base)
    output.pop("privacy_accounting", None)
    output["schema_version"] = 2
    output["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    output["selection"] = {
        "paper_method": "FedMPE",
        "source_scenario": args.primary_scenario,
        "display_policy": "Progress only",
        "selection_basis": (
            "Representative configuration chosen after the complete five-policy "
            "ablation; it is not described as preregistered or globally optimal."
        ),
        "training_reused_without_retraining": True,
    }
    output["evidence_scope"] = {
        "dataset": "BraTS 2021",
        "model": "3D U-Net",
        "seeds": SEEDS,
        "rounds": ROUNDS,
        "operand_model": "quantized_operands",
        "overall_latency": "serial auxiliary-operator execution",
        "admission_result": "deadline bound under full admission",
    }
    output["legacy_field_mapping"] = {
        "avg_dp_total_ms": "avg_operator_total_ms",
        "avg_dp_background_ms": "avg_operator_admitted_ms",
        "avg_dp_overhead_ms": "avg_operator_visible_bound_ms",
        "avg_delta_c_cycles": "avg_profiled_timing_slack_cycles",
        "avg_c_priv_cycles": "avg_operator_cost_cycles",
        "actual_serial_latency": (
            "avg_latency_ms plus avg_dp_background_ms in the frozen histories"
        ),
    }
    output["results"].pop(REFERENCE_SCENARIO, None)
    output["results"]["FedMPE"] = {
        "metrics": {name: values["mean"] for name, values in summaries.items()},
        "metrics_std": {name: values["sample_sd"] for name, values in summaries.items()},
        "metrics_ci95": {name: values["ci95_half_width"] for name, values in summaries.items()},
        "normalized": {
            "avg_actual_serial_latency_ms": normalized_actual_latency,
            "avg_admission_bound_latency_ms": normalized_bound_latency,
            "avg_local_training_energy_mJ": normalized_energy,
        },
        "seeds": seed_records,
        "scenario_config": primary_scenario_config,
        "source_scenario": args.primary_scenario,
    }
    output["validation"] = {
        "status": "passed",
        "seeds": SEEDS,
        "rounds": ROUNDS,
        "matched_client_schedules": True,
        "canonical_partition_match": True,
        "matched_operator_configuration": True,
        "hardware_profile_sha256": EXPECTED_PROFILE_SHA256,
        "operator_accounting_tolerance_ms": 1e-6,
        "serial_latency_tolerance_ms": 1e-6,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    for path_text, before_hash in source_hashes_before.items():
        if sha256(Path(path_text)) != before_hash:
            raise ValueError(f"Frozen history changed during assembly: {path_text}")

    inputs = [
        {"path": str(path.resolve()), "sha256": sha256(path)}
        for path in input_paths
    ]
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "tool": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))},
        "primary_policy": "Progress only",
        "source_scenario": args.primary_scenario,
        "no_retraining": True,
        "frozen_histories_unchanged": True,
        "protocol": {
            "seeds": SEEDS,
            "rounds": ROUNDS,
            "partition_input_sha256": EXPECTED_PARTITION_INPUT_SHA256,
            "partition_reference_sha256": EXPECTED_PARTITION_REFERENCE_SHA256,
            "hardware_profile_sha256": EXPECTED_PROFILE_SHA256,
            "operator_configuration": primary_operator_config,
        },
        "inputs": inputs,
        "output": {
            "path": str(args.output.resolve()),
            "sha256": sha256(args.output),
            "canonical_json_sha256": canonical_json_sha256(output),
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        "Assembled TECS submission results from Progress only: "
        f"5 seeds x {ROUNDS} rounds -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
