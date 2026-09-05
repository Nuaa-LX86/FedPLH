#!/usr/bin/env python3
"""Validate and summarize the matched FedMPE precision-policy ablation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scipy.stats import t as student_t


SCENARIOS = {
    "ACF_static_BF16": {
        "label": "Static BF16",
        "mode": "static_high",
        "lamda": 0.0,
        "deterministic": True,
        "scheduler_stream": "acf_scheduler:static_bf16",
    },
    "ACF_static_FP8": {
        "label": "Static FP8",
        "mode": "static_low",
        "lamda": 0.0,
        "deterministic": True,
        "scheduler_stream": "acf_scheduler:static_fp8",
    },
    "ACF_progress_only": {
        "label": "Progress only",
        "mode": "time_decay",
        "lamda": 0.0,
        "deterministic": False,
        "scheduler_stream": "acf_scheduler:progress_only",
    },
    "ACF_entropy_only": {
        "label": "Entropy only",
        "mode": "entropy_only",
        "lamda": 1.0,
        "deterministic": False,
        "scheduler_stream": "acf_scheduler:entropy_only",
    },
    "ACF_full": {
        "label": "Entropy + progress",
        "mode": "entropy_time",
        "lamda": 0.5,
        "deterministic": False,
        "scheduler_stream": "acf_scheduler",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Required JSON file is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_partition_evidence(payload: dict[str, Any]) -> dict[str, Any]:
    """Remove the frozen file location while retaining scientific content."""
    canonical = json.loads(json.dumps(payload))
    canonical.get("partition", {}).pop("frozen_partition_file", None)
    return canonical


def finite(value: object, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} is not finite: {value}")
    return number


def mean_sd_ci(values: list[float]) -> dict[str, float | int | list[float]]:
    if not values:
        raise ValueError("Cannot summarize an empty value list")
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    critical = float(student_t.ppf(0.975, len(values) - 1)) if len(values) > 1 else 0.0
    half_width = critical * sd / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {
        "n": len(values),
        "values": values,
        "mean": mean,
        "sample_sd": sd,
        "ci95_half_width": half_width,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
        "ci_method": "two-sided Student-t",
    }


def paired_summary(left: list[float], right: list[float]) -> dict[str, Any]:
    if len(left) != len(right):
        raise ValueError("Paired comparisons require equal-length vectors")
    differences = [a - b for a, b in zip(left, right)]
    output = mean_sd_ci(differences)
    output["positive_count"] = sum(value > 0.0 for value in differences)
    output["negative_count"] = sum(value < 0.0 for value in differences)
    output["zero_count"] = sum(value == 0.0 for value in differences)
    return output


def dominates(left: dict[str, float], right: dict[str, float]) -> bool:
    no_worse = (
        left["test_dice"] >= right["test_dice"]
        and left["actual_latency_ms"] <= right["actual_latency_ms"]
        and left["local_training_energy_mJ"] <= right["local_training_energy_mJ"]
    )
    strictly_better = (
        left["test_dice"] > right["test_dice"]
        or left["actual_latency_ms"] < right["actual_latency_ms"]
        or left["local_training_energy_mJ"] < right["local_training_energy_mJ"]
    )
    return no_worse and strictly_better


def validate_scenario_config(manifest: dict[str, Any], scenario: str) -> None:
    expected = SCENARIOS[scenario]
    resolved = manifest.get("resolved_acf", {})
    observed = {
        "mode": resolved.get("mode"),
        "lamda": float(resolved.get("lamda", -1.0)),
        "deterministic": bool(resolved.get("deterministic_decision")),
        "scheduler_stream": resolved.get("scheduler_seed_namespace"),
    }
    if observed != {
        "mode": expected["mode"],
        "lamda": expected["lamda"],
        "deterministic": expected["deterministic"],
        "scheduler_stream": expected["scheduler_stream"],
    }:
        raise ValueError(f"Resolved policy mismatch for {scenario}: {observed}")
    if resolved.get("low_precision") != "FP8_E5M2":
        raise ValueError(f"Unexpected low precision for {scenario}")
    if resolved.get("high_precision") != "BF16":
        raise ValueError(f"Unexpected high precision for {scenario}")
    if float(resolved.get("budget_threshold", -1.0)) != 0.0:
        raise ValueError(f"Budget recovery must be inactive for {scenario}")

    config = manifest.get("scenario_config", {})
    if config.get("acf", {}).get("strategy") != "EntropyAware":
        raise ValueError(f"Learning strategy is not matched for {scenario}")
    dp = config.get("dp", {})
    required_dp = {
        "enable": True,
        "simulate_hardware_beu": True,
        "simulate_hardware_pec": True,
        "dp_cost_model": "paper",
        "clip_norm": 1.0,
        "noise_multiplier": 0.1,
    }
    for key, expected_value in required_dp.items():
        if dp.get(key) != expected_value:
            raise ValueError(
                f"Operator configuration mismatch for {scenario}: "
                f"{key}={dp.get(key)!r}"
            )


def metric_record(history: dict[str, Any], scenario: str, seed: int) -> dict[str, Any]:
    metrics = history.get("metrics", {})
    detailed = metrics.get("test_dice_detailed")
    if not isinstance(detailed, dict):
        raise ValueError(f"Missing detailed test Dice for {scenario}, seed {seed}")
    class_values = {
        key: finite(detailed[key], f"{scenario} seed {seed} {key}")
        for key in ("WT", "TC", "ET", "Mean")
    }
    expected_mean = statistics.mean(
        [class_values["WT"], class_values["TC"], class_values["ET"]]
    )
    if abs(expected_mean - class_values["Mean"]) > 1e-9:
        raise ValueError(f"Detailed Dice does not close for {scenario}, seed {seed}")

    test_dice = finite(metrics.get("test_dice"), f"{scenario} seed {seed} Dice")
    if abs(test_dice - class_values["Mean"]) > 1e-9:
        raise ValueError(f"Aggregate and detailed test Dice disagree for {scenario}, seed {seed}")

    visible_latency = finite(
        metrics.get("avg_latency_ms"), f"{scenario} seed {seed} latency"
    )
    covered_operator = finite(
        metrics.get("avg_dp_background_ms"),
        f"{scenario} seed {seed} covered operator cost",
    )
    return {
        "scenario": scenario,
        "label": SCENARIOS[scenario]["label"],
        "seed": seed,
        "test_dice": test_dice,
        "WT": class_values["WT"],
        "TC": class_values["TC"],
        "ET": class_values["ET"],
        "full_credit_bound_latency_ms": visible_latency,
        "covered_operator_cost_ms": covered_operator,
        "actual_latency_ms": visible_latency + covered_operator,
        "local_training_energy_mJ": finite(
            metrics.get("avg_local_training_energy_mJ"),
            f"{scenario} seed {seed} local-training energy",
        ),
        "bf16_assignment_rate": finite(
            metrics.get("high_precision_assignment_rate"),
            f"{scenario} seed {seed} BF16 rate",
        ),
    }


def latex_pm(mean: float, sd: float, digits: int) -> str:
    return f"{mean:.{digits}f}\\(\\pm\\){sd:.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--expected-rounds", type=int, required=True)
    parser.add_argument("--expected-seeds", required=True)
    parser.add_argument("--expected-partition-input-sha256", required=True)
    parser.add_argument("--expected-partition-output-sha256", required=True)
    parser.add_argument("--expected-profile-sha256", required=True)
    parser.add_argument("--phase", choices=["smoke", "full"], required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path)
    parser.add_argument("--latex-output", type=Path)
    parser.add_argument("--manifest-output", type=Path)
    args = parser.parse_args()

    expected_seeds = [int(value) for value in args.expected_seeds.split(",")]
    run_manifest_path = args.run_root / "run_manifest.json"
    run_manifest = load_json(run_manifest_path)
    arguments = run_manifest.get("arguments", {})
    if arguments.get("hmpe_operand_model") != "quantized_operands":
        raise ValueError("The ablation is not operand-complete")
    if int(arguments.get("rounds", -1)) != args.expected_rounds:
        raise ValueError("Run manifest has the wrong round count")
    if int(arguments.get("clients", -1)) != 20:
        raise ValueError("Run manifest has the wrong client count")
    if str(arguments.get("seeds")) != args.expected_seeds:
        raise ValueError("Run manifest has the wrong seed set")

    partition_input = run_manifest.get("methodology", {}).get("partition_input", {})
    if partition_input.get("sha256", "").upper() != args.expected_partition_input_sha256.upper():
        raise ValueError("Input partition hash does not match the frozen protocol")
    profile = run_manifest.get("hardware_profile", {})
    if profile.get("sha256", "").upper() != args.expected_profile_sha256.upper():
        raise ValueError("Hardware profile hash does not match the frozen protocol")
    emitted_partition_path = args.run_root / "partition_evidence.json"
    reference_partition_path = args.reference_root / "partition_evidence.json"
    if sha256(reference_partition_path).upper() != args.expected_partition_output_sha256.upper():
        raise ValueError("The frozen reference partition evidence hash has changed")
    emitted_partition_sha256 = sha256(emitted_partition_path)
    if canonical_partition_evidence(load_json(emitted_partition_path)) != canonical_partition_evidence(
        load_json(reference_partition_path)
    ):
        raise ValueError("Emitted partition evidence differs from the reference content")

    reference_manifest = load_json(args.reference_root / "run_manifest.json")
    for key in ("inventory_sha256",):
        if run_manifest.get("dataset", {}).get(key) != reference_manifest.get("dataset", {}).get(key):
            raise ValueError(f"Dataset {key} differs from the matched reference")

    records: list[dict[str, Any]] = []
    schedules_by_seed: dict[int, Any] = {}
    scheduler_seeds_by_seed: dict[int, dict[str, int]] = {}
    inputs = [run_manifest_path, emitted_partition_path]
    for scenario in SCENARIOS:
        scenario_root = args.run_root / "unet" / scenario
        observed_seed_dirs = sorted(
            int(path.name.removeprefix("seed"))
            for path in scenario_root.glob("seed*")
            if path.is_dir() and path.name.removeprefix("seed").isdigit()
        )
        if observed_seed_dirs != expected_seeds:
            raise ValueError(
                f"{scenario} seeds are {observed_seed_dirs}, expected {expected_seeds}"
            )
        for seed in expected_seeds:
            seed_root = scenario_root / f"seed{seed}"
            manifest_path = seed_root / "run_manifest.json"
            history_path = seed_root / "training_history.json"
            manifest = load_json(manifest_path)
            history = load_json(history_path)
            inputs.extend([manifest_path, history_path])
            if manifest.get("status") != "completed":
                raise ValueError(f"Incomplete run: {scenario}, seed {seed}")
            if manifest.get("partition_evidence_sha256", "").upper() != emitted_partition_sha256.upper():
                raise ValueError(f"Partition evidence mismatch: {scenario}, seed {seed}")
            validate_scenario_config(manifest, scenario)
            scheduler_seed = int(manifest["resolved_acf"]["scheduler_seed"])
            scheduler_seeds_by_seed.setdefault(seed, {})[scenario] = scheduler_seed
            history_policy = history.get("acf_policy", {})
            if int(history_policy.get("scheduler_seed", -1)) != scheduler_seed:
                raise ValueError(
                    f"Scheduler seed differs between manifest and history: "
                    f"{scenario}, seed {seed}"
                )
            if history_policy.get("scheduler_seed_namespace") != SCENARIOS[scenario]["scheduler_stream"]:
                raise ValueError(
                    f"Scheduler namespace differs between manifest and history: "
                    f"{scenario}, seed {seed}"
                )
            if history.get("round") != list(range(args.expected_rounds)):
                raise ValueError(f"Round sequence mismatch: {scenario}, seed {seed}")
            for field in ("train_loss", "val_dice", "latency_ms"):
                values = history.get(field, [])
                if len(values) != args.expected_rounds:
                    raise ValueError(f"{field} length mismatch: {scenario}, seed {seed}")
                for index, value in enumerate(values):
                    finite(value, f"{scenario} seed {seed} {field}[{index}]")
            schedule = manifest.get("client_schedule")
            if seed not in schedules_by_seed:
                schedules_by_seed[seed] = schedule
            elif schedules_by_seed[seed] != schedule:
                raise ValueError(f"Client schedule differs across policies for seed {seed}")
            reference_seed = load_json(
                args.reference_root / "unet" / "HMPE-ACF" / f"seed{seed}" / "run_manifest.json"
            )
            if schedule != reference_seed.get("client_schedule")[: args.expected_rounds]:
                raise ValueError(f"Client schedule differs from reference for seed {seed}")
            records.append(metric_record(history, scenario, seed))

    dynamic_scenarios = (
        "ACF_progress_only",
        "ACF_entropy_only",
        "ACF_full",
    )
    for seed, seed_map in scheduler_seeds_by_seed.items():
        dynamic_seeds = [seed_map[scenario] for scenario in dynamic_scenarios]
        if len(set(dynamic_seeds)) != len(dynamic_seeds):
            raise ValueError(
                f"Dynamic precision policies share a scheduler seed for seed {seed}"
            )

    static_bf16 = [row for row in records if row["scenario"] == "ACF_static_BF16"]
    static_fp8 = [row for row in records if row["scenario"] == "ACF_static_FP8"]
    if any(abs(row["bf16_assignment_rate"] - 1.0) > 1e-12 for row in static_bf16):
        raise ValueError("Static BF16 did not assign BF16 for every selected client")
    if any(abs(row["bf16_assignment_rate"]) > 1e-12 for row in static_fp8):
        raise ValueError("Static FP8 assigned BF16 unexpectedly")

    metrics = (
        "test_dice", "WT", "TC", "ET", "actual_latency_ms",
        "full_credit_bound_latency_ms", "covered_operator_cost_ms",
        "local_training_energy_mJ", "bf16_assignment_rate",
    )
    by_scenario: dict[str, Any] = {}
    for scenario, metadata in SCENARIOS.items():
        scenario_rows = [row for row in records if row["scenario"] == scenario]
        by_scenario[scenario] = {
            "label": metadata["label"],
            "metrics": {
                metric: mean_sd_ci([float(row[metric]) for row in scenario_rows])
                for metric in metrics
            },
        }

    bf16_latency = {
        row["seed"]: row["actual_latency_ms"] for row in static_bf16
    }
    bf16_energy = {
        row["seed"]: row["local_training_energy_mJ"] for row in static_bf16
    }
    for row in records:
        row["actual_latency_norm_to_static_bf16"] = (
            row["actual_latency_ms"] / bf16_latency[row["seed"]]
        )
        row["energy_norm_to_static_bf16"] = (
            row["local_training_energy_mJ"] / bf16_energy[row["seed"]]
        )
    for scenario in SCENARIOS:
        scenario_rows = [row for row in records if row["scenario"] == scenario]
        by_scenario[scenario]["metrics"]["actual_latency_norm_to_static_bf16"] = mean_sd_ci(
            [row["actual_latency_norm_to_static_bf16"] for row in scenario_rows]
        )
        by_scenario[scenario]["metrics"]["energy_norm_to_static_bf16"] = mean_sd_ci(
            [row["energy_norm_to_static_bf16"] for row in scenario_rows]
        )

    comparisons: dict[str, Any] = {}
    full_rows = [row for row in records if row["scenario"] == "ACF_full"]
    for control in SCENARIOS:
        if control == "ACF_full":
            continue
        control_rows = [row for row in records if row["scenario"] == control]
        comparisons[f"ACF_full_minus_{control}"] = {
            metric: paired_summary(
                [row[metric] for row in full_rows],
                [row[metric] for row in control_rows],
            )
            for metric in (
                "test_dice", "WT", "TC", "ET", "actual_latency_ms",
                "local_training_energy_mJ", "bf16_assignment_rate",
            )
        }

    means = {
        scenario: {
            metric: float(by_scenario[scenario]["metrics"][metric]["mean"])
            for metric in ("test_dice", "actual_latency_ms", "local_training_energy_mJ")
        }
        for scenario in SCENARIOS
    }
    adaptive_gate = None
    synergy_gate = None
    if args.phase == "full":
        dice_cmp = comparisons["ACF_full_minus_ACF_static_FP8"]["test_dice"]
        latency_cmp = comparisons["ACF_full_minus_ACF_static_BF16"]["actual_latency_ms"]
        energy_cmp = comparisons["ACF_full_minus_ACF_static_BF16"]["local_training_energy_mJ"]
        adaptive_gate = bool(
            means["ACF_full"]["test_dice"] > means["ACF_static_FP8"]["test_dice"]
            and means["ACF_full"]["actual_latency_ms"] < means["ACF_static_BF16"]["actual_latency_ms"]
            and means["ACF_full"]["local_training_energy_mJ"] < means["ACF_static_BF16"]["local_training_energy_mJ"]
            and int(dice_cmp["positive_count"]) >= 3
            and int(latency_cmp["negative_count"]) >= 3
            and int(energy_cmp["negative_count"]) >= 3
        )
        synergy_gate = not dominates(means["ACF_progress_only"], means["ACF_full"]) and not dominates(
            means["ACF_entropy_only"], means["ACF_full"]
        )

    reference_comparison = None
    if args.phase == "full":
        reference_comparison = {"status": "passed", "seeds": {}}
        for seed in expected_seeds:
            current_history = load_json(
                args.run_root / "unet" / "ACF_full" / f"seed{seed}" / "training_history.json"
            )
            reference_history = load_json(
                args.reference_root / "unet" / "HMPE-ACF" / f"seed{seed}" / "training_history.json"
            )
            precision_match = current_history.get("client_precisions") == reference_history.get("client_precisions")
            dice_delta = finite(current_history["metrics"]["test_dice"], "current Dice") - finite(
                reference_history["metrics"]["test_dice"], "reference Dice"
            )
            latency_delta = (
                finite(current_history["metrics"]["avg_latency_ms"], "current latency")
                - finite(reference_history["metrics"]["avg_latency_ms"], "reference latency")
            )
            if not precision_match or abs(dice_delta) > 1e-8 or abs(latency_delta) > 1e-8:
                reference_comparison["status"] = "blocked"
            reference_comparison["seeds"][str(seed)] = {
                "precision_trace_match": precision_match,
                "test_dice_delta": dice_delta,
                "full_credit_latency_delta_ms": latency_delta,
            }

    output = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if reference_comparison is None or reference_comparison["status"] == "passed" else "blocked",
        "phase": args.phase,
        "scope": "matched operand-complete precision-policy ablation on BraTS 2021 3D U-Net",
        "seeds": expected_seeds,
        "rounds": args.expected_rounds,
        "records": records,
        "scenario_summary": by_scenario,
        "paired_comparisons": comparisons,
        "claim_gates": {
            "adaptive_operating_point": adaptive_gate,
            "entropy_progress_non_dominated": synergy_gate,
        },
        "reference_full_comparison": reference_comparison,
        "evidence_boundary": {
            "second_dataset": False,
            "second_model": False,
            "formal_dp": False,
            "actual_latency_definition": "full-credit bound latency plus covered operator cost",
        },
        "partition_evidence": {
            "input_sha256": args.expected_partition_input_sha256.upper(),
            "reference_export_sha256": args.expected_partition_output_sha256.upper(),
            "current_export_sha256": emitted_partition_sha256.upper(),
            "canonical_content_matches_reference": True,
        },
    }

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    generated_outputs = [args.json_output]
    if args.csv_output:
        args.csv_output.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(records[0])
        with args.csv_output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        generated_outputs.append(args.csv_output)

    if args.latex_output:
        rows = []
        for scenario in SCENARIOS:
            summary = by_scenario[scenario]
            values = summary["metrics"]
            label = summary["label"]
            rows.append(
                f"{label} & "
                f"{latex_pm(values['test_dice']['mean'], values['test_dice']['sample_sd'], 2)} & "
                f"{latex_pm(values['actual_latency_norm_to_static_bf16']['mean'], values['actual_latency_norm_to_static_bf16']['sample_sd'], 3)} & "
                f"{latex_pm(values['energy_norm_to_static_bf16']['mean'], values['energy_norm_to_static_bf16']['sample_sd'], 3)} & "
                f"{100.0 * values['bf16_assignment_rate']['mean']:.1f}\\% \\\\"
            )
        full_vs_fp8 = comparisons["ACF_full_minus_ACF_static_FP8"]
        full_vs_bf16 = comparisons["ACF_full_minus_ACF_static_BF16"]
        if adaptive_gate:
            main_finding = (
                "The combined policy lies between the static endpoints: its mean "
                "Dice exceeded Static FP8, while its serial latency and scoped "
                "energy remained below Static BF16. These directions held in "
                f"{full_vs_fp8['test_dice']['positive_count']}/5, "
                f"{full_vs_bf16['actual_latency_ms']['negative_count']}/5, and "
                f"{full_vs_bf16['local_training_energy_mJ']['negative_count']}/5 "
                "paired seeds, respectively."
            )
        else:
            main_finding = (
                "The combined policy did not improve the static trade-off in all "
                "three directions and is therefore treated as a precision "
                "configuration rather than a separate source of performance."
            )
        if synergy_gate:
            joint_finding = (
                "Neither single-input policy dominated the combined policy across "
                "Dice, serial latency, and scoped energy."
            )
        else:
            dominating_policies = [
                SCENARIOS[scenario]["label"]
                for scenario in ("ACF_progress_only", "ACF_entropy_only")
                if dominates(means[scenario], means["ACF_full"])
            ]
            subject = " and ".join(dominating_policies)
            joint_finding = (
                f"{subject} provided a stronger three-metric operating point than "
                "the combined setting. Entropy and progress are therefore treated "
                "as alternative controller inputs rather than a verified joint gain."
            )
        latex_lines = [
            "% Generated from the matched five-policy, five-seed ablation. Do not edit.",
            "\\renewcommand{\\PrecisionPolicyAblationRows}{%",
            *[f"  {row}" for row in rows],
            "}",
            f"\\renewcommand{{\\FullVsFPClassDiffs}}{{{full_vs_fp8['WT']['mean']:+.2f}, {full_vs_fp8['TC']['mean']:+.2f}, and {full_vs_fp8['ET']['mean']:+.2f}}}",
            f"\\renewcommand{{\\FullVsBFClassDiffs}}{{{full_vs_bf16['WT']['mean']:+.2f}, {full_vs_bf16['TC']['mean']:+.2f}, and {full_vs_bf16['ET']['mean']:+.2f}}}",
            "\\renewcommand{\\PrecisionPolicyAdaptiveGate}{"
            + ("passed" if adaptive_gate else "not passed") + "}",
            "\\renewcommand{\\PrecisionPolicySynergyGate}{"
            + ("passed" if synergy_gate else "not passed") + "}",
            f"\\renewcommand{{\\PrecisionPolicyMainFinding}}{{{main_finding}}}",
            f"\\renewcommand{{\\PrecisionPolicyJointFinding}}{{{joint_finding}}}",
        ]
        args.latex_output.parent.mkdir(parents=True, exist_ok=True)
        args.latex_output.write_text("\n".join(latex_lines) + "\n", encoding="ascii")
        generated_outputs.append(args.latex_output)

    if args.manifest_output:
        manifest = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": output["status"],
            "tool": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))},
            "inputs": [
                {"path": str(path.resolve()), "sha256": sha256(path)}
                for path in inputs
            ],
            "outputs": [
                {"path": str(path.resolve()), "sha256": sha256(path)}
                for path in generated_outputs
            ],
        }
        args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
        args.manifest_output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    if output["status"] != "passed":
        raise ValueError("The new full-policy run does not reproduce the frozen reference")
    print(
        f"Validated {len(records)} records: {len(SCENARIOS)} policies x "
        f"{len(expected_seeds)} seed(s), {args.expected_rounds} rounds each"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
