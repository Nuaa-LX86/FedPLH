#!/usr/bin/env python3
"""Export audited TECS result artifacts into one neutral LaTeX macro file."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "postprocessed_summaries" / "tecs_submission_results.json"
DEFAULT_PRIMARY_ROOT = (
    ROOT
    / "audited_runs"
    / "tecs_precision_policy_ablation_operand_complete_20260904"
)
DEFAULT_SOTA_AUDIT = (
    ROOT / "validated_aggregate_evidence" / "sota_adapter_five_seed_audit.json"
)
DEFAULT_OUTPUT = ROOT / "TECS_submission" / "source" / "generated_result_values.tex"
DEFAULT_MANIFEST = (
    ROOT / "validated_aggregate_evidence" / "tecs_result_macro_manifest.json"
)
SEEDS = list(range(5))
PRIMARY_SCENARIO = "ACF_progress_only"
T_CRITICAL_DF4 = 2.7764451051977987


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"required artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def finite(value: object, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise SystemExit(f"{label} is not finite: {value}")
    return number


def command(name: str, value: object) -> str:
    return f"\\renewcommand{{\\{name}}}{{{value}}}"


def text_pm(mean: float, sd: float, digits: int = 2) -> str:
    return f"{mean:.{digits}f}\\(\\pm\\){sd:.{digits}f}"


def math_pm(mean: float, sd: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f}\\pm{sd:.{digits}f}"


def result_entry(results: dict[str, Any], name: str) -> dict[str, Any]:
    try:
        entry = results["results"][name]
    except KeyError as exc:
        raise SystemExit(f"submission results lack required scenario {name}") from exc
    if sorted(int(seed) for seed in entry.get("seeds", {})) != SEEDS:
        raise SystemExit(f"{name} does not contain exactly seeds 0..4")
    return entry


def metric(entry: dict[str, Any], name: str) -> float:
    return finite(entry["metrics"][name], f"{name} mean")


def metric_sd(entry: dict[str, Any], name: str) -> float:
    return finite(entry["metrics_std"][name], f"{name} sample SD")


def mean_sd(entry: dict[str, Any], name: str) -> tuple[float, float]:
    return metric(entry, name), metric_sd(entry, name)


def normalized(entry: dict[str, Any], name: str) -> tuple[float, float]:
    record = entry["normalized"][name]
    values = record.get("values")
    count = record.get("n", len(values) if isinstance(values, list) else -1)
    if int(count) != 5:
        raise SystemExit(f"{name} normalization does not contain five paired seeds")
    if isinstance(values, list):
        for index, value in enumerate(values):
            finite(value, f"{name} normalized pair {index}")
    return finite(record["mean"], name), finite(record["std"], f"{name} SD")


def seed_metric(entry: dict[str, Any], seed: int, name: str) -> float:
    return finite(entry["seeds"][str(seed)][name], f"seed {seed} {name}")


def paired_stats(left: list[float], right: list[float]) -> tuple[float, float, float]:
    if len(left) != 5 or len(right) != 5:
        raise SystemExit("paired statistics require exactly five values per method")
    differences = [a - b for a, b in zip(left, right)]
    mean = statistics.mean(differences)
    sd = statistics.stdev(differences)
    return mean, sd, T_CRITICAL_DF4 * sd / math.sqrt(len(differences))


def sota_summary(audit: dict[str, Any]) -> dict[str, tuple[float, float]]:
    if audit.get("status") != "passed":
        raise SystemExit("SOTA adapter audit is not passing")
    output: dict[str, tuple[float, float]] = {}
    for method in ("FedEvi", "FedCLAM"):
        stats = audit["records"][method]["test_dice"]
        if int(stats["count"]) != 5:
            raise SystemExit(f"{method} audit does not contain five seeds")
        output[method] = (
            finite(stats["mean"], f"{method} mean Dice"),
            finite(stats["std"], f"{method} Dice SD"),
        )
    return output


def precision_allocation_summary(run_root: Path) -> tuple[str, list[Path]]:
    paths = sorted(
        (run_root / "unet" / PRIMARY_SCENARIO).glob("seed*/training_history.json")
    )
    if len(paths) != 5:
        raise SystemExit(
            f"expected five Progress-only histories, found {len(paths)}"
        )
    counts: dict[str, int] = {}
    total = 0
    for path in paths:
        history = load_json(path)
        if history.get("round") != list(range(80)):
            raise SystemExit(f"precision history is not an 80-round run: {path}")
        precision_rounds = history.get("client_precisions", [])
        if len(precision_rounds) != 80:
            raise SystemExit(f"client precision trace is incomplete: {path}")
        for round_values in precision_rounds:
            for precision in round_values:
                counts[str(precision)] = counts.get(str(precision), 0) + 1
                total += 1
    if total == 0:
        raise SystemExit("precision histories contain no assignments")
    parts = []
    for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        escaped_name = name.replace("_", "\\_")
        parts.append(f"{escaped_name} {100.0 * count / total:.1f}\\%")
    return (
        "Across five seeds, the progress controller issued "
        + ", ".join(parts)
        + f" over {total} selected-client assignments",
        paths,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--primary-run-root", type=Path, default=DEFAULT_PRIMARY_ROOT)
    parser.add_argument("--sota-audit", type=Path, default=DEFAULT_SOTA_AUDIT)
    parser.add_argument("--beu-boundary", type=Path, required=True)
    parser.add_argument("--credit-sensitivity", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    paper = load_json(args.paper_results)
    if paper.get("validation", {}).get("status") != "passed":
        raise SystemExit("submission result validation is not passing")
    selection = paper.get("selection", {})
    if selection.get("source_scenario") != PRIMARY_SCENARIO:
        raise SystemExit("FedMPE is not mapped from the approved Progress-only policy")
    if paper.get("seeds") != SEEDS:
        raise SystemExit("submission results do not use exactly seeds 0..4")

    sota = sota_summary(load_json(args.sota_audit))
    boundary = load_json(args.beu_boundary)
    credit = load_json(args.credit_sensitivity)
    if int(credit.get("seed_count", -1)) != 5 or int(credit.get("record_count", -1)) != 400:
        raise SystemExit("BEU credit evidence is not the required 5 x 80 records")

    fedavg = result_entry(paper, "FP32_noDP")
    fedbn = result_entry(paper, "FedBN")
    mao = result_entry(paper, "Mao_etal")
    bitfusion = result_entry(paper, "BitFusion")
    fp_operator = result_entry(paper, "FP32_softDP")
    fedmpe = result_entry(paper, "FedMPE")

    dice = {
        "FedAvg": mean_sd(fp_operator, "test_dice"),
        "FedBN": mean_sd(fedbn, "test_dice"),
        "FedEvi": sota["FedEvi"],
        "FedCLAM": sota["FedCLAM"],
        "Mao": mean_sd(mao, "test_dice"),
        "FedMPE": mean_sd(fedmpe, "test_dice"),
    }
    dice_diff, _, dice_diff_ci = paired_stats(
        [seed_metric(fedmpe, seed, "test_dice") for seed in SEEDS],
        [seed_metric(mao, seed, "test_dice") for seed in SEEDS],
    )

    mao_latency_norm = normalized(mao, "avg_latency_ms")
    mao_energy_norm = normalized(mao, "avg_local_training_energy_mJ")
    fedmpe_actual_norm = normalized(fedmpe, "avg_actual_serial_latency_ms")
    fedmpe_bound_norm = normalized(fedmpe, "avg_admission_bound_latency_ms")
    fedmpe_energy_norm = normalized(fedmpe, "avg_local_training_energy_mJ")
    fp_without_norm = normalized(fedavg, "avg_latency_ms")
    fp_operator_norm = normalized(fp_operator, "avg_latency_ms")
    bitfusion_norm = normalized(bitfusion, "avg_latency_ms")

    fedbn_latency = mean_sd(fedbn, "avg_latency_ms")
    mao_latency = mean_sd(mao, "avg_latency_ms")
    fedmpe_actual_latency = mean_sd(fedmpe, "avg_actual_serial_latency_ms")
    fedmpe_bound_latency = mean_sd(fedmpe, "avg_latency_ms")
    fedbn_energy = tuple(value / 1000.0 for value in mean_sd(fedbn, "avg_local_training_energy_mJ"))
    mao_energy = tuple(value / 1000.0 for value in mean_sd(mao, "avg_local_training_energy_mJ"))
    fedmpe_energy = tuple(value / 1000.0 for value in mean_sd(fedmpe, "avg_local_training_energy_mJ"))

    total_mean, _ = mean_sd(fedmpe, "avg_operator_total_ms")
    admitted_mean, _ = mean_sd(fedmpe, "avg_operator_admitted_ms")
    visible_mean, _ = mean_sd(fedmpe, "avg_operator_visible_bound_ms")
    if abs(total_mean - admitted_mean - visible_mean) > 1e-6:
        raise SystemExit("FedMPE operator cost accounting does not close")
    if abs(fedmpe_actual_latency[0] - fedmpe_bound_latency[0] - admitted_mean) > 1e-6:
        raise SystemExit("FedMPE serial and deadline-bound latency do not close")

    fp_cost_values = [
        seed_metric(fp_operator, seed, "avg_dp_overhead_ms") for seed in SEEDS
    ]
    fp_increment_values = [
        100.0
        * (seed_metric(fp_operator, seed, "avg_latency_ms") - seed_metric(fedavg, seed, "avg_latency_ms"))
        / seed_metric(fedavg, seed, "avg_latency_ms")
        for seed in SEEDS
    ]
    serial_visible = [
        seed_metric(fedmpe, seed, "avg_operator_total_ms") for seed in SEEDS
    ]

    allocation_summary, allocation_paths = precision_allocation_summary(
        args.primary_run_root
    )
    observed = {
        "FedAvg": dice["FedAvg"][0],
        "FedBN": dice["FedBN"][0],
        "FedEvi": dice["FedEvi"][0],
        "FedCLAM": dice["FedCLAM"][0],
        "FedMPE": dice["FedMPE"][0],
    }
    best_method, best_dice = max(observed.items(), key=lambda item: item[1])
    if best_method == "FedMPE":
        matched_summary = f"FedMPE has the highest observed mean Dice at {best_dice:.2f}\\%"
    else:
        gap = best_dice - observed["FedMPE"]
        matched_summary = (
            f"{best_method} has the highest observed mean Dice at {best_dice:.2f}\\%, "
            f"with FedMPE lower by {gap:.2f} percentage points"
        )

    latency_reduction = 100.0 * (
        mao_latency[0] - fedmpe_actual_latency[0]
    ) / mao_latency[0]
    energy_reduction = 100.0 * (
        mao_energy[0] - fedmpe_energy[0]
    ) / mao_energy[0]
    bound_gap = 100.0 * (
        fedmpe_actual_norm[0] - fedmpe_bound_norm[0]
    ) / fedmpe_actual_norm[0]

    lines = ["% Generated from audited five-seed TECS artifacts. Do not edit by hand."]
    for name, macro in (
        ("FedAvg", "FedAvgDice"),
        ("FedBN", "FedBNDice"),
        ("FedEvi", "FedEviDice"),
        ("FedCLAM", "FedCLAMDice"),
        ("Mao", "MaoDice"),
        ("FedMPE", "FedMPEDice"),
    ):
        lines.append(command(macro, text_pm(*dice[name])))
    lines.extend(
        [
            command("FedMPEVsMaoDiceDiff", f"{dice_diff:+.2f}"),
            command("FedMPEVsMaoDiceCI", f"{dice_diff_ci:.2f}"),
            command("MatchedSOTASummary", matched_summary),
            command("PrecisionAllocationSummary", allocation_summary),
            command("FedBNLatencyMs", text_pm(*fedbn_latency, digits=0)),
            command("MaoLatencyMs", text_pm(*mao_latency, digits=0)),
            command("FedMPEActualLatencyMs", text_pm(*fedmpe_actual_latency, digits=0)),
            command("FedMPEAdmissionBoundMs", text_pm(*fedmpe_bound_latency, digits=0)),
            command("FedBNEnergyJ", text_pm(*fedbn_energy, digits=2)),
            command("MaoEnergyJ", text_pm(*mao_energy, digits=2)),
            command("FedMPEEnergyJ", text_pm(*fedmpe_energy, digits=2)),
            command("FedBNVisibleOperatorCost", f"{metric(fedbn, 'avg_dp_overhead_ms'):.2f}"),
            command("MaoVisibleOperatorCost", f"{metric(mao, 'avg_dp_overhead_ms'):.2f}"),
            command("MaoLatency", math_pm(*mao_latency_norm, digits=4)),
            command("MaoEnergy", math_pm(*mao_energy_norm)),
            command("FedMPEActualLatency", math_pm(*fedmpe_actual_norm, digits=4)),
            command("FedMPEAdmissionBound", math_pm(*fedmpe_bound_norm, digits=4)),
            command("FedMPEEnergy", math_pm(*fedmpe_energy_norm)),
            command("FedMPEVsMaoLatencyReduction", f"{latency_reduction:.2f}\\%"),
            command("FedMPEVsMaoEnergyReduction", f"{energy_reduction:.2f}\\%"),
            command("FPWithoutOperatorLatency", math_pm(*fp_without_norm, digits=4)),
            command("FPWithOperatorLatency", math_pm(*fp_operator_norm, digits=4)),
            command("BitFusionLatency", math_pm(*bitfusion_norm, digits=4)),
            command("BEUAdmissionBoundGap", f"{bound_gap:.2f}\\%"),
            command("FPWithoutOperatorLatencyMs", text_pm(*mean_sd(fedavg, 'avg_latency_ms'))),
            command("FPWithOperatorLatencyMs", text_pm(*mean_sd(fp_operator, 'avg_latency_ms'))),
            command("FedMPESerialLatencyMs", text_pm(*fedmpe_actual_latency)),
            command("FPOperatorCost", f"{statistics.mean(fp_cost_values):.2f}"),
            command("FedMPEOperatorCost", f"{total_mean:.2f}"),
            command("FedMPEAdmittedCost", f"{admitted_mean:.2f}"),
            command("SerialVisibleOperatorCost", f"{statistics.mean(serial_visible):.2f}"),
            command("FPOperatorRelativeIncrement", f"{statistics.mean(fp_increment_values):.2f}"),
            command("FedMPEVisibleResidual", f"{visible_mean:.2f}"),
            command(
                "FedMPEAdmissionOutcome",
                (
                    "zero residual in the deadline bound under full admission"
                    if abs(visible_mean) < 0.005
                    else f"a {visible_mean:.2f}\\,ms residual in the admission bound"
                ),
            ),
            command(
                "BEUBoundaryMultiplier",
                f"{finite(boundary['coverage_threshold_multiplier'], 'BEU boundary'):.3f}",
            ),
            command(
                "BEUStressVisibleCost",
                f"{finite(boundary['visible_cost_ms_at_30x'], 'BEU stress visible cost'):.2f}",
            ),
            command(
                "BEUCreditThreshold",
                f"{100.0 * finite(credit['full_coverage_credit_factor'], 'credit threshold'):.2f}\\%",
            ),
            command(
                "BEUCreditMedian",
                f"{100.0 * finite(credit['median_required_credit_factor'], 'credit median'):.2f}\\%",
            ),
            command(
                "BEUCreditUpperMax",
                f"{100.0 * finite(credit['p95_required_credit_factor'], 'credit p95'):.2f}\\%/"
                f"{100.0 * finite(credit['maximum_required_credit_factor'], 'credit maximum'):.2f}\\%",
            ),
        ]
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="ascii")
    inputs = [
        args.paper_results,
        args.sota_audit,
        args.beu_boundary,
        args.credit_sensitivity,
        *allocation_paths,
    ]
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "tool": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))},
        "primary_policy": "Progress only",
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "inputs": [
            {"path": str(path.resolve()), "sha256": sha256(path)} for path in inputs
        ],
        "closure": {
            "operator_total_ms": total_mean,
            "admitted_operator_cost_ms": admitted_mean,
            "deadline_bound_visible_residual_ms": visible_mean,
            "actual_serial_latency_mean_ms": fedmpe_actual_latency[0],
            "deadline_bound_latency_mean_ms": fedmpe_bound_latency[0],
            "serial_to_bound_gap_percent": bound_gap,
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Exported audited TECS result macros to {args.output}")


if __name__ == "__main__":
    main()
