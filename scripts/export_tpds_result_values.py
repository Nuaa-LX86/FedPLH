#!/usr/bin/env python3
"""Export audited TPDS result artifacts into one LaTeX override file."""

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
DEFAULT_RUN_ROOT = (
    ROOT / "audited_runs" / "tpds_operand_complete_five_seed_20260902"
)
DEFAULT_SOTA_AUDIT = (
    ROOT / "validated_aggregate_evidence" / "sota_adapter_five_seed_audit.json"
)
DEFAULT_OUTPUT = (
    ROOT / "TPDS_final_submission" / "source" / "generated_result_values.tex"
)
DEFAULT_MANIFEST = (
    ROOT / "validated_aggregate_evidence" / "tpds_result_macro_manifest.json"
)


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
        raise SystemExit(f"paper results lack required scenario {name}") from exc
    if sorted(int(seed) for seed in entry.get("seeds", {})) != list(range(5)):
        raise SystemExit(f"{name} does not contain exactly seeds 0..4")
    return entry


def metric(entry: dict[str, Any], name: str) -> float:
    return finite(entry["metrics"][name], name)


def metric_sd(entry: dict[str, Any], name: str) -> float:
    return finite(entry["metrics_std"][name], f"{name} sample SD")


def normalized(entry: dict[str, Any], name: str) -> tuple[float, float]:
    record = entry["normalized"][name]
    values = record.get("values")
    pair_count = record.get("n")
    if pair_count is None and isinstance(values, list):
        pair_count = len(values)
    if int(pair_count if pair_count is not None else -1) != 5:
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
    return mean, sd, 1.96 * sd / math.sqrt(len(differences))


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


def acf_allocation_summary(run_root: Path) -> tuple[str, list[Path]]:
    paths = sorted((run_root / "unet" / "HMPE-ACF").glob("seed*/training_history.json"))
    if len(paths) != 5:
        raise SystemExit(f"expected five FedPLH histories for ACF allocation, found {len(paths)}")
    counts: dict[str, int] = {}
    total = 0
    for path in paths:
        history = load_json(path)
        if history.get("round") != list(range(80)):
            raise SystemExit(f"ACF allocation history is not an 80-round run: {path}")
        precision_rounds = history.get("client_precisions", [])
        if len(precision_rounds) != 80:
            raise SystemExit(f"client precision trace is incomplete: {path}")
        for round_values in precision_rounds:
            for precision in round_values:
                counts[str(precision)] = counts.get(str(precision), 0) + 1
                total += 1
    if total == 0:
        raise SystemExit("ACF precision traces contain no assignments")
    parts = []
    for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        escaped_name = name.replace("_", "\\_")
        parts.append(f"{escaped_name} {100.0 * count / total:.1f}\\%")
    return (
        "Across the five seeded schedules, ACF issued "
        + ", ".join(parts)
        + f" over {total} selected-client assignments",
        paths,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paper-results", type=Path,
        default=DEFAULT_RUN_ROOT / "unet" / "summaries" / "paper_results.json",
    )
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--sota-audit", type=Path, default=DEFAULT_SOTA_AUDIT)
    parser.add_argument("--beu-boundary", type=Path, required=True)
    parser.add_argument("--credit-sensitivity", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    paper = load_json(args.paper_results)
    if paper.get("validation", {}).get("status") != "passed":
        raise SystemExit("paper result validation is not passing")
    if paper.get("seeds") != list(range(5)):
        raise SystemExit("paper results do not use exactly seeds 0..4")
    sota = sota_summary(load_json(args.sota_audit))
    boundary = load_json(args.beu_boundary)
    credit = load_json(args.credit_sensitivity)
    if int(credit.get("seed_count", -1)) != 5 or int(credit.get("record_count", -1)) != 400:
        raise SystemExit("BEU credit evidence is not the required 5 x 80 records")

    fedavg = result_entry(paper, "FP32_noDP")
    fedbn = result_entry(paper, "FedBN")
    mao = result_entry(paper, "Mao_etal")
    bitfusion = result_entry(paper, "BitFusion")
    fp_softdp = result_entry(paper, "FP32_softDP")
    fedplh_nodp = result_entry(paper, "HMPE-ACF_noDP")
    fedplh = result_entry(paper, "HMPE-ACF")

    dice = {
        "FedAvg": (metric(fedavg, "test_dice"), metric_sd(fedavg, "test_dice")),
        "FedBN": (metric(fedbn, "test_dice"), metric_sd(fedbn, "test_dice")),
        "FedEvi": sota["FedEvi"],
        "FedCLAM": sota["FedCLAM"],
        "Mao": (metric(mao, "test_dice"), metric_sd(mao, "test_dice")),
        "FedPLH": (metric(fedplh, "test_dice"), metric_sd(fedplh, "test_dice")),
    }
    fedplh_seed_dice = [seed_metric(fedplh, seed, "test_dice") for seed in range(5)]
    mao_seed_dice = [seed_metric(mao, seed, "test_dice") for seed in range(5)]
    dice_diff, _, dice_diff_ci = paired_stats(fedplh_seed_dice, mao_seed_dice)

    latency_norm = {
        name: normalized(entry, "avg_latency_ms")
        for name, entry in {
            "FedBN": fedbn, "Mao": mao, "FedPLH": fedplh,
            "FPNoDP": fedavg, "FPSoftDP": fp_softdp,
            "BitFusion": bitfusion, "FedPLHNoDP": fedplh_nodp,
        }.items()
    }
    energy_norm = {
        name: normalized(entry, "avg_local_training_energy_mJ")
        for name, entry in {"Mao": mao, "FedPLH": fedplh}.items()
    }

    serial_latency = []
    serial_normalized = []
    serial_visible = []
    for seed in range(5):
        current = seed_metric(fedplh, seed, "avg_latency_ms")
        covered = seed_metric(fedplh, seed, "avg_dp_background_ms")
        serial = current + covered
        serial_latency.append(serial)
        serial_visible.append(
            seed_metric(fedplh, seed, "avg_dp_overhead_ms") + covered
        )
        serial_normalized.append(
            serial / seed_metric(fedbn, seed, "avg_latency_ms")
        )
    serial_latency_mean = statistics.mean(serial_latency)
    serial_latency_sd = statistics.stdev(serial_latency)
    serial_norm_mean = statistics.mean(serial_normalized)
    serial_norm_sd = statistics.stdev(serial_normalized)
    beu_direct_latency_reduction = 100.0 * (
        serial_norm_mean - latency_norm["FedPLH"][0]
    ) / serial_norm_mean

    fp_cost_values = [
        seed_metric(fp_softdp, seed, "avg_dp_overhead_ms") for seed in range(5)
    ]
    fp_increment_values = [
        100.0
        * (
            seed_metric(fp_softdp, seed, "avg_latency_ms")
            - seed_metric(fedavg, seed, "avg_latency_ms")
        )
        / seed_metric(fedavg, seed, "avg_latency_ms")
        for seed in range(5)
    ]

    acf_summary, acf_paths = acf_allocation_summary(args.run_root)
    observed = {
        "FedAvg": dice["FedAvg"][0], "FedBN": dice["FedBN"][0],
        "FedEvi": dice["FedEvi"][0], "FedCLAM": dice["FedCLAM"][0],
        "FedPLH": dice["FedPLH"][0],
    }
    best_method, best_dice = max(observed.items(), key=lambda item: item[1])
    if best_method == "FedPLH":
        matched_summary = (
            f"FedPLH has the highest observed mean Dice at {best_dice:.2f}\\%"
        )
    else:
        gap = best_dice - observed["FedPLH"]
        matched_summary = (
            f"{best_method} has the highest observed mean Dice at {best_dice:.2f}\\%, "
            f"with FedPLH lower by {gap:.2f} percentage points"
        )

    def mean_sd(entry: dict[str, Any], name: str) -> tuple[float, float]:
        return metric(entry, name), metric_sd(entry, name)

    fedbn_latency = mean_sd(fedbn, "avg_latency_ms")
    mao_latency = mean_sd(mao, "avg_latency_ms")
    fedplh_latency = mean_sd(fedplh, "avg_latency_ms")
    fedbn_energy = tuple(value / 1000.0 for value in mean_sd(fedbn, "avg_local_training_energy_mJ"))
    mao_energy = tuple(value / 1000.0 for value in mean_sd(mao, "avg_local_training_energy_mJ"))
    fedplh_energy = tuple(value / 1000.0 for value in mean_sd(fedplh, "avg_local_training_energy_mJ"))

    visible_mean, visible_sd = mean_sd(fedplh, "avg_dp_overhead_ms")
    total_mean, total_sd = mean_sd(fedplh, "avg_dp_total_ms")
    covered_mean, covered_sd = mean_sd(fedplh, "avg_dp_background_ms")
    if abs(total_mean - visible_mean - covered_mean) > 1e-6:
        raise SystemExit("FedPLH visible and covered SoftDP costs do not close")

    lines = ["% Generated from audited five-seed result artifacts. Do not edit by hand."]
    for name, macro in (
        ("FedAvg", "FedAvgDice"), ("FedBN", "FedBNDice"),
        ("FedEvi", "FedEviDice"), ("FedCLAM", "FedCLAMDice"),
        ("Mao", "MaoDice"), ("FedPLH", "FedPLHDice"),
    ):
        lines.append(command(macro, text_pm(*dice[name])))
    lines.extend([
        command("FedPLHVsMaoDiceDiff", f"{dice_diff:+.2f}"),
        command("FedPLHVsMaoDiceCI", f"{dice_diff_ci:.2f}"),
        command("MatchedSOTASummary", matched_summary),
        command("ACFAllocationSummary", acf_summary),
        command("FedBNLatencyMs", text_pm(*fedbn_latency)),
        command("MaoLatencyMs", text_pm(*mao_latency)),
        command("FedPLHLatencyMs", text_pm(*fedplh_latency)),
        command("FedBNEnergyJ", text_pm(*fedbn_energy, digits=3)),
        command("MaoEnergyJ", text_pm(*mao_energy, digits=3)),
        command("FedPLHEnergyJ", text_pm(*fedplh_energy, digits=3)),
        command("FedBNVisibleCost", f"{metric(fedbn, 'avg_dp_overhead_ms'):.2f}"),
        command("MaoVisibleCost", f"{metric(mao, 'avg_dp_overhead_ms'):.2f}"),
        command("MaoLatency", math_pm(*latency_norm["Mao"], digits=4)),
        command("MaoEnergy", math_pm(*energy_norm["Mao"])),
        command("FedPLHLatency", math_pm(*latency_norm["FedPLH"], digits=4)),
        command("FedPLHEnergy", math_pm(*energy_norm["FedPLH"])),
        command("FedPLHVsMaoLatencyDelta", f"{fedplh_latency[0] - mao_latency[0]:+.2f}"),
        command("FedPLHVsMaoEnergyDelta", f"{fedplh_energy[0] - mao_energy[0]:+.3f}"),
        command("FPNoDPLatency", math_pm(*latency_norm["FPNoDP"], digits=4)),
        command("FPSoftDPLatency", math_pm(*latency_norm["FPSoftDP"], digits=4)),
        command("BitFusionLatency", math_pm(*latency_norm["BitFusion"], digits=4)),
        command("FedPLHNoDPLatency", math_pm(*latency_norm["FedPLHNoDP"], digits=4)),
        command("SerialLatency", math_pm(serial_norm_mean, serial_norm_sd, digits=4)),
        command("BEUDirectLatencyReduction", f"{beu_direct_latency_reduction:.2f}\\%"),
        command("FPNoDPLatencyMs", text_pm(*mean_sd(fedavg, "avg_latency_ms"))),
        command("FPSoftDPLatencyMs", text_pm(*mean_sd(fp_softdp, "avg_latency_ms"))),
        command("FedPLHNoDPLatencyMs", text_pm(*mean_sd(fedplh_nodp, "avg_latency_ms"))),
        command("SerialLatencyMs", text_pm(serial_latency_mean, serial_latency_sd)),
        command("FPSoftDPCost", f"{statistics.mean(fp_cost_values):.2f}"),
        command("FedPLHSoftDPCost", f"{total_mean:.2f}"),
        command("FedPLHCoveredCost", f"{covered_mean:.2f}"),
        command("SerialVisibleCost", f"{statistics.mean(serial_visible):.2f}"),
        command("FPSoftDPRelativeIncrement", f"{statistics.mean(fp_increment_values):.2f}"),
        command("FedPLHVisibleCost", f"{visible_mean:.2f}"),
        command(
            "FedPLHCoverageOutcome",
            (
                "zero uncovered SoftDP increment"
                if abs(visible_mean) < 0.005
                else f"a {visible_mean:.2f}\\,ms uncovered SoftDP increment"
            ),
        ),
        command("BEUBoundaryMultiplier", f"{finite(boundary['coverage_threshold_multiplier'], 'BEU boundary'):.3f}"),
        command("BEUStressVisibleCost", f"{finite(boundary['visible_cost_ms_at_30x'], 'BEU 30x visible cost'):.2f}"),
        command("BEUCreditThreshold", f"{100.0 * finite(credit['full_coverage_credit_factor'], 'credit threshold'):.2f}\\%"),
        command("BEUCreditMedian", f"{100.0 * finite(credit['median_required_credit_factor'], 'credit median'):.2f}\\%"),
        command(
            "BEUCreditPFull",
            f"{100.0 * finite(credit['p95_required_credit_factor'], 'credit p95'):.2f}\\%/"
            f"{100.0 * finite(credit['maximum_required_credit_factor'], 'credit maximum'):.2f}\\%",
        ),
    ])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="ascii")
    inputs = [
        args.paper_results, args.sota_audit, args.beu_boundary,
        args.credit_sensitivity, *acf_paths,
    ]
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "tool": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__)),
        },
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "inputs": [
            {"path": str(path.resolve()), "sha256": sha256(path)} for path in inputs
        ],
        "closure": {
            "fedplh_softdp_total_ms": total_mean,
            "fedplh_visible_ms": visible_mean,
            "fedplh_covered_ms": covered_mean,
            "serial_latency_mean_ms": serial_latency_mean,
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Exported audited TPDS result macros to {args.output}")


if __name__ == "__main__":
    main()
