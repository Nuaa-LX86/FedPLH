import argparse
import csv
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


def _ci95(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return 0.0
    critical = T_CRITICAL_975.get(len(values) - 1, 1.96)
    return float(
        critical * values.std(ddof=1) / math.sqrt(len(values))
    )


def _summary(values):
    values = [float(value) for value in values]
    return {
        "n": len(values),
        "mean": float(np.mean(values)) if values else None,
        "std": (
            float(np.std(values, ddof=1))
            if len(values) > 1
            else 0.0
        ),
        "ci95_half_width": _ci95(values),
        "values": values,
    }


def _rankdata(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while (
            end < len(values)
            and values[order[end]] == values[order[start]]
        ):
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _correlations(first, second):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(
        [
            np.nan if value is None else float(value)
            for value in second
        ],
        dtype=np.float64,
    )
    mask = np.isfinite(first) & np.isfinite(second)
    first = first[mask]
    second = second[mask]
    if len(first) < 2:
        return {"pearson": None, "spearman": None, "n": len(first)}
    return {
        "pearson": float(np.corrcoef(first, second)[0, 1]),
        "spearman": float(
            np.corrcoef(_rankdata(first), _rankdata(second))[0, 1]
        ),
        "n": int(len(first)),
    }


def _load_history(path):
    with path.open("r", encoding="utf-8") as handle:
        history = json.load(handle)
    rounds = len(history.get("round", []))
    if rounds != 80:
        raise ValueError(f"{path} has {rounds} rounds; expected 80")
    if int(history.get("local_epochs", -1)) != 2:
        raise ValueError(f"{path} does not use two local epochs")
    residuals = np.abs(history.get("latency_residual_ms", []))
    if len(residuals) and float(np.max(residuals)) > 1e-6:
        raise ValueError(f"{path} has a non-closing latency breakdown")
    epsilon = np.asarray(history.get("epsilon", []), dtype=np.float64)
    if len(epsilon) > 1 and float(np.min(np.diff(epsilon))) < -1e-9:
        raise ValueError(f"{path} has decreasing cumulative epsilon")
    return history


def _collect_stage(stage_name, stage_root):
    records = []
    if not stage_root.exists():
        raise FileNotFoundError(stage_root)
    for history_path in sorted(
        (stage_root / "unet").rglob("training_history.json")
    ):
        relative = history_path.relative_to(stage_root / "unet")
        scenario = relative.parts[0]
        seed_part = next(
            part for part in relative.parts if part.startswith("seed")
        )
        split_parts = [
            part for part in relative.parts if part.startswith("split")
        ]
        seed = int(seed_part.replace("seed", ""))
        split_seed = (
            int(split_parts[0].replace("split", ""))
            if split_parts
            else 0
        )
        history = _load_history(history_path)
        records.append({
            "stage": stage_name,
            "scenario": scenario,
            "seed": seed,
            "split_seed": split_seed,
            "path": str(history_path.resolve()),
            "history": history,
        })
    return records


def _index(records):
    return {
        (
            record["stage"],
            record["scenario"],
            record["split_seed"],
            record["seed"],
        ): record
        for record in records
    }


def _paired_metric(records_by_key, pairs, metric):
    differences = []
    rows = []
    for first_key, second_key in pairs:
        first = records_by_key[first_key]["history"]["metrics"][metric]
        second = records_by_key[second_key]["history"]["metrics"][metric]
        difference = float(first) - float(second)
        differences.append(difference)
        rows.append({
            "first": first_key,
            "second": second_key,
            "first_value": float(first),
            "second_value": float(second),
            "difference": difference,
        })
    return {"summary": _summary(differences), "pairs": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = []
    records.extend(_collect_stage("main", args.root / "main"))
    records.extend(_collect_stage("ablation", args.root / "ablation"))
    for split_seed in (0, 1, 2):
        records.extend(
            _collect_stage(
                f"stress_split{split_seed}",
                args.root / f"stress_split{split_seed}",
            )
        )
    records_by_key = _index(records)

    scenario_metrics = {}
    for stage in sorted({record["stage"] for record in records}):
        for scenario in sorted({
            record["scenario"]
            for record in records
            if record["stage"] == stage
        }):
            selected = [
                record for record in records
                if record["stage"] == stage
                and record["scenario"] == scenario
            ]
            scenario_metrics[f"{stage}/{scenario}"] = {
                metric: _summary([
                    record["history"]["metrics"][metric]
                    for record in selected
                ])
                for metric in [
                    "test_dice",
                    "avg_latency_ms",
                    "avg_energy_mJ",
                    "high_precision_assignment_rate",
                ]
            }

    mechanism_comparisons = {}
    for ablation_scenario in [
        "ACF_static_FP8",
        "ACF_progress_only",
        "ACF_entropy_only",
    ]:
        pairs = [
            (
                ("main", "ACF_full", 0, seed),
                ("ablation", ablation_scenario, 0, seed),
            )
            for seed in (0, 1, 2)
        ]
        mechanism_comparisons[f"ACF_full_minus_{ablation_scenario}"] = {
            metric: _paired_metric(
                records_by_key,
                pairs,
                metric,
            )
            for metric in [
                "test_dice",
                "avg_latency_ms",
                "avg_energy_mJ",
            ]
        }

    stress_comparisons = {}
    for reference in ["ACF_progress_only", "ACF_FedBN"]:
        pairs = [
            (
                (
                    f"stress_split{split_seed}",
                    "ACF_full",
                    split_seed,
                    split_seed,
                ),
                (
                    f"stress_split{split_seed}",
                    reference,
                    split_seed,
                    split_seed,
                ),
            )
            for split_seed in (0, 1, 2)
        ]
        stress_comparisons[f"ACF_full_minus_{reference}"] = {
            metric: _paired_metric(
                records_by_key,
                pairs,
                metric,
            )
            for metric in [
                "test_dice",
                "avg_latency_ms",
                "avg_energy_mJ",
            ]
        }

    assignment_correlations = []
    for seed in (0, 1, 2, 3, 4):
        history = records_by_key[
            ("main", "ACF_full", 0, seed)
        ]["history"]
        assignment_correlations.append({
            "seed": seed,
            **_correlations(
                history["client_entropies"],
                history["metrics"][
                    "high_precision_assignment_rate_by_client"
                ],
            ),
        })

    output = {
        "schema_version": 1,
        "root": str(args.root.resolve()),
        "validated_run_count": len(records),
        "validation": {
            "rounds": 80,
            "local_epochs": 2,
            "latency_breakdown_closed": True,
            "cumulative_epsilon_nondecreasing": True,
        },
        "scenario_metrics": scenario_metrics,
        "mechanism_comparisons": mechanism_comparisons,
        "strong_heterogeneity_comparisons": stress_comparisons,
        "entropy_assignment_correlations": assignment_correlations,
    }
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
                "split_seed",
                "seed",
                "test_dice",
                "avg_latency_ms",
                "avg_energy_mJ",
                "high_precision_assignment_rate",
                "history_path",
            ],
        )
        writer.writeheader()
        for record in records:
            metrics = record["history"]["metrics"]
            writer.writerow({
                "stage": record["stage"],
                "scenario": record["scenario"],
                "split_seed": record["split_seed"],
                "seed": record["seed"],
                "test_dice": metrics["test_dice"],
                "avg_latency_ms": metrics["avg_latency_ms"],
                "avg_energy_mJ": metrics["avg_energy_mJ"],
                "high_precision_assignment_rate": metrics[
                    "high_precision_assignment_rate"
                ],
                "history_path": record["path"],
            })


if __name__ == "__main__":
    main()
