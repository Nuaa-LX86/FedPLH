from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.dp_sgd import (
    DEFAULT_RDP_ORDERS,
    LEGACY_RDP_ORDERS,
    RDPAccountant,
)
from utils.reproducibility import sha256_file, write_json_atomic


DEFAULT_SCENARIOS = [
    "FP32_noDP",
    "FP32_softDP",
    "FedBN",
    "FedPAQ",
    "Mao_etal",
    "BitFusion",
    "HMPE-ACF_noDP",
    "HMPE-ACF",
]

SUMMARY_METRICS = [
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
]


def parse_int_list(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_str_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def sample_stats(values: Iterable[float]) -> Dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"mean": 0.0, "std": 0.0, "ci95": 0.0, "values": []}
    std = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    return {
        "mean": float(np.mean(array)),
        "std": std,
        "ci95": float(1.96 * std / np.sqrt(array.size)),
        "values": [float(value) for value in array],
    }


def load_seed_result(results_dir: Path, scenario: str, seed: int) -> Dict[str, Any]:
    seed_dir = results_dir / scenario / f"seed{seed}"
    history_path = seed_dir / "training_history.json"
    manifest_path = seed_dir / "run_manifest.json"
    if not history_path.is_file():
        raise FileNotFoundError(f"Missing history: {history_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing run manifest: {manifest_path}")

    with history_path.open("r", encoding="utf-8") as handle:
        history = json.load(handle)
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    if manifest.get("status") != "completed":
        raise ValueError(f"Incomplete run manifest: {manifest_path}")
    if int(manifest.get("seed", -1)) != seed:
        raise ValueError(f"Seed mismatch in {manifest_path}")
    if manifest.get("scenario") != scenario:
        raise ValueError(f"Scenario mismatch in {manifest_path}")
    if "metrics" not in history:
        raise ValueError(f"Missing metrics in {history_path}")

    return {
        "history": history,
        "manifest": manifest,
        "metrics": history["metrics"],
    }


def validate_history(
    scenario: str,
    seed: int,
    result: Dict[str, Any],
    schedule_reference: Dict[int, List[List[int]]],
    epsilon_reference: Dict[tuple, List[float]],
) -> List[str]:
    history = result["history"]
    manifest = result["manifest"]
    warnings = []

    residuals = np.asarray(history.get("latency_residual_ms", []), dtype=np.float64)
    if residuals.size == 0 or float(np.max(np.abs(residuals))) > 1e-6:
        raise ValueError(f"Latency does not close for {scenario}, seed {seed}")

    epsilon = np.asarray(history.get("epsilon", []), dtype=np.float64)
    if epsilon.size > 1 and float(np.min(np.diff(epsilon))) < -1e-9:
        raise ValueError(f"Epsilon decreases for {scenario}, seed {seed}")

    aggregation_weights = history.get("aggregation_weights")
    aggregation_flags = history.get("is_aggregation_round")
    participating = history.get("participating_clients")
    client_sizes = history.get("client_num_samples")
    optimizer_steps = history.get("client_optimizer_steps")
    privacy_events = history.get("client_privacy_events")
    if aggregation_weights is not None:
        if not (
            len(aggregation_weights)
            == len(aggregation_flags)
            == len(participating)
        ):
            raise ValueError(
                f"Aggregation audit lengths differ for {scenario}, seed {seed}"
            )
        for round_index, (is_aggregation, client_ids, weights) in enumerate(
            zip(aggregation_flags, participating, aggregation_weights)
        ):
            if not is_aggregation:
                if weights:
                    raise ValueError(
                        f"Non-aggregation round {round_index} records weights "
                        f"for {scenario}, seed {seed}"
                    )
                continue
            expected_counts = [float(client_sizes[int(cid)]) for cid in client_ids]
            denominator = sum(expected_counts)
            expected_weights = [
                count / denominator for count in expected_counts
            ]
            if not np.allclose(
                weights,
                expected_weights,
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(
                    f"Incorrect aggregation weights at round {round_index} "
                    f"for {scenario}, seed {seed}"
                )
    if optimizer_steps is not None:
        if len(optimizer_steps) != len(participating):
            raise ValueError(
                f"Optimizer-step audit length differs for {scenario}, seed {seed}"
            )
        for round_index, (client_ids, step_counts) in enumerate(
            zip(participating, optimizer_steps)
        ):
            if len(client_ids) != len(step_counts):
                raise ValueError(
                    f"Optimizer-step/client count mismatch at round "
                    f"{round_index} for {scenario}, seed {seed}"
                )
            if any(int(value) <= 0 for value in step_counts):
                raise ValueError(
                    f"Non-positive optimizer steps at round {round_index} "
                    f"for {scenario}, seed {seed}"
                )
    if privacy_events is not None and optimizer_steps is not None:
        if len(privacy_events) != len(optimizer_steps):
            raise ValueError(
                f"Privacy-event audit length differs for {scenario}, seed {seed}"
            )
        dp_enabled_for_event_check = bool(
            manifest.get("scenario_config", {}).get("dp", {}).get("enable")
        )
        for round_index, (round_events, step_counts) in enumerate(
            zip(privacy_events, optimizer_steps)
        ):
            if len(round_events) != len(step_counts):
                raise ValueError(
                    f"Privacy-event/client count mismatch at round "
                    f"{round_index} for {scenario}, seed {seed}"
                )
            if dp_enabled_for_event_check:
                for events, steps in zip(round_events, step_counts):
                    if len(events) != int(steps):
                        raise ValueError(
                            f"Privacy events do not match optimizer steps at "
                            f"round {round_index} for {scenario}, seed {seed}"
                        )

    energy_scope = result["metrics"].get("energy_scope")
    if energy_scope is not None and "local-training" not in energy_scope:
        warnings.append(
            f"Unexpected energy scope for {scenario}, seed {seed}"
        )

    schedule = manifest.get("client_schedule")
    if seed not in schedule_reference:
        schedule_reference[seed] = schedule
    elif schedule_reference[seed] != schedule:
        raise ValueError(f"Client schedule mismatch for {scenario}, seed {seed}")

    dp_enabled = bool(manifest.get("scenario_config", {}).get("dp", {}).get("enable"))
    if dp_enabled:
        epsilon_values = [float(value) for value in history.get("epsilon", [])]
        effective_schedule = history.get("participating_clients", schedule)
        effective_privacy_events = history.get("client_privacy_events")
        epsilon_key = (
            int(seed),
            json.dumps(effective_schedule, separators=(",", ":")),
            json.dumps(
                effective_privacy_events,
                separators=(",", ":"),
                sort_keys=True,
            ),
        )
        if epsilon_key not in epsilon_reference:
            epsilon_reference[epsilon_key] = epsilon_values
        elif not np.allclose(
            epsilon_reference[epsilon_key],
            epsilon_values,
            rtol=0.0,
            atol=1e-9,
        ):
            raise ValueError(
                f"DP methods with the same effective schedule have different "
                f"accountant curves: "
                f"{scenario}, seed {seed}"
            )

    if result["metrics"].get("privacy_accounting_scope") not in (
        None,
        "cumulative maximum over all clients",
    ):
        warnings.append(f"Unexpected accounting scope for {scenario}, seed {seed}")
    return warnings


def reconstruct_accounting(
    results_dir: Path,
    loaded: Dict[str, Dict[int, Dict[str, Any]]],
    scenarios: List[str],
    seeds: List[int],
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    root_manifest_path = results_dir.parent / "run_manifest.json"
    with root_manifest_path.open("r", encoding="utf-8") as handle:
        root_manifest = json.load(handle)
    arguments = root_manifest["arguments"]
    batch_size = int(arguments["batch_size"])
    local_epochs = int(arguments["local_epochs"])
    delta = float(arguments["delta"])
    reconstructed: Dict[str, Dict[int, Dict[str, Any]]] = {}

    for scenario in scenarios:
        reconstructed[scenario] = {}
        for seed in seeds:
            manifest = loaded[scenario][seed]["manifest"]
            dp_enabled = bool(
                manifest.get("scenario_config", {})
                .get("dp", {})
                .get("enable")
            )
            if not dp_enabled:
                continue

            history = loaded[scenario][seed]["history"]
            client_sizes = [
                int(value)
                for value in history.get("client_num_samples", [])
            ]
            if not client_sizes:
                raise ValueError(
                    f"Missing client_num_samples for {scenario}, seed {seed}"
                )
            schedule = history.get("participating_clients", [])
            if not schedule:
                raise ValueError(
                    f"Missing effective participating-client log for "
                    f"{scenario}, seed {seed}"
                )
            noise_multiplier = float(
                manifest["scenario_config"]["dp"]["noise_multiplier"]
            )
            recorded_privacy_events = history.get("client_privacy_events")
            accountants = [RDPAccountant() for _ in client_sizes]
            legacy_accountants = [
                RDPAccountant(LEGACY_RDP_ORDERS)
                for _ in client_sizes
            ]
            epsilon_per_client = [0.0 for _ in client_sizes]
            epsilon_curve = []
            epsilon_participating_max = []
            epsilon_per_client_curve = []

            for round_index, participating_clients in enumerate(schedule):
                for client_offset, client_id in enumerate(
                    participating_clients
                ):
                    client_id = int(client_id)
                    local_size = client_sizes[client_id]
                    if recorded_privacy_events is not None:
                        events = recorded_privacy_events[
                            round_index
                        ][client_offset]
                        for event in events:
                            for accountant_group in (
                                accountants,
                                legacy_accountants,
                            ):
                                accountant_group[client_id].add_segment(
                                    float(event["sample_rate"]),
                                    float(event["noise_multiplier"]),
                                    1,
                                )
                    else:
                        full_batches, remainder = divmod(
                            local_size,
                            batch_size,
                        )
                        for _ in range(local_epochs):
                            if full_batches:
                                for accountant_group in (
                                    accountants,
                                    legacy_accountants,
                                ):
                                    accountant_group[client_id].add_segment(
                                        batch_size / float(local_size),
                                        noise_multiplier,
                                        full_batches,
                                    )
                            if remainder:
                                for accountant_group in (
                                    accountants,
                                    legacy_accountants,
                                ):
                                    accountant_group[client_id].add_segment(
                                        remainder / float(local_size),
                                        noise_multiplier,
                                        1,
                                    )
                    epsilon_per_client[client_id] = accountants[
                        client_id
                    ].get_epsilon(delta)

                epsilon_curve.append(float(max(epsilon_per_client)))
                epsilon_participating_max.append(
                    float(
                        max(
                            epsilon_per_client[int(client_id)]
                            for client_id in participating_clients
                        )
                    )
                )
                epsilon_per_client_curve.append(list(epsilon_per_client))

            selected_orders = []
            for accountant in accountants:
                _, selected_order = accountant.get_epsilon_and_order(delta)
                if selected_order is not None:
                    selected_orders.append(selected_order)
            if selected_orders and (
                min(selected_orders) <= min(DEFAULT_RDP_ORDERS)
                or max(selected_orders) >= max(DEFAULT_RDP_ORDERS)
            ):
                raise ValueError(
                    f"Optimal RDP order is on the expanded grid boundary for "
                    f"{scenario}, seed {seed}: {min(selected_orders)}.."
                    f"{max(selected_orders)}"
                )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                legacy_epsilon_per_client = [
                    accountant.get_epsilon(delta)
                    for accountant in legacy_accountants
                ]

            actual = history.get("epsilon_per_client", [])
            if not actual:
                raise ValueError(
                    f"Missing epsilon_per_client for {scenario}, seed {seed}"
                )
            if np.allclose(
                epsilon_per_client,
                actual[-1],
                rtol=0.0,
                atol=1e-9,
            ):
                recorded_grid = "expanded"
            elif np.allclose(
                legacy_epsilon_per_client,
                actual[-1],
                rtol=0.0,
                atol=1e-9,
            ):
                recorded_grid = "legacy"
            else:
                raise ValueError(
                    f"Recorded accountant does not match actual optimizer "
                    f"updates for {scenario}, seed {seed}"
                )

            loaded[scenario][seed]["metrics"] = dict(
                loaded[scenario][seed]["metrics"]
            )
            loaded[scenario][seed]["metrics"]["final_epsilon"] = float(
                epsilon_curve[-1]
            )
            loaded[scenario][seed]["metrics"][
                "privacy_accounting_scope"
            ] = "cumulative maximum over all clients"

            reconstructed[scenario][seed] = {
                "epsilon": epsilon_curve,
                "epsilon_participating_max": epsilon_participating_max,
                "epsilon_per_client": epsilon_per_client_curve,
                "recorded_grid": recorded_grid,
                "selected_order_min": (
                    float(min(selected_orders)) if selected_orders else None
                ),
                "selected_order_max": (
                    float(max(selected_orders)) if selected_orders else None
                ),
            }

    return reconstructed


def build_summary(
    results_dir: Path,
    scenarios: List[str],
    seeds: List[int],
    baseline: str,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    loaded: Dict[str, Dict[int, Dict[str, Any]]] = {}
    schedule_reference: Dict[int, List[List[int]]] = {}
    epsilon_reference: Dict[tuple, List[float]] = {}
    warnings: List[str] = []

    for scenario in scenarios:
        loaded[scenario] = {}
        for seed in seeds:
            result = load_seed_result(results_dir, scenario, seed)
            warnings.extend(
                validate_history(
                    scenario,
                    seed,
                    result,
                    schedule_reference,
                    epsilon_reference,
                )
            )
            loaded[scenario][seed] = result

    if baseline not in loaded:
        raise ValueError(f"Baseline {baseline} is not included")

    reconstructed_accounting = reconstruct_accounting(
        results_dir,
        loaded,
        scenarios,
        seeds,
    )

    summary: Dict[str, Any] = {}
    for scenario in scenarios:
        seed_metrics = {
            str(seed): loaded[scenario][seed]["metrics"]
            for seed in seeds
        }
        metrics = {}
        metrics_std = {}
        metrics_ci95 = {}
        for metric in SUMMARY_METRICS:
            values = [
                loaded[scenario][seed]["metrics"].get(metric)
                for seed in seeds
            ]
            numeric_values = [
                float(value)
                for value in values
                if isinstance(value, (int, float)) and np.isfinite(value)
            ]
            if numeric_values:
                stats = sample_stats(numeric_values)
                metrics[metric] = stats["mean"]
                metrics_std[metric] = stats["std"]
                metrics_ci95[metric] = stats["ci95"]

        normalized = {}
        for metric in (
            "avg_latency_ms",
            "avg_local_training_energy_mJ",
        ):
            ratios = []
            for seed in seeds:
                numerator_value = loaded[scenario][seed]["metrics"].get(metric)
                denominator_value = loaded[baseline][seed]["metrics"].get(metric)
                if numerator_value is None or denominator_value is None:
                    continue
                numerator = float(numerator_value)
                denominator = float(denominator_value)
                if denominator <= 0:
                    raise ValueError(
                        f"Non-positive baseline {metric} for seed {seed}"
                    )
                ratios.append(numerator / denominator)
            if ratios:
                normalized[metric] = sample_stats(ratios)

        summary[scenario] = {
            "metrics": metrics,
            "metrics_std": metrics_std,
            "metrics_ci95": metrics_ci95,
            "normalized": normalized,
            "seeds": seed_metrics,
            "scenario_config": loaded[scenario][seeds[0]]["manifest"][
                "scenario_config"
            ],
        }

    paper_results = {
        "schema_version": 1,
        "postprocessing": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_sha256": {
                path.as_posix(): sha256_file(
                    Path(__file__).resolve().parents[1] / path
                )
                for path in (
                    Path("scripts/build_paper_results.py"),
                    Path("training/dp_sgd.py"),
                    Path("visualization/plot_generator.py"),
                    Path("plot_beu_boundary.py"),
                )
            },
        },
        "baseline": baseline,
        "normalization": "paired by training seed, then mean and sample SD",
        "error_bars": "sample standard deviation across training seeds",
        "seeds": seeds,
        "privacy_accounting": {
            "curve": "cumulative maximum over all clients",
            "orders_grid": {
                "minimum": float(min(DEFAULT_RDP_ORDERS)),
                "maximum": float(max(DEFAULT_RDP_ORDERS)),
                "count": len(DEFAULT_RDP_ORDERS),
            },
            "recorded_grid_by_scenario_seed": {
                scenario: {
                    str(seed): reconstructed_accounting[scenario][seed][
                        "recorded_grid"
                    ]
                    for seed in seeds
                    if seed in reconstructed_accounting[scenario]
                }
                for scenario in scenarios
            },
            "selected_order_range_by_scenario_seed": {
                scenario: {
                    str(seed): [
                        reconstructed_accounting[scenario][seed][
                            "selected_order_min"
                        ],
                        reconstructed_accounting[scenario][seed][
                            "selected_order_max"
                        ],
                    ]
                    for seed in seeds
                    if seed in reconstructed_accounting[scenario]
                }
                for scenario in scenarios
            },
            "mechanism": "soft gradient perturbation",
            "formal_dp_guarantee": False,
            "limitation": (
                "The optimizer does not perform per-sample clipping; the "
                "reported epsilon is a nominal RDP-accountant output, not a "
                "formal (epsilon, delta)-DP guarantee."
            ),
        },
        "validation": {
            "status": "passed",
            "latency_closure_tolerance_ms": 1e-6,
            "epsilon_monotonic": True,
            "matched_effective_schedule_dp_accountant_curves": True,
            "accountant_matches_scheduled_steps": True,
            "accountant_order_optimum_interior": True,
            "shared_base_client_schedule": True,
            "method_specific_effective_schedule": (
                "FedPAQ uses the scheduled cohort in every communication round"
            ),
            "warnings": warnings,
        },
        "results": summary,
    }

    representative_scenario = (
        "HMPE-ACF" if "HMPE-ACF" in loaded else scenarios[0]
    )
    representative_seed = seeds[0]
    representative_history = copy.deepcopy(
        loaded[representative_scenario][representative_seed]["history"]
    )
    corrected = reconstructed_accounting.get(
        representative_scenario,
        {},
    ).get(representative_seed)
    if corrected is not None:
        representative_history["epsilon"] = corrected["epsilon"]
        representative_history["epsilon_participating_max"] = corrected[
            "epsilon_participating_max"
        ]
        representative_history["epsilon_per_client"] = corrected[
            "epsilon_per_client"
        ]
    representative_history["metrics"] = dict(
        loaded[representative_scenario][representative_seed]["metrics"]
    )
    representative_history["privacy_accounting"] = paper_results[
        "privacy_accounting"
    ]
    return paper_results, representative_history


def write_csv(path: Path, paper_results: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
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
                "local_training_energy_mean_mJ",
                "local_training_energy_sd_mJ",
                "normalized_energy_mean",
                "normalized_energy_sd",
                "final_epsilon_mean",
                "final_epsilon_sd",
            ]
        )
        for method, entry in paper_results["results"].items():
            metrics = entry["metrics"]
            std = entry["metrics_std"]
            energy_metric = "avg_local_training_energy_mJ"
            energy_normalized = entry["normalized"][energy_metric]
            writer.writerow(
                [
                    method,
                    metrics.get("test_dice", ""),
                    std.get("test_dice", ""),
                    metrics.get("avg_latency_ms", ""),
                    std.get("avg_latency_ms", ""),
                    entry["normalized"]["avg_latency_ms"]["mean"],
                    entry["normalized"]["avg_latency_ms"]["std"],
                    metrics.get(energy_metric, ""),
                    std.get(energy_metric, ""),
                    energy_normalized["mean"],
                    energy_normalized["std"],
                    metrics.get("final_epsilon", ""),
                    std.get("final_epsilon", ""),
                ]
            )


def write_table_rows(path: Path, paper_results: Dict[str, Any]) -> None:
    display_names = {
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
        energy_metric = "avg_local_training_energy_mJ"
        energy = entry["normalized"][energy_metric]
        rows.append(
            f"{display_names.get(method, method)} & "
            f"{metrics['test_dice']:.2f}$\\pm${std['test_dice']:.2f} & "
            f"{latency['mean']:.3f}$\\pm${latency['std']:.3f} & "
            f"{energy['mean']:.3f}$\\pm${energy['std']:.3f} \\\\"
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--scenarios", default=",".join(DEFAULT_SCENARIOS))
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--baseline", default="FedBN")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    scenarios = parse_str_list(args.scenarios)
    seeds = parse_int_list(args.seeds)
    paper_results, representative_history = build_summary(
        results_dir,
        scenarios,
        seeds,
        args.baseline,
    )

    summaries_dir = results_dir / "summaries"
    write_json_atomic(summaries_dir / "paper_results.json", paper_results)
    write_json_atomic(
        summaries_dir / "representative_history.json",
        representative_history,
    )
    write_csv(summaries_dir / "paper_results.csv", paper_results)
    write_table_rows(summaries_dir / "table_iii_rows.tex", paper_results)
    print(f"Validated paper results written to {summaries_dir.resolve()}")


if __name__ == "__main__":
    main()
