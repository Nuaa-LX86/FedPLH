from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path
from typing import Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    }
)


def load_profile(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_cycles_from_paper_results(
    path: Path,
    method: str,
) -> Tuple[float, float]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    results = payload.get("results", payload)
    metrics = results[method]["metrics"]
    return (
        float(metrics["avg_delta_c_cycles"]),
        float(metrics["avg_c_priv_cycles"]),
    )


def _analysis_input_sha256(payload: dict, required_fields: Sequence[str]) -> str:
    projected = {key: payload[key] for key in required_fields}
    canonical = json.dumps(
        projected,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
        newline="\n",
    )


def compute_credit_factor_sensitivity(
    history_paths: Sequence[Path],
    *,
    expected_seed_count: int | None = None,
    expected_round_count: int | None = None,
) -> dict:
    paths = sorted(Path(path) for path in history_paths)
    if not paths:
        raise ValueError("At least one training_history.json path is required")
    if expected_seed_count is not None and len(paths) != expected_seed_count:
        raise ValueError(
            f"Expected {expected_seed_count} seed histories, found {len(paths)}"
        )

    seed_ids = [path.parent.name for path in paths]
    if len(set(seed_ids)) != len(seed_ids):
        raise ValueError("Training-history paths contain duplicate seed identifiers")

    required = ("round", "delta_c_cycles", "c_priv_cycles")
    ratios = []
    inputs = []
    per_seed = []

    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(
                f"{path} is missing required fields: {', '.join(missing)}"
            )

        lengths = {key: len(payload[key]) for key in required}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"{path} has unaligned round-level arrays: {lengths}")
        if lengths["round"] == 0:
            raise ValueError(f"{path} contains no round-level records")
        if (
            expected_round_count is not None
            and lengths["round"] != expected_round_count
        ):
            raise ValueError(
                f"Expected {expected_round_count} rounds in {path}, "
                f"found {lengths['round']}"
            )
        if len(set(payload["round"])) != lengths["round"]:
            raise ValueError(f"{path} contains duplicate round identifiers")

        seed_ratios = []
        for round_id, delta_c, c_priv in zip(
            payload["round"],
            payload["delta_c_cycles"],
            payload["c_priv_cycles"],
        ):
            delta_c = float(delta_c)
            c_priv = float(c_priv)
            if delta_c <= 0 or c_priv <= 0:
                raise ValueError(
                    f"{path} round {round_id} has non-positive cycle values"
                )
            seed_ratios.append(c_priv / delta_c)

        ratios.extend(seed_ratios)
        seed_id = path.parent.name
        per_seed.append(
            {
                "seed_id": seed_id,
                "round_count": len(seed_ratios),
                "maximum_required_credit_factor": max(seed_ratios),
            }
        )
        inputs.append(
            {
                "seed_id": seed_id,
                "file": f"{seed_id}/training_history.json",
                "sha256": _analysis_input_sha256(payload, required),
                "round_count": len(seed_ratios),
            }
        )

    values = np.asarray(ratios, dtype=float)
    quantiles = np.quantile(values, [0.50, 0.95, 0.99], method="linear")
    maximum = float(np.max(values))
    return {
        "schema_version": 1,
        "analysis": "BEU schedulable-credit-factor sensitivity",
        "scope": (
            "Participant-mean round-level records from frozen HMPE-ACF traces; "
            "not a per-client, per-update, or critical-path-straggler analysis."
        ),
        "definition": "required_credit_factor = c_priv_cycles / delta_c_cycles",
        "input_hash_basis": (
            "Canonical JSON projection of round, delta_c_cycles, and "
            "c_priv_cycles; unrelated history fields are excluded."
        ),
        "expected_seed_count": expected_seed_count,
        "expected_round_count_per_seed": expected_round_count,
        "seed_count": len(paths),
        "record_count": int(values.size),
        "mean_required_credit_factor": float(np.mean(values)),
        "median_required_credit_factor": float(quantiles[0]),
        "p95_required_credit_factor": float(quantiles[1]),
        "p99_required_credit_factor": float(quantiles[2]),
        "minimum_required_credit_factor": float(np.min(values)),
        "maximum_required_credit_factor": maximum,
        "full_coverage_credit_factor": maximum,
        "quantile_method": "linear",
        "per_seed": per_seed,
        "inputs": inputs,
    }


def compute_boundary(
    delta_c_cycles: float,
    c_priv_cycles: float,
    frequency_mhz: float,
    max_multiplier: float,
) -> dict:
    if delta_c_cycles <= 0 or c_priv_cycles <= 0:
        raise ValueError("delta_c_cycles and c_priv_cycles must be positive")
    if frequency_mhz <= 0:
        raise ValueError("frequency_mhz must be positive")

    threshold = delta_c_cycles / c_priv_cycles
    multipliers = np.unique(
        np.concatenate(
            [
                np.linspace(0.5, max_multiplier, 400),
                np.asarray([1.0, threshold, 30.0]),
            ]
        )
    )
    multipliers.sort()
    total_privacy_cycles = multipliers * c_priv_cycles
    covered_cycles = np.minimum(delta_c_cycles, total_privacy_cycles)
    coverage_ratio = covered_cycles / total_privacy_cycles
    uncovered_cycles = np.maximum(0.0, total_privacy_cycles - delta_c_cycles)
    visible_ms = uncovered_cycles / (frequency_mhz * 1e6) * 1e3

    def visible_at(multiplier: float) -> float:
        cycles = max(0.0, multiplier * c_priv_cycles - delta_c_cycles)
        return cycles / (frequency_mhz * 1e6) * 1e3

    return {
        "delta_c_cycles": delta_c_cycles,
        "c_priv_cycles": c_priv_cycles,
        "frequency_mhz": frequency_mhz,
        "coverage_threshold_multiplier": threshold,
        "visible_cost_ms_at_30x": visible_at(30.0),
        "profile_basis": (
            "Five-seed, per-round mean over participating clients; "
            "Delta C and C_priv are client-round aggregate cycle counts."
        ),
        "critical_path_excluded": (
            "This participant-mean boundary is not derived from the "
            "critical-path straggler SoftDP latency."
        ),
        "multipliers": multipliers.tolist(),
        "coverage_ratio": coverage_ratio.tolist(),
        "uncovered_cost_ms": visible_ms.tolist(),
    }


def plot_boundary(data: dict, output_path: Path) -> None:
    multipliers = np.asarray(data["multipliers"])
    coverage_ratio = np.asarray(data["coverage_ratio"])
    visible_ms = np.asarray(data["uncovered_cost_ms"])
    threshold = float(data["coverage_threshold_multiplier"])

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.75))

    axes[0].plot(multipliers, coverage_ratio, color="#4E79A7", lw=1.8)
    axes[0].axvline(threshold, color="#E15759", ls="--", lw=1.2)
    axes[0].fill_between(
        multipliers,
        0,
        coverage_ratio,
        where=multipliers <= threshold,
        color="#59A14F",
        alpha=0.15,
        label="Full budget coverage",
    )
    axes[0].fill_between(
        multipliers,
        0,
        coverage_ratio,
        where=multipliers > threshold,
        color="#F28E2B",
        alpha=0.15,
        label="Partial budget coverage",
    )
    axes[0].set_xlabel(r"Privacy-cost multiplier $m$")
    axes[0].set_ylabel("Budget coverage ratio")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("(a) Participant-mean budget boundary", pad=7)
    axes[0].legend(frameon=False, loc="lower left")
    axes[0].scatter([1.0], [1.0], color="#222222", s=20, zorder=4)
    axes[0].annotate(
        r"Evaluated point: $m=1$",
        xy=(1.0, 1.0),
        xytext=(4.0, 0.88),
        fontsize=7,
        arrowprops={"arrowstyle": "->", "lw": 0.75},
    )
    axes[0].annotate(
        f"Full-coverage boundary\n$m={threshold:.3f}$",
        xy=(threshold, 1.0),
        xytext=(14.0, 0.73),
        fontsize=7,
        arrowprops={"arrowstyle": "->", "lw": 0.75},
    )

    axes[1].plot(multipliers, visible_ms, color="#E15759", lw=1.8)
    axes[1].axvline(threshold, color="#E15759", ls="--", lw=1.2)
    axes[1].scatter([1.0], [0.0], color="#222222", s=20, zorder=4)
    axes[1].annotate(
        r"Evaluated point: $m=1$, 0 ms",
        xy=(1.0, 0.0),
        xytext=(4.0, 58.0),
        fontsize=7,
        arrowprops={"arrowstyle": "->", "lw": 0.75},
    )
    axes[1].scatter(
        [30.0],
        [data["visible_cost_ms_at_30x"]],
        color="#222222",
        s=18,
        zorder=3,
    )
    axes[1].annotate(
        f"30x: {data['visible_cost_ms_at_30x']:.1f} ms",
        xy=(30.0, data["visible_cost_ms_at_30x"]),
        xytext=(21.0, data["visible_cost_ms_at_30x"] * 0.72),
        fontsize=7,
        arrowprops={"arrowstyle": "->", "lw": 0.8},
    )
    axes[1].set_xlabel(r"Privacy-cost multiplier $m$")
    axes[1].set_ylabel("Uncovered latency (ms)")
    axes[1].set_title("(b) Uncovered modeled cost", pad=7)

    for axis in axes:
        axis.grid(True, ls="--", alpha=0.35)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    fig.subplots_adjust(
        left=0.09,
        right=0.985,
        bottom=0.20,
        top=0.88,
        wspace=0.34,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix(".png"), dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="hardware_profile.json")
    parser.add_argument("--paper_results", default="")
    parser.add_argument("--method", default="HMPE-ACF")
    parser.add_argument("--delta_c_cycles", type=float)
    parser.add_argument("--c_priv_cycles", type=float)
    parser.add_argument(
        "--history_glob",
        default="",
        help=(
            "Optional recursive glob for frozen HMPE-ACF training histories. "
            "When provided, credit-factor sensitivity is embedded in the "
            "figure JSON."
        ),
    )
    parser.add_argument(
        "--credit_output",
        default="",
        help="Optional standalone JSON path for credit-factor sensitivity.",
    )
    parser.add_argument(
        "--expected_seed_count",
        type=int,
        default=5,
        help="Required seed-history count when --history_glob is used.",
    )
    parser.add_argument(
        "--expected_round_count",
        type=int,
        default=80,
        help="Required round count per seed when --history_glob is used.",
    )
    parser.add_argument("--max_multiplier", type=float, default=35.0)
    parser.add_argument(
        "--output",
        default="paper_figures_unet/Fig6_BEU_Boundary.pdf",
    )
    args = parser.parse_args()

    profile = load_profile(Path(args.profile))
    frequency_mhz = float(
        profile["design_parameters"]["clock_frequency_MHz"]
    )

    if args.paper_results:
        delta_c_cycles, c_priv_cycles = load_cycles_from_paper_results(
            Path(args.paper_results),
            args.method,
        )
    elif args.delta_c_cycles is not None and args.c_priv_cycles is not None:
        delta_c_cycles = float(args.delta_c_cycles)
        c_priv_cycles = float(args.c_priv_cycles)
    else:
        raise ValueError(
            "Provide --paper_results or both --delta_c_cycles and --c_priv_cycles"
        )

    data = compute_boundary(
        delta_c_cycles,
        c_priv_cycles,
        frequency_mhz,
        args.max_multiplier,
    )
    if args.history_glob:
        history_paths = [
            Path(path) for path in glob.glob(args.history_glob, recursive=True)
        ]
        sensitivity = compute_credit_factor_sensitivity(
            history_paths,
            expected_seed_count=args.expected_seed_count,
            expected_round_count=args.expected_round_count,
        )
        data["credit_factor_sensitivity"] = sensitivity
        if args.credit_output:
            credit_output = Path(args.credit_output)
            _write_json(credit_output, sensitivity)
    elif args.credit_output:
        raise ValueError("--credit_output requires --history_glob")
    output_path = Path(args.output)
    plot_boundary(data, output_path)
    json_path = output_path.with_suffix(".json")
    _write_json(json_path, data)

    print(
        f"threshold={data['coverage_threshold_multiplier']:.6f}, "
        f"30x_uncovered_ms={data['visible_cost_ms_at_30x']:.6f}, "
        f"frequency={frequency_mhz:.3f} MHz"
    )


if __name__ == "__main__":
    main()
