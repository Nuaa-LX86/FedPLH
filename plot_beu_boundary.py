from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


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

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))

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
    axes[0].set_title("(a) Participant-mean budget boundary", fontsize=10, pad=8)
    axes[0].legend(fontsize=7)

    axes[1].plot(multipliers, visible_ms, color="#E15759", lw=1.8)
    axes[1].axvline(threshold, color="#E15759", ls="--", lw=1.2)
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
    axes[1].set_title("(b) Uncovered modeled cost", fontsize=10, pad=8)

    for axis in axes:
        axis.grid(True, ls="--", alpha=0.35)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    fig.subplots_adjust(
        left=0.09,
        right=0.985,
        bottom=0.27,
        top=0.84,
        wspace=0.34,
    )
    fig.text(
        0.5,
        0.035,
        "Five-seed, per-round participating-client mean; "
        r"$C_{\mathrm{priv}}$ is a client-round aggregate.",
        ha="center",
        fontsize=7,
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
    parser.add_argument("--max_multiplier", type=float, default=35.0)
    parser.add_argument(
        "--output",
        default="paper_figures_unet/Fig_BEU_Boundary.pdf",
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
    output_path = Path(args.output)
    plot_boundary(data, output_path)
    json_path = output_path.with_suffix(".json")
    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    print(
        f"threshold={data['coverage_threshold_multiplier']:.6f}, "
        f"30x_uncovered_ms={data['visible_cost_ms_at_30x']:.6f}, "
        f"frequency={frequency_mhz:.3f} MHz"
    )


if __name__ == "__main__":
    main()
