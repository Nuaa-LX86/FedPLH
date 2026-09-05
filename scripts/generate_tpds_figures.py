#!/usr/bin/env python3
"""Generate the TPDS data figures from audited matched-run artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visualization.plot_generator import PlotGenerator


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_histories(
    results_dir: Path,
    final_scenario: str,
) -> tuple[list[dict], list[Path]]:
    histories: list[dict] = []
    paths: list[Path] = []
    for seed in range(5):
        path = results_dir / final_scenario / f"seed{seed}" / "training_history.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing final FedPLH history: {path}")
        history = json.loads(path.read_text(encoding="utf-8"))
        if history.get("round") != list(range(80)):
            raise ValueError(f"history is not a complete 80-round run: {path}")
        for field in ("val_dice", "train_loss"):
            values = np.asarray(history.get(field, []), dtype=float)
            if values.shape != (80,) or not np.isfinite(values).all():
                raise ValueError(f"invalid {field} trajectory: {path}")
        histories.append(history)
        paths.append(path)
    return histories, paths


def load_public_trajectories(trajectory_dir: Path) -> tuple[list[dict], list[Path]]:
    histories: list[dict] = []
    paths: list[Path] = []
    for seed in range(5):
        path = trajectory_dir / f"seed{seed}" / "trajectory.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing released trajectory: {path}")
        history = json.loads(path.read_text(encoding="utf-8"))
        if history.get("round") != list(range(80)):
            raise ValueError(f"trajectory is not a complete 80-round run: {path}")
        for field in ("val_dice", "train_loss"):
            values = np.asarray(history.get(field, []), dtype=float)
            if values.shape != (80,) or not np.isfinite(values).all():
                raise ValueError(f"invalid {field} trajectory: {path}")
        histories.append(history)
        paths.append(path)
    return histories, paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "audited_runs" / "tpds_operand_complete_five_seed_20260902" / "unet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "TPDS_final_submission" / "source" / "figures",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "validated_aggregate_evidence" / "tpds_figure_manifest.json",
    )
    parser.add_argument("--paper-results", type=Path)
    parser.add_argument("--trajectory-dir", type=Path)
    parser.add_argument("--final-scenario", default="HMPE-ACF")
    args = parser.parse_args()

    paper_results_path = args.paper_results or (
        args.results_dir / "summaries" / "paper_results.json"
    )
    if not paper_results_path.is_file():
        raise FileNotFoundError(f"missing audited paper results: {paper_results_path}")
    paper_payload = json.loads(paper_results_path.read_text(encoding="utf-8"))
    results = paper_payload.get("results")
    if not isinstance(results, dict) or "HMPE-ACF" not in results:
        raise ValueError("paper_results.json lacks the final HMPE-ACF entry")

    if args.trajectory_dir is None:
        histories, history_paths = load_histories(args.results_dir, args.final_scenario)
    else:
        histories, history_paths = load_public_trajectories(args.trajectory_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plotter = PlotGenerator(str(args.output_dir))
    plotter.plot_beu_latency_breakdown(results, save_name="Fig4_BEU_Breakdown")
    plotter.plot_training_curves(histories, save_name="Fig5_Convergence")

    outputs = [
        args.output_dir / f"{stem}.{extension}"
        for stem in ("Fig4_BEU_Breakdown", "Fig5_Convergence")
        for extension in ("pdf", "png")
    ]
    if any(not path.is_file() or path.stat().st_size == 0 for path in outputs):
        raise RuntimeError("one or more TPDS data-figure outputs are missing")

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "scope": "Figs. 4 and 5 from final operand-complete five-seed runs",
        "inputs": {
            str(paper_results_path.resolve()): sha256(paper_results_path),
            **{str(path.resolve()): sha256(path) for path in history_paths},
        },
        "tools": {
            "scripts/generate_tpds_figures.py": sha256(Path(__file__)),
            "visualization/plot_generator.py": sha256(
                ROOT / "visualization" / "plot_generator.py"
            ),
        },
        "outputs": {str(path.resolve()): sha256(path) for path in outputs},
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Generated TPDS Figs. 4--5 and manifest: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
