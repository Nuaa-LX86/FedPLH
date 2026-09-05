#!/usr/bin/env python3
"""Export the sanitized evidence package used by the FedMPE TECS manuscript."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
HARDWARE_EVIDENCE = ROOT / "硬件代码" / "FedPLH_VCU128" / "evidence"
BASE_RUN = ROOT / "audited_runs" / "tpds_operand_complete_five_seed_20260902"
SOTA_RUN = ROOT / "audited_runs" / "tpds_sota_adapters_five_seed_20260902"
ABLATION_RUN = (
    ROOT / "audited_runs" / "tecs_precision_policy_ablation_operand_complete_20260904"
)
PROGRESS_RUN = ABLATION_RUN / "unet" / "ACF_progress_only"
PARTITION_SOURCE = (
    ROOT
    / "experiment_protocols"
    / "tetc_semantic_20260615"
    / "partitions"
    / "main_alpha0p5_p0"
)
TEXT_HASH_SUFFIXES = {
    ".cff",
    ".csv",
    ".json",
    ".md",
    ".ps1",
    ".py",
    ".tex",
    ".txt",
    ".yaml",
    ".yml",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_sha256(path: Path) -> str:
    if path.suffix.lower() in TEXT_HASH_SUFFIXES:
        normalized = (
            path.read_text(encoding="utf-8")
            .replace("\r\n", "\n")
            .replace("\r", "\n")
        )
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return sha256(path)


def write_utf8_lf(path: Path, text: str) -> None:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(normalized)


def sanitize_string(value: str) -> str:
    separator = r"(?:\\{1,2}|/)"
    root_parts = [re.escape(part) for part in ROOT.resolve().parts]
    root_pattern = re.compile(
        r"(?i)" + separator.join(root_parts) + separator + r"[^'\",}\]\r\n]+"
    )

    def replace_workspace_path(match: re.Match[str]) -> str:
        normalized = re.sub(r"[\\/]+", "/", match.group(0))
        root = re.sub(r"[\\/]+", "/", str(ROOT.resolve()))
        relative = normalized[len(root) + 1 :]
        if relative.startswith("硬件代码/FedPLH_VCU128/"):
            return "not_released/raw_hardware_evidence/" + Path(relative).name
        if relative.startswith("build/FedPLH_VCU128/"):
            return "not_released/vivado_build/" + Path(relative).name
        if relative.startswith("audited_runs/"):
            return "not_released/" + relative
        return relative

    sanitized = root_pattern.sub(replace_workspace_path, value)
    external_pattern = re.compile(
        r"(?i)(?<![A-Za-z0-9+.-])[A-Z]:" + separator + r"[^'\",}\]\r\n]+"
    )
    return external_pattern.sub(
        lambda match: "not_released/external_path/"
        + Path(re.sub(r"[\\/]+", "/", match.group(0))).name,
        sanitized,
    )


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {sanitize_string(str(key)): sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, str):
        return sanitize_string(value)
    return value


def export_json(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise SystemExit(f"missing public-evidence source: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    write_utf8_lf(
        destination,
        json.dumps(sanitize(payload), indent=2, ensure_ascii=False) + "\n",
    )


def export_text(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise SystemExit(f"missing public-evidence source: {source}")
    write_utf8_lf(destination, sanitize_string(source.read_text(encoding="utf-8")))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)

    evidence = ROOT / "validated_aggregate_evidence"
    json_exports = {
        HARDWARE_EVIDENCE / "fpga_vcu128_integrated_profile.json": output / "sanitized_profile_values" / "fpga_vcu128_integrated_profile.json",
        ROOT / "hardware_profile_vcu128.json": output / "hardware_profile.json",
        HARDWARE_EVIDENCE / "selected_operating_points.json": output / "sanitized_profile_values" / "selected_operating_points.json",
        HARDWARE_EVIDENCE / "rtl_test_manifest.json": output / "sanitized_profile_values" / "rtl_test_manifest.json",
        HARDWARE_EVIDENCE / "saif_power_manifest.json": output / "sanitized_profile_values" / "saif_power_manifest.json",
        evidence / "hmpe_training_contract_audit.json": output / "validated_aggregate_evidence" / "hmpe_training_contract_audit.json",
        evidence / "sota_adapter_five_seed_audit.json": output / "validated_aggregate_evidence" / "sota_adapter_five_seed_audit.json",
        evidence / "sota_adapter_provenance.json": output / "validated_aggregate_evidence" / "sota_adapter_provenance.json",
        evidence / "beu_credit_factor_sensitivity.json": output / "validated_aggregate_evidence" / "beu_credit_factor_sensitivity.json",
        evidence / "tecs_primary_result_manifest.json": output / "validated_aggregate_evidence" / "tecs_primary_result_manifest.json",
        evidence / "tecs_precision_policy_ablation.json": output / "validated_aggregate_evidence" / "tecs_precision_policy_ablation.json",
        evidence / "tecs_precision_policy_ablation_manifest.json": output / "validated_aggregate_evidence" / "tecs_precision_policy_ablation_manifest.json",
        evidence / "tecs_result_macro_manifest.json": output / "validated_aggregate_evidence" / "tecs_result_macro_manifest.json",
        evidence / "tecs_figure_manifest.json": output / "validated_aggregate_evidence" / "tecs_figure_manifest.json",
        evidence / "tecs_bibliography_audit.json": output / "validated_aggregate_evidence" / "tecs_bibliography_audit.json",
        BASE_RUN / "run_manifest.json": output / "frozen_experiment_protocols" / "operand_complete_baselines_20260902" / "run_manifest.json",
        SOTA_RUN / "run_manifest.json": output / "frozen_experiment_protocols" / "sota_adapters_20260902" / "run_manifest.json",
        ABLATION_RUN / "run_manifest.json": output / "frozen_experiment_protocols" / "precision_policy_ablation_20260904" / "run_manifest.json",
        ABLATION_RUN / "partition_evidence.json": output / "frozen_experiment_protocols" / "precision_policy_ablation_20260904" / "partition_evidence.json",
        PARTITION_SOURCE / "partition_evidence.json": output / "frozen_experiment_protocols" / "shared_brats_partition_alpha0p5" / "partition_evidence.json",
    }
    for source, destination in json_exports.items():
        export_json(source, destination)

    text_exports = {
        HARDWARE_EVIDENCE / "candidate_sweep_results.csv": output / "sanitized_profile_values" / "candidate_sweep_results.csv",
        ROOT / "sanitized_profile_values" / "hardware_comparison_provenance.csv": output / "sanitized_profile_values" / "hardware_comparison_provenance.csv",
        evidence / "tecs_precision_policy_ablation.csv": output / "validated_aggregate_evidence" / "tecs_precision_policy_ablation.csv",
        PARTITION_SOURCE / "client_distribution.csv": output / "frozen_experiment_protocols" / "shared_brats_partition_alpha0p5" / "client_distribution.csv",
        ROOT / "postprocessed_summaries" / "tecs_submission_results.json": output / "postprocessed_summaries" / "tecs_submission_results.json",
        ROOT / "TECS_submission" / "source" / "generated_result_values.tex": output / "postprocessed_summaries" / "generated_result_values.tex",
        ROOT / "TECS_submission" / "source" / "generated_fpga_values.tex": output / "postprocessed_summaries" / "generated_fpga_values.tex",
        ROOT / "TECS_submission" / "source" / "generated_ablation_values.tex": output / "postprocessed_summaries" / "generated_ablation_values.tex",
    }
    for source, destination in text_exports.items():
        export_text(source, destination)

    beu_inputs = output / "validated_aggregate_evidence" / "beu_credit_factor_inputs"
    trajectories = output / "postprocessed_summaries" / "convergence_inputs"
    for seed in range(5):
        source = PROGRESS_RUN / f"seed{seed}" / "training_history.json"
        history = json.loads(source.read_text(encoding="utf-8"))
        beu_input = {
            "seed": seed,
            "round": history.get("round"),
            "profiled_timing_slack_cycles": history.get("delta_c_cycles"),
            "operator_cost_cycles": history.get("c_priv_cycles"),
            "scope": "participant-mean round records from the Progress-only configuration",
        }
        write_utf8_lf(
            beu_inputs / f"seed{seed}" / "training_history.json",
            json.dumps(beu_input, indent=2) + "\n",
        )

        trajectory = {
            "seed": seed,
            "round": history.get("round"),
            "val_dice": history.get("val_dice"),
            "train_loss": history.get("train_loss"),
            "scope": "aggregate learning trajectory; no images, labels, gradients, or checkpoints",
        }
        write_utf8_lf(
            trajectories / f"seed{seed}" / "trajectory.json",
            json.dumps(trajectory, indent=2) + "\n",
        )

    ignored_parts = {".git", ".pytest_cache", "__pycache__"}
    files = sorted(
        path
        for path in output.rglob("*")
        if path.is_file()
        and path.name
        not in {"PUBLIC_ARTIFACT_SHA256.json", ".gitattributes", ".gitignore"}
        and not ignored_parts.intersection(path.parts)
        and path.suffix.lower() not in {".pyc", ".pyo"}
    )
    manifest = {
        "schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "hash_policy": "utf8_text_lf_else_raw_bytes",
        "scope": (
            "sanitized FedMPE TECS evidence; no raw images, checkpoints, RTL, "
            "or complete Vivado reports"
        ),
        "files": {
            path.relative_to(output).as_posix(): artifact_sha256(path)
            for path in files
        },
    }
    write_utf8_lf(
        output / "PUBLIC_ARTIFACT_SHA256.json",
        json.dumps(manifest, indent=2) + "\n",
    )
    print(f"Exported {len(files)} sanitized artifacts to {output}")


if __name__ == "__main__":
    main()
