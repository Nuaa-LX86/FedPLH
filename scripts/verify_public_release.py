#!/usr/bin/env python3
"""Verify hashes and evidence gates in the sanitized FedMPE release."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


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


def artifact_sha256(path: Path, policy: str) -> str:
    if policy == "utf8_text_lf_else_raw_bytes" and path.suffix.lower() in TEXT_HASH_SUFFIXES:
        normalized = (
            path.read_text(encoding="utf-8")
            .replace("\r\n", "\n")
            .replace("\r", "\n")
        )
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return sha256(path)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def linear_quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = load_json(root / "PUBLIC_ARTIFACT_SHA256.json")
    hash_policy = manifest.get("hash_policy", "raw_bytes")
    failures: list[str] = []
    for relative, expected in manifest.get("files", {}).items():
        path = root / relative
        if not path.is_file():
            failures.append(f"missing: {relative}")
        elif artifact_sha256(path, hash_policy) != expected:
            failures.append(f"hash mismatch: {relative}")

    paper_path = root / "postprocessed_summaries" / "tecs_submission_results.json"
    paper = load_json(paper_path)
    if paper.get("validation", {}).get("status") != "passed":
        failures.append("paper result validation is not passing")
    if paper.get("seeds") != list(range(5)):
        failures.append("paper result seed set is not 0..4")

    selection = paper.get("selection", {})
    if selection.get("source_scenario") != "ACF_progress_only":
        failures.append("representative result is not mapped from Progress only")
    if selection.get("display_policy") != "Progress only":
        failures.append("representative policy display name is not Progress only")
    if selection.get("training_reused_without_retraining") is not True:
        failures.append("no-retraining provenance flag is not true")

    fedmpe = paper.get("results", {}).get("FedMPE", {})
    if fedmpe.get("source_scenario") != "ACF_progress_only":
        failures.append("FedMPE result source is not ACF_progress_only")
    seed_records = fedmpe.get("seeds", {})
    if sorted(int(seed) for seed in seed_records) != list(range(5)):
        failures.append("FedMPE result does not contain five aligned seeds")
    tolerance = 1e-6
    for seed, record in seed_records.items():
        total = float(record.get("avg_operator_total_ms", math.nan))
        admitted = float(record.get("avg_operator_admitted_ms", math.nan))
        visible = float(record.get("avg_operator_visible_bound_ms", math.nan))
        bound = float(record.get("avg_latency_ms", math.nan))
        serial = float(record.get("avg_actual_serial_latency_ms", math.nan))
        if not math.isclose(total, admitted + visible, rel_tol=0.0, abs_tol=tolerance):
            failures.append(f"operator accounting does not close for seed{seed}")
        if not math.isclose(serial, bound + admitted, rel_tol=0.0, abs_tol=tolerance):
            failures.append(f"serial latency does not close for seed{seed}")

    profile = load_json(
        root / "sanitized_profile_values" / "fpga_vcu128_integrated_profile.json"
    )
    if profile.get("status") != "selected_post_route_profiles":
        failures.append("VCU128 profile is not a selected post-route profile")
    implementations = profile.get("implementations", {})
    if not all(name in implementations for name in ("client_core", "server_core")):
        failures.append("client/server implementation split is incomplete")

    for audit_name in (
        "hmpe_training_contract_audit.json",
        "sota_adapter_five_seed_audit.json",
        "tecs_precision_policy_ablation.json",
        "tecs_primary_result_manifest.json",
    ):
        audit = load_json(root / "validated_aggregate_evidence" / audit_name)
        if audit.get("status") != "passed":
            failures.append(f"audit is not passing: {audit_name}")

    ablation = load_json(
        root / "validated_aggregate_evidence" / "tecs_precision_policy_ablation.json"
    )
    if ablation.get("claim_gates", {}).get("selected_policy") != "ACF_progress_only":
        failures.append("ablation does not select Progress only")
    progress = ablation.get("scenario_summary", {}).get("ACF_progress_only", {})
    progress_values = progress.get("metrics", {}).get("test_dice", {}).get("values", [])
    if len(progress_values) != 5:
        failures.append("Progress-only ablation result does not contain five seeds")

    primary = load_json(
        root / "validated_aggregate_evidence" / "tecs_primary_result_manifest.json"
    )
    if primary.get("source_scenario") != "ACF_progress_only":
        failures.append("primary-result manifest has the wrong source scenario")
    if primary.get("no_retraining") is not True:
        failures.append("primary-result manifest does not assert no retraining")
    if primary.get("frozen_histories_unchanged") is not True:
        failures.append("primary-result manifest does not preserve frozen histories")
    if primary.get("protocol", {}).get("seeds") != list(range(5)):
        failures.append("primary-result manifest seed set is not 0..4")
    if primary.get("protocol", {}).get("rounds") != 80:
        failures.append("primary-result manifest does not specify 80 rounds")
    expected_canonical_hash = primary.get("output", {}).get("canonical_json_sha256")
    if expected_canonical_hash != canonical_json_sha256(paper):
        failures.append("primary-result canonical hash does not match the public result")

    credit = load_json(
        root / "validated_aggregate_evidence" / "beu_credit_factor_sensitivity.json"
    )
    if credit.get("seed_count") != 5 or credit.get("record_count") != 400:
        failures.append("BEU credit evidence is not 5 seeds x 80 rounds")
    ratios: list[float] = []
    for seed in range(5):
        payload = load_json(
            root
            / "validated_aggregate_evidence"
            / "beu_credit_factor_inputs"
            / f"seed{seed}"
            / "training_history.json"
        )
        rounds = payload.get("round", [])
        slack = payload.get(
            "profiled_timing_slack_cycles", payload.get("delta_c_cycles", [])
        )
        costs = payload.get(
            "operator_cost_cycles", payload.get("c_priv_cycles", [])
        )
        if not (len(rounds) == len(slack) == len(costs) == 80):
            failures.append(f"BEU input seed{seed} is not an 80-round record")
            continue
        for delta_c, operator_cost in zip(slack, costs):
            if float(delta_c) <= 0 or float(operator_cost) < 0:
                failures.append(f"BEU input seed{seed} contains invalid cycle counts")
                break
            ratios.append(float(operator_cost) / float(delta_c))

    if len(ratios) == 400:
        recomputed = {
            "mean_required_credit_factor": statistics.fmean(ratios),
            "median_required_credit_factor": statistics.median(ratios),
            "p95_required_credit_factor": linear_quantile(ratios, 0.95),
            "p99_required_credit_factor": linear_quantile(ratios, 0.99),
            "minimum_required_credit_factor": min(ratios),
            "maximum_required_credit_factor": max(ratios),
            "full_coverage_credit_factor": max(ratios),
        }
        for field, actual in recomputed.items():
            reported = float(credit.get(field, math.nan))
            if not math.isclose(actual, reported, rel_tol=1e-12, abs_tol=1e-15):
                failures.append(
                    f"BEU credit summary mismatch for {field}: "
                    f"reported={reported}, recomputed={actual}"
                )

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print(f"PASS: {len(manifest['files'])} artifact hashes and all release gates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
