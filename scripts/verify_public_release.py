#!/usr/bin/env python3
"""Verify hashes and evidence gates in the public FedMPE release."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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
    failures: list[str] = []
    for relative, expected in manifest.get("files", {}).items():
        path = root / relative
        if not path.is_file():
            failures.append(f"missing: {relative}")
        elif sha256(path) != expected:
            failures.append(f"hash mismatch: {relative}")

    paper = load_json(root / "postprocessed_summaries" / "paper_results.json")
    if paper.get("validation", {}).get("status") != "passed":
        failures.append("paper result validation is not passing")
    if paper.get("seeds") != list(range(5)):
        failures.append("paper result seed set is not 0..4")
    for method, record in paper.get("results", {}).items():
        if sorted(int(seed) for seed in record.get("seeds", {})) != list(range(5)):
            failures.append(f"{method} does not contain five aligned seeds")

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
        "tpds_submission_qa.json",
        "tecs_submission_qa.json",
    ):
        audit = load_json(root / "validated_aggregate_evidence" / audit_name)
        if audit.get("status") != "passed":
            failures.append(f"audit is not passing: {audit_name}")

    ablation = load_json(
        root / "validated_aggregate_evidence" / "precision_policy_ablation.json"
    )
    records = ablation.get("records", [])
    expected_policies = {
        "ACF_static_BF16",
        "ACF_static_FP8",
        "ACF_progress_only",
        "ACF_entropy_only",
        "ACF_full",
    }
    if ablation.get("status") != "passed" or ablation.get("rounds") != 80:
        failures.append("precision-policy ablation audit is not passing")
    if ablation.get("seeds") != list(range(5)) or len(records) != 25:
        failures.append("precision-policy ablation is not 5 policies x 5 seeds")
    if {row.get("scenario") for row in records} != expected_policies:
        failures.append("precision-policy set does not match the frozen protocol")
    for row in records:
        class_mean = (float(row["WT"]) + float(row["TC"]) + float(row["ET"])) / 3.0
        if not math.isclose(class_mean, float(row["test_dice"]), abs_tol=1e-9):
            failures.append(
                f"classwise Dice does not reproduce Mean for {row['scenario']} seed{row['seed']}"
            )

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
        slack = payload.get("delta_c_cycles", [])
        costs = payload.get("c_priv_cycles", [])
        if not (len(rounds) == len(slack) == len(costs) == 80):
            failures.append(f"BEU input seed{seed} is not an 80-round record")
            continue
        for delta_c, c_priv in zip(slack, costs):
            if float(delta_c) <= 0 or float(c_priv) < 0:
                failures.append(f"BEU input seed{seed} contains invalid cycle counts")
                break
            ratios.append(float(c_priv) / float(delta_c))

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
