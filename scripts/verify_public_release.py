#!/usr/bin/env python3
"""Verify hashes and evidence gates in a sanitized FedPLH release."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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
    ):
        audit = load_json(root / "validated_aggregate_evidence" / audit_name)
        if audit.get("status") != "passed":
            failures.append(f"audit is not passing: {audit_name}")

    credit = load_json(
        root / "validated_aggregate_evidence" / "beu_credit_factor_sensitivity.json"
    )
    if credit.get("seed_count") != 5 or credit.get("record_count") != 400:
        failures.append("BEU credit evidence is not 5 seeds x 80 rounds")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print(f"PASS: {len(manifest['files'])} artifact hashes and all release gates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
