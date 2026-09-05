"""Check the public five-policy FedMPE ablation artifact."""

from __future__ import annotations

import json
import math
from pathlib import Path


ARTIFACT = (
    Path(__file__).resolve().parents[1]
    / "validated_aggregate_evidence"
    / "precision_policy_ablation.json"
)

POLICIES = {
    "ACF_static_BF16",
    "ACF_static_FP8",
    "ACF_progress_only",
    "ACF_entropy_only",
    "ACF_full",
}


def main() -> int:
    data = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    records = data.get("records", [])
    assert data.get("status") == "passed"
    assert data.get("rounds") == 80
    assert data.get("seeds") == [0, 1, 2, 3, 4]
    assert len(records) == 25
    assert {row["scenario"] for row in records} == POLICIES

    for policy in POLICIES:
        rows = [row for row in records if row["scenario"] == policy]
        assert {row["seed"] for row in rows} == {0, 1, 2, 3, 4}
        for row in rows:
            class_mean = (row["WT"] + row["TC"] + row["ET"]) / 3.0
            assert math.isclose(class_mean, row["test_dice"], abs_tol=1e-9)

    bf16 = [r["bf16_assignment_rate"] for r in records if r["scenario"] == "ACF_static_BF16"]
    fp8 = [r["bf16_assignment_rate"] for r in records if r["scenario"] == "ACF_static_FP8"]
    assert bf16 == [1.0] * 5
    assert fp8 == [0.0] * 5
    print("Validated 25 records: 5 policies x 5 seeds, 80 rounds each")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
