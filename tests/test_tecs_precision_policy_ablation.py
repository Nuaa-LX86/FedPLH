from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analyze_tecs_precision_policy_ablation.py"
)
SPEC = importlib.util.spec_from_file_location("analyze_tecs_ablation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_mean_sd_ci_uses_student_t() -> None:
    summary = MODULE.mean_sd_ci([1.0, 2.0, 3.0, 4.0, 5.0])
    expected = 2.7764451051977987 * math.sqrt(2.5) / math.sqrt(5)
    assert summary["mean"] == 3.0
    assert summary["sample_sd"] == pytest.approx(math.sqrt(2.5))
    assert summary["ci95_half_width"] == pytest.approx(expected)
    assert summary["ci_method"] == "two-sided Student-t"


def test_paired_summary_counts_directions() -> None:
    summary = MODULE.paired_summary(
        [2.0, 1.0, 4.0, 5.0, 5.0],
        [1.0, 2.0, 4.0, 3.0, 6.0],
    )
    assert summary["positive_count"] == 2
    assert summary["negative_count"] == 2
    assert summary["zero_count"] == 1


def test_dominance_requires_no_worse_and_one_strict_gain() -> None:
    reference = {
        "test_dice": 86.0,
        "actual_latency_ms": 100.0,
        "local_training_energy_mJ": 10.0,
    }
    better = {
        "test_dice": 86.0,
        "actual_latency_ms": 99.0,
        "local_training_energy_mJ": 10.0,
    }
    tradeoff = {
        "test_dice": 87.0,
        "actual_latency_ms": 101.0,
        "local_training_energy_mJ": 10.0,
    }
    assert MODULE.dominates(better, reference)
    assert not MODULE.dominates(reference, reference)
    assert not MODULE.dominates(tradeoff, reference)
