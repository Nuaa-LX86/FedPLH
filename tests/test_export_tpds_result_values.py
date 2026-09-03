from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_tpds_result_values.py"
SPEC = importlib.util.spec_from_file_location("export_tpds_result_values", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_paired_stats_and_latex_formatting() -> None:
    mean, sd, ci = MODULE.paired_stats(
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [0.5, 1.5, 2.5, 3.5, 4.5],
    )
    assert mean == 0.5
    assert sd == 0.0
    assert ci == 0.0
    assert MODULE.text_pm(87.123, 0.456) == "87.12\\(\\pm\\)0.46"
    assert MODULE.math_pm(0.3942, 0.0014) == "0.394\\pm0.001"


def test_normalized_requires_five_pairs() -> None:
    entry = {
        "normalized": {
            "avg_latency_ms": {"n": 5, "mean": 0.4, "std": 0.01}
        }
    }
    assert MODULE.normalized(entry, "avg_latency_ms") == (0.4, 0.01)


def test_normalized_accepts_explicit_pair_values() -> None:
    entry = {
        "normalized": {
            "avg_latency_ms": {
                "mean": 0.4,
                "std": 0.01,
                "values": [0.39, 0.40, 0.41, 0.40, 0.40],
            }
        }
    }
    assert MODULE.normalized(entry, "avg_latency_ms") == (0.4, 0.01)
