from __future__ import annotations

import numpy as np
import pytest

from training.acf_scheduler import ACFScheduler


CLIENT_STATS = [
    {"entropy": 0.2},
    {"entropy": 0.5},
    {"entropy": 1.0},
]


def make_scheduler(mode: str, *, seed: int = 7, deterministic: bool = False):
    return ACFScheduler(
        total_rounds=10,
        client_stats=CLIENT_STATS,
        lamda=0.5,
        budget_safety_threshold=0.0,
        mode=mode,
        low_precision="FP8_E5M2",
        high_precision="BF16",
        deterministic=deterministic,
        seed=seed,
    )


def test_static_modes_select_expected_precision() -> None:
    low = make_scheduler("static_low", deterministic=True)
    high = make_scheduler("static_high", deterministic=True)

    for client_id in range(len(CLIENT_STATS)):
        for round_id in range(10):
            assert low.get_execution_plan(client_id, round_id)["compute"] == "FP8_E5M2"
            assert high.get_execution_plan(client_id, round_id)["compute"] == "BF16"


@pytest.mark.parametrize("mode", ["entropy_time", "entropy_only", "time_decay"])
def test_dynamic_modes_are_reproducible_for_a_fixed_seed(mode: str) -> None:
    first = make_scheduler(mode, seed=19)
    second = make_scheduler(mode, seed=19)

    first_trace = [
        first.get_execution_plan(client_id % 3, round_id % 10)["compute"]
        for round_id, client_id in enumerate(range(60))
    ]
    second_trace = [
        second.get_execution_plan(client_id % 3, round_id % 10)["compute"]
        for round_id, client_id in enumerate(range(60))
    ]
    assert first_trace == second_trace


def test_entropy_and_progress_probabilities_match_their_definitions() -> None:
    entropy = make_scheduler("entropy_only")
    progress = make_scheduler("time_decay")
    combined = make_scheduler("entropy_time")

    assert entropy._p_high(0, 0) == pytest.approx(0.2)
    assert progress._p_high(0, 4) == pytest.approx(0.5)
    assert combined._p_high(0, 4) == pytest.approx(0.35)


def test_budget_recovery_branch_is_inactive_at_zero_threshold() -> None:
    scheduler = make_scheduler("static_high", deterministic=True)
    plan = scheduler.get_execution_plan(0, 0, current_budget=0.0)
    assert plan["compute"] == "BF16"
    assert plan["reason"].startswith("High_Precision")


def test_unknown_mode_is_rejected() -> None:
    scheduler = make_scheduler("not_a_mode")
    with pytest.raises(ValueError, match="Unsupported precision scheduling mode"):
        scheduler.get_execution_plan(0, 0)


def test_dynamic_rng_isolated_from_numpy_global_state() -> None:
    np.random.seed(123)
    before = np.random.random()
    scheduler = make_scheduler("entropy_time", seed=23)
    for round_id in range(10):
        scheduler.get_execution_plan(round_id % 3, round_id)
    after = np.random.random()

    np.random.seed(123)
    assert before == pytest.approx(np.random.random())
    assert after == pytest.approx(np.random.random())
