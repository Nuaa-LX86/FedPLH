from __future__ import annotations

import json

from simulator.acf_simulator import ACFSimulator


def test_aggregation_uses_server_clock_domain(tmp_path) -> None:
    profile = {
        "metadata": {"source": "unit-test profile"},
        "design_parameters": {
            "clock_frequency_MHz": 100.0,
            "server_clock_frequency_MHz": 250.0,
            "memory_bandwidth_GBps": 32.0,
            "ops_per_cycle": {"FP32": 1},
        },
        "compute_costs_per_op": {"FP32": {"energy_pJ": 1.0}},
        "memory_costs": {"DRAM_access_per_byte_pJ": 1.0},
        "security_costs": {
            "per_example_clipping": {"cycles_per_param": 0.5},
            "noise_generation": {"cycles_per_param": 0.05},
        },
        "federation_costs": {
            "PEC_hardware": {
                "pipeline_depth": 12,
                "throughput_bytes_per_cycle": 4.0,
                "parallel_lanes": 1,
            },
            "software_baseline": {"software_conversion_factor": 12.0},
        },
    }
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile), encoding="utf-8")
    simulator = ACFSimulator(str(path))

    model_mib = 1.0
    actual_ms = simulator.simulate_aggregation(1, model_mib, "PEC")
    input_bytes = 1024.0 * 1024.0
    expected_ms = (
        input_bytes / (4.0 * 250.0e6) + 12.0 / 250.0e6
    ) * 1e3
    assert abs(actual_ms - expected_ms) < 1e-12
