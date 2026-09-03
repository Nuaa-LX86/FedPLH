import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_simulator_profile_from_fpga.py"


def test_builds_separate_client_server_clock_profile(tmp_path):
    modes = {
        "FP32": 1, "TF32": 4, "BF16": 9, "FP16": 4,
        "FP8_E5M2": 36, "FP8_E4M3": 36, "INT8": 9, "INT4": 36,
    }
    convention = {
        "modes": {name: {"packed_products": products} for name, products in modes.items()}
    }
    mode_power = {
        name.lower(): {
            "hmpe_array_dynamic_w": 0.1,
            "target_frequency_mhz": 190.0,
        }
        for name in modes
    }
    integrated = {
        "status": "selected_post_route_profiles",
        "metadata": {
            "device_part": "xcvu37p-fsvh2892-2L-e",
            "vivado_version": "2026.1",
        },
        "validation": {"rtl_evidence_audit": {"status": "passed"}},
        "implementations": {
            "client_core": {"selected_run": {"target_frequency_mhz": 190}},
            "server_core": {"selected_run": {"target_frequency_mhz": 250}},
        },
        "runtime_parameters": {
            "hmpe": {"array_count": 4, "mode_power": mode_power},
            "sac": {
                "accepted_input_bytes_per_cycle": 2.5,
                "completion_amortized_bytes_per_cycle": 2.25,
                "packet_pipeline_initiation_interval": 1,
                "pipeline_fill_drain_cycles": 12,
                "address_hazard_policy": "stall overlapping row/mask pairs",
                "parallel_elements": 32,
                "fp32_multiply_latency_cycles": 4,
                "fp32_add_latency_cycles": 6,
            },
        },
    }
    integrated_path = tmp_path / "integrated.json"
    convention_path = tmp_path / "operation.json"
    output_path = tmp_path / "hardware.json"
    integrated_path.write_text(json.dumps(integrated), encoding="utf-8")
    convention_path.write_text(json.dumps(convention), encoding="utf-8")

    subprocess.run([
        sys.executable, str(SCRIPT),
        "--integrated-profile", str(integrated_path),
        "--operation-convention", str(convention_path),
        "--output", str(output_path),
    ], check=True)
    result = json.loads(output_path.read_text(encoding="utf-8"))

    assert result["design_parameters"]["clock_frequency_MHz"] == 190.0
    assert result["design_parameters"]["server_clock_frequency_MHz"] == 250.0
    assert result["design_parameters"]["ops_per_cycle"]["FP32"] == 4
    assert result["design_parameters"]["ops_per_cycle"]["FP8_E5M2"] == 144
    assert result["federation_costs"]["PEC_hardware"]["parallel_lanes"] == 1
    assert result["federation_costs"]["PEC_hardware"]["throughput_bytes_per_cycle"] == 2.5
    assert result["federation_costs"]["PEC_hardware"][
        "completion_amortized_bytes_per_cycle"
    ] == 2.25
    assert result["federation_costs"]["PEC_hardware"][
        "packet_pipeline_initiation_interval"
    ] == 1
    assert result["federation_costs"]["PEC_hardware"]["pipeline_depth"] == 12
    expected = 0.1 / (190e6 * 4) * 1e12
    assert abs(result["compute_costs_per_op"]["FP32"]["energy_pJ"] - expected) < 1e-9
