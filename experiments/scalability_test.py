import json
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.acf_simulator import ACFSimulator


def test_pec_scalability(output_dir="results/scalability"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    profile_path = Path("hardware_profile.json").resolve()
    sim = ACFSimulator(str(profile_path))
    pec_config = sim.hw_profile["federation_costs"]["PEC_hardware"]

    client_counts = [2, 5, 10, 20, 50, 100, 200, 500, 1000]
    model_sizes = [10, 50, 100]
    results = {
        "schema_version": 1,
        "assumptions": {
            "communication_included": False,
            "resource_scaling": "fixed",
            "parallel_lanes": int(pec_config.get("parallel_lanes", 1)),
            "throughput_bytes_per_cycle_per_lane": float(
                pec_config["throughput_bytes_per_cycle"]
            ),
            "pipeline_depth_cycles": float(pec_config["pipeline_depth"]),
            "clock_frequency_mhz": float(sim.clock_freq_mhz),
            "memory_bandwidth_gbps": float(sim.mem_bw_gbps),
            "input_bytes": "K multiplied by model update size",
        },
        "model_sizes_mb": model_sizes,
        "clients": {},
    }

    for client_count in client_counts:
        row = {}
        for model_size in model_sizes:
            suffix = f"{model_size}mb"
            sac_latency = sim.simulate_aggregation(
                client_count,
                model_size,
                "PEC",
            )
            cpu_latency = sim.simulate_aggregation(
                client_count,
                model_size,
                "Software",
            )
            row[f"sac_{suffix}"] = float(sac_latency)
            row[f"cpu_{suffix}"] = float(cpu_latency)
            row[f"speedup_{suffix}"] = float(cpu_latency / sac_latency)
        results["clients"][str(client_count)] = row

    with (output_path / "scalability_results.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(results, handle, indent=2)
    print(f"Scalability results written to {output_path.resolve()}")
