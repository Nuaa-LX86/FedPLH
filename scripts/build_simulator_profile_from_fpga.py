#!/usr/bin/env python3
"""Build the trace-model compatibility profile from audited VCU128 evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODE_MAP = {
    "FP32": "fp32",
    "TF32": "tf32",
    "BF16": "bf16",
    "FP16": "fp16",
    "FP8_E5M2": "fp8_e5m2",
    "FP8_E4M3": "fp8_e4m3",
    "INT8": "int8",
    "INT4": "int4",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def positive(value: object, label: str) -> float:
    number = float(value)
    if number <= 0:
        raise SystemExit(f"{label} must be positive, found {number}")
    return number


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--integrated-profile", type=Path,
        default=ROOT / "硬件代码" / "FedPLH_VCU128" / "evidence"
        / "fpga_vcu128_integrated_profile.json",
    )
    parser.add_argument(
        "--operation-convention", type=Path,
        default=ROOT / "硬件代码" / "FedPLH_VCU128" / "manifest"
        / "operation_convention.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "hardware_profile_vcu128.json",
    )
    parser.add_argument("--memory-bandwidth-gbps", type=float, default=32.0)
    parser.add_argument("--dram-pj-per-byte", type=float, default=80.0)
    parser.add_argument("--sram-pj-per-byte", type=float, default=5.0)
    parser.add_argument("--clipping-cycles-per-param", type=float, default=0.5)
    parser.add_argument("--noise-cycles-per-param", type=float, default=0.05)
    parser.add_argument("--software-conversion-factor", type=float, default=12.0)
    args = parser.parse_args()

    integrated = json.loads(args.integrated_profile.read_text(encoding="utf-8"))
    convention = json.loads(args.operation_convention.read_text(encoding="utf-8"))
    if integrated.get("status") != "selected_post_route_profiles":
        raise SystemExit("integrated profile is not a selected post-route profile")
    if integrated.get("validation", {}).get("rtl_evidence_audit", {}).get("status") != "passed":
        raise SystemExit("integrated profile does not carry passing RTL evidence")

    client = integrated["implementations"]["client_core"]
    server = integrated["implementations"]["server_core"]
    runtime = integrated["runtime_parameters"]
    array_count = int(runtime["hmpe"]["array_count"])
    client_clock_mhz = positive(client["selected_run"]["target_frequency_mhz"], "client clock")
    server_clock_mhz = positive(server["selected_run"]["target_frequency_mhz"], "server clock")
    sac_rate = positive(
        runtime["sac"]["accepted_input_bytes_per_cycle"],
        "SAC accepted input rate",
    )
    mode_power = runtime["hmpe"].get("mode_power", {})

    ops_per_cycle: dict[str, int] = {}
    compute_costs: dict[str, dict[str, float | str]] = {}
    for canonical, profile_key in MODE_MAP.items():
        packed_products = int(convention["modes"][canonical]["packed_products"])
        array_products = array_count * packed_products
        power_record = mode_power.get(profile_key)
        if power_record is None:
            raise SystemExit(f"missing mode-specific SAIF power for {profile_key}")
        dynamic_w = positive(
            power_record["hmpe_array_dynamic_w"],
            f"{canonical} HMPE-array dynamic power",
        )
        report_frequency_mhz = positive(
            power_record["target_frequency_mhz"],
            f"{canonical} SAIF frequency",
        )
        if abs(report_frequency_mhz - client_clock_mhz) > 1e-9:
            raise SystemExit(
                f"{canonical} SAIF frequency {report_frequency_mhz} MHz does not "
                f"match selected client clock {client_clock_mhz} MHz"
            )
        energy_pj = dynamic_w / (client_clock_mhz * 1e6 * array_products) * 1e12
        ops_per_cycle[canonical] = array_products
        compute_costs[canonical] = {
            "energy_pJ": energy_pj,
            "unit": "pJ per packed product",
            "scope": "SAIF-annotated HMPE-array dynamic power only",
        }

    if args.memory_bandwidth_gbps <= 0 or args.software_conversion_factor <= 0:
        raise SystemExit("memory bandwidth and software conversion factor must be positive")

    profile = {
        "metadata": {
            "source": "Audited VCU128 OOC post-route and SAIF profiles",
            "device_part": integrated["metadata"]["device_part"],
            "vivado_version": integrated["metadata"]["vivado_version"],
            "validation_level": "RTL tests + OOC post-route + SAIF; no board deployment",
            "integrated_profile": str(args.integrated_profile.resolve()),
            "integrated_profile_sha256": sha256(args.integrated_profile),
            "operation_convention": str(args.operation_convention.resolve()),
            "operation_convention_sha256": sha256(args.operation_convention),
        },
        "design_parameters": {
            "clock_frequency_MHz": client_clock_mhz,
            "server_clock_frequency_MHz": server_clock_mhz,
            "memory_bandwidth_GBps": args.memory_bandwidth_gbps,
            "memory_bandwidth_scope": "modeled phase-1 assumption; not board measured",
            "hmpe_array_count": array_count,
            "ops_per_cycle": ops_per_cycle,
            "operation_unit": "packed products per accepted HMPE-array input",
        },
        "compute_costs_per_op": compute_costs,
        "memory_costs": {
            "DRAM_access_per_byte_pJ": args.dram_pj_per_byte,
            "SRAM_access_per_byte_pJ": args.sram_pj_per_byte,
            "scope": "modeled local-training memory energy assumptions",
        },
        "security_costs": {
            "per_example_clipping": {
                "cycles_per_param": args.clipping_cycles_per_param,
            },
            "noise_generation": {
                "cycles_per_param": args.noise_cycles_per_param,
            },
            "scope": "modeled SoftDP operator cost; operand datapath is not implemented",
        },
        "federation_costs": {
            "PEC_hardware": {
                "name": "SAC",
                "clock_frequency_MHz": server_clock_mhz,
                "pipeline_depth": int(
                    runtime["sac"]["pipeline_fill_drain_cycles"]
                ),
                "arithmetic_pipeline_cycles": int(
                    runtime["sac"]["fp32_multiply_latency_cycles"]
                    + runtime["sac"]["fp32_add_latency_cycles"]
                ),
                "throughput_bytes_per_cycle": sac_rate,
                "completion_amortized_bytes_per_cycle": float(
                    runtime["sac"]["completion_amortized_bytes_per_cycle"]
                ),
                "parallel_lanes": 1,
                "parallel_elements": int(runtime["sac"]["parallel_elements"]),
                "packet_pipeline_initiation_interval": runtime["sac"][
                    "packet_pipeline_initiation_interval"
                ],
                "address_hazard_policy": runtime["sac"]["address_hazard_policy"],
                "throughput_scope": (
                    "back-to-back non-overlapping accepted input rate from RTL "
                    "cycle characterization; completion-amortized rate retained"
                ),
            },
            "software_baseline": {
                "software_conversion_factor": args.software_conversion_factor,
                "scope": "modeled software reference factor; not FPGA measured",
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote simulator compatibility profile to {args.output}")


if __name__ == "__main__":
    main()
