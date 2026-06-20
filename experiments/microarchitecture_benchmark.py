import json
import sys
import os
from pathlib import Path

# Path Fix
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulator.acf_simulator import ACFSimulator


def benchmark_microarchitecture(output_dir='results/microarch'):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print("⚙️  MICROARCHITECTURE BENCHMARK (Profile Fixed)")
    print("=" * 80)

    # [FIX] 强制读取根目录下的配置文件
    profile_path = os.path.abspath('hardware_profile.json')

    if not os.path.exists(profile_path):
        print(f"⚠️ Critical Warning: Profile not found at {profile_path}!")
        sim = ACFSimulator()  # Fallback
    else:
        print(f"✅ Loaded Profile: {profile_path}")
        sim = ACFSimulator(profile_path)

    # 测试负载 (增加 INT8 以支持 BitFusion 对比)
    precisions = ['FP32', 'BF16', 'FP16', 'FP8_E5M2', 'FP8_E4M3', 'INT8']
    workloads = [
        {'name': 'Conv3x3_64ch', 'macs': 64 * 64 * 3 * 3 * 112 * 112, 'params': 64 * 3 * 3 * 3,
         'in_bytes': 3 * 224 * 224 * 4, 'out_bytes': 64 * 112 * 112 * 4},
        {'name': 'Conv1x1_256ch', 'macs': 256 * 256 * 1 * 1 * 56 * 56, 'params': 256 * 256 * 1 * 1,
         'in_bytes': 256 * 56 * 56 * 4, 'out_bytes': 256 * 56 * 56 * 4},
        {'name': 'FC_1024', 'macs': 1024 * 1000, 'params': 1024 * 1000, 'in_bytes': 1024 * 4, 'out_bytes': 1000 * 4}
    ]

    results = {}
    for wl in workloads:
        wl_name = wl['name']
        results[wl_name] = {}
        for p in precisions:
            res = sim.simulate_layer_training(**wl, compute_mode=p)
            throughput_gops = (wl['macs'] / res['cycles']) * (sim.clock_freq_mhz / 1000.0)  # Gops/s

            energy_joule = res['energy_pJ'] * 1e-12
            time_sec = res['cycles'] / (sim.clock_freq_mhz * 1e6)
            power_w = energy_joule / time_sec if time_sec > 0 else 1.0

            efficiency_gops_w = throughput_gops / power_w

            results[wl_name][p] = {
                'latency_us': sim._cycles_to_ms(res['cycles']) * 1000,
                'throughput_gops': throughput_gops,
                'efficiency_gops_w': efficiency_gops_w
            }

    with open(f'{output_dir}/microarch_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("✅ Benchmark Complete")