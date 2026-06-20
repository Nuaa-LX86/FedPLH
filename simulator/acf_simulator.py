# simulator/acf_simulator.py (Enhanced Version)

import json
import math
from typing import Dict, Any, List, Tuple


class ACFSimulator:
    """
    HMPE-ACF 硬件模拟器（生产级版本）

    新增功能:
    1. 精确的Roofline模型
    2. BEU预算追踪
    3. PEC可扩展性建模
    """

    def __init__(self, hardware_profile_path: str = None):
        if hardware_profile_path:
            try:
                with open(hardware_profile_path, 'r') as f:
                    self.hw_profile = json.load(f)
                print(f"Loaded hardware profile: {self.hw_profile['metadata']['source']}")
            except FileNotFoundError:
                print("Profile not found, using default.")
                self.hw_profile = self._generate_default_profile()
        else:
            self.hw_profile = self._generate_default_profile()

        # 提取关键参数
        self.clock_freq_mhz = self.hw_profile['design_parameters']['clock_frequency_MHz']
        self.clock_period_ns = 1000.0 / self.clock_freq_mhz

        self.mem_bw_gbps = self.hw_profile['design_parameters']['memory_bandwidth_GBps']
        self.bytes_per_cycle = self.mem_bw_gbps / (self.clock_freq_mhz / 1000.0)

        # 精度字节映射
        self.precision_bytes = {
            'FP32': 4, 'TF32': 4,
            'BF16': 2, 'FP16': 2,
            'FP8_E5M2': 1, 'FP8_E4M3': 1,
            'INT8': 1, 'INT4': 0.5
        }

    def _generate_default_profile(self):
        """生成默认配置（应与JSON一致）"""
        return {
            "metadata": {"source": "Default Config"},
            "design_parameters": {
                "clock_frequency_MHz": 1000,
                "memory_bandwidth_GBps": 32.0,
                "ops_per_cycle": {
                    "FP32": 1024, "BF16": 2048, "FP16": 2048,
                    "FP8_E5M2": 4096, "FP8_E4M3": 4096
                }
            },
            "compute_costs_per_op": {
                "FP32": {"energy_pJ": 4.12},
                "BF16": {"energy_pJ": 1.03},
                "FP16": {"energy_pJ": 1.02},
                "FP8_E5M2": {"energy_pJ": 0.33},
                "FP8_E4M3": {"energy_pJ": 0.33}
            },
            "memory_costs": {
                "DRAM_access_per_byte_pJ": 80.0
            },
            "security_costs": {
                "per_example_clipping": {"cycles_per_param": 0.5},
                "noise_generation": {"cycles_per_param": 0.1}
            },
            "federation_costs": {
                "PEC_hardware": {
                    "pipeline_depth": 14,
                    "throughput_bytes_per_cycle": 32,
                    "parallel_lanes": 1
                },
                "software_baseline": {
                    "software_conversion_factor": 12.0
                }
            }
        }

    def _cycles_to_ms(self, cycles: float) -> float:
        """周期转毫秒"""
        return (cycles * self.clock_period_ns) / 1e6

    def _pj_to_mj(self, pj: float) -> float:
        """皮焦转毫焦"""
        return pj / 1e9

    def simulate_layer_training(self,
                                macs: int,
                                params: int,
                                in_bytes: int,
                                out_bytes: int,
                                compute_mode: str,
                                **kwargs) -> Dict[str, float]:
        """
        单层训练模拟（Roofline模型）

        Returns:
            {
                'cycles': 实际执行周期,
                'energy_pJ': 能耗,
                'is_memory_bound': 是否受访存限制,
                'arithmetic_intensity': 算术强度
            }
        """
        mode = compute_mode if compute_mode in self.precision_bytes else 'FP32'

        # 1. 计算时间
        ops_per_cycle = self.hw_profile['design_parameters']['ops_per_cycle'].get(mode, 1024)
        t_compute_ideal = macs / ops_per_cycle

        # 2. 访存时间
        scale_factor = self.precision_bytes[mode] / 4.0
        total_data_bytes = (in_bytes + out_bytes) * scale_factor
        total_param_bytes = params * scale_factor
        total_transfer = total_data_bytes + total_param_bytes

        t_memory = total_transfer / self.bytes_per_cycle

        # 3. Roofline: 取最大值
        t_actual = max(t_compute_ideal, t_memory)
        is_mem_bound = t_memory > t_compute_ideal

        # 4. 算术强度 (Ops/Byte)
        arithmetic_intensity = macs / max(1, total_transfer)

        # 5. 能耗计算
        energy_pj_per_op = self.hw_profile['compute_costs_per_op'][mode]['energy_pJ']
        energy_compute = macs * energy_pj_per_op

        mem_energy_per_byte = self.hw_profile['memory_costs']['DRAM_access_per_byte_pJ']
        energy_memory = total_transfer * mem_energy_per_byte

        return {
            "cycles": t_actual,
            "energy_pJ": energy_compute + energy_memory,
            "is_memory_bound": is_mem_bound,
            "arithmetic_intensity": arithmetic_intensity
        }

    def simulate_model_training(self,
                                workload_layers: List[Dict],
                                policy: Dict[str, str],
                                enable_beu: bool = True) -> Dict[str, Any]:
        """
        完整模型训练模拟

        关键改进: 真实的BEU预算追踪
        """
        compute_mode = policy.get('compute', 'FP32')

        total_time_cycles = 0.0
        total_energy_pj = 0.0
        accumulated_budget_cycles = 0.0

        layer_details = []

        # 阶段1: 前向+反向计算
        for layer in workload_layers:
            # FP32基准
            res_fp32 = self.simulate_layer_training(**layer, compute_mode='FP32')
            # 实际精度
            res_actual = self.simulate_layer_training(**layer, compute_mode=compute_mode)

            total_time_cycles += res_actual['cycles']
            total_energy_pj += res_actual['energy_pJ']

            # BEU预算累积
            if enable_beu:
                saved = res_fp32['cycles'] - res_actual['cycles']
                if saved > 0:
                    accumulated_budget_cycles += saved

            layer_details.append({
                'name': layer.get('name', 'unknown'),
                'cycles': res_actual['cycles'],
                'saved_cycles': max(0, res_fp32['cycles'] - res_actual['cycles']),
                'is_memory_bound': res_actual['is_memory_bound']
            })

        # 阶段2: DP隐私开销
        security_mode = policy.get('security', 'None')
        dp_overhead_cycles = 0.0
        dp_energy_pj = 0.0
        dp_hidden_ratio = 0.0

        if security_mode != 'None':
            total_params = sum(l['params'] for l in workload_layers)
            batch_size = policy.get('batch_size', 4)

            # 计算DP成本
            clip_cost_per_param = self.hw_profile['security_costs']['per_example_clipping']['cycles_per_param']
            noise_cost_per_param = self.hw_profile['security_costs']['noise_generation']['cycles_per_param']

            dp_total_cycles = (total_params * batch_size * clip_cost_per_param) + \
                              (total_params * noise_cost_per_param)

            if enable_beu:
                if accumulated_budget_cycles >= dp_total_cycles:
                    # 完全隐藏
                    dp_overhead_cycles = 0
                    dp_hidden_ratio = 1.0
                else:
                    # 部分隐藏
                    dp_overhead_cycles = dp_total_cycles - accumulated_budget_cycles
                    dp_hidden_ratio = accumulated_budget_cycles / dp_total_cycles
            else:
                # 无BEU: 全部开销
                dp_overhead_cycles = dp_total_cycles
                dp_hidden_ratio = 0.0

            # DP能耗（简化）
            dp_energy_pj = dp_total_cycles * 2.0

        return {
            "cycles": total_time_cycles,
            "total_latency_ms": self._cycles_to_ms(total_time_cycles + dp_overhead_cycles),
            "compute_latency_ms": self._cycles_to_ms(total_time_cycles),
            "dp_overhead_ms": self._cycles_to_ms(dp_overhead_cycles),
            "total_energy_mJ": self._pj_to_mj(total_energy_pj + dp_energy_pj),
            "budget_accumulated_cycles": accumulated_budget_cycles,
            "budget_utilization": dp_hidden_ratio,
            "layer_details": layer_details
        }

    def simulate_aggregation(self,
                             num_clients: int,
                             model_size_mib: float,
                             method: str = 'PEC') -> float:
        """
        聚合延迟模拟

        Returns:
            latency_ms: 聚合延迟（毫秒）
        """
        if num_clients <= 0:
            raise ValueError("num_clients must be positive")
        if model_size_mib <= 0:
            raise ValueError("model_size_mib must be positive")

        update_bytes = model_size_mib * 1024 * 1024
        input_bytes = update_bytes * num_clients
        memory_bandwidth_bytes_s = self.mem_bw_gbps * 1e9

        if method == 'PEC':
            # PEC: O(1) 可扩展性
            pec_config = self.hw_profile['federation_costs']['PEC_hardware']
            pipeline_depth = float(pec_config['pipeline_depth'])
            throughput = float(pec_config['throughput_bytes_per_cycle'])
            parallel_lanes = int(pec_config.get('parallel_lanes', 1))
            if parallel_lanes <= 0:
                raise ValueError("PEC parallel_lanes must be positive")

            # 延迟 = 流水线填充 + 数据传输
            lane_bandwidth_bytes_s = (
                throughput * parallel_lanes * self.clock_freq_mhz * 1e6
            )
            effective_bandwidth_bytes_s = min(
                memory_bandwidth_bytes_s,
                lane_bandwidth_bytes_s,
            )
            transfer_seconds = input_bytes / effective_bandwidth_bytes_s
            pipeline_seconds = pipeline_depth / (self.clock_freq_mhz * 1e6)
            return (transfer_seconds + pipeline_seconds) * 1e3

        elif method == 'Software':
            # 软件: O(N) 线性增长
            alpha_sw = float(
                self.hw_profile['federation_costs']['software_baseline'][
                    'software_conversion_factor'
                ]
            )

            # 每个客户端的参数需要单独处理
            return alpha_sw * input_bytes / memory_bandwidth_bytes_s * 1e3

        else:
            # Baseline: 纯带宽限制
            return input_bytes / memory_bandwidth_bytes_s * 1e3

    def roofline_analysis(self, workload_layers: List[Dict], precision: str) -> Dict:
        """
        Roofline 模型分析

        用于生成论文中的 Roofline 图表
        """
        ops_per_cycle = self.hw_profile['design_parameters']['ops_per_cycle'][precision]
        peak_flops = ops_per_cycle * self.clock_freq_mhz * 1e6  # FLOPS

        mem_bw_bytes_per_sec = self.mem_bw_gbps * 1e9

        results = []
        for layer in workload_layers:
            macs = layer['macs']
            total_bytes = layer['in_bytes'] + layer['out_bytes'] + layer['params'] * self.precision_bytes[precision]

            arithmetic_intensity = macs / total_bytes  # Ops/Byte

            # Roofline
            compute_bound_perf = peak_flops
            memory_bound_perf = mem_bw_bytes_per_sec * arithmetic_intensity

            actual_perf = min(compute_bound_perf, memory_bound_perf)

            results.append({
                'layer': layer.get('name', 'unknown'),
                'arithmetic_intensity': arithmetic_intensity,
                'achieved_gflops': actual_perf / 1e9,
                'is_compute_bound': compute_bound_perf < memory_bound_perf
            })

        return {
            'peak_flops_gflops': peak_flops / 1e9,
            'peak_bandwidth_gbs': mem_bw_bytes_per_sec / 1e9,
            'layers': results
        }
