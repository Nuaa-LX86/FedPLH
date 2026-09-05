import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset.dataset_loader import (
    FULL_VOLUME_SCOPE,
    get_federated_dataloaders,
    write_partition_evidence,
)
from experiments.losses import DiceLoss
from experiments.microarchitecture_benchmark import benchmark_microarchitecture
from experiments.roofline_generator import generate_roofline_data
from experiments.scalability_test import test_pec_scalability
from models.fedmpq_proxy import FedMPQProxy
from models.precision_wrapper import HMPEPrecisionEmulator
from models.unet3d import UNet3D
from training.federated_trainer import FederatedTrainer
from training.sota_adapters import EvidentialDiceLoss3D, FedCLAMDiceFIMLoss3D
from utils.reproducibility import (
    build_client_schedule,
    collect_environment,
    dataset_inventory,
    derive_seed,
    fingerprint_files,
    seed_everything,
    sha256_file,
    write_json_atomic,
)
from visualization.plot_generator import PlotGenerator

try:
    from models.swin_unetr import SwinUNETR

    HAS_UNETR = True
except ImportError:
    HAS_UNETR = False


def ensure_hardware_profile(profile_path: str) -> Path:
    resolved = Path(profile_path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Hardware profile is required: {resolved}. Automatic profile "
            "generation is disabled to prevent configuration drift."
        )
    return resolved


def set_seed(seed: int, deterministic: bool = False):
    seed_everything(seed, deterministic=deterministic)



def parse_seeds(seeds: str, default_seed: int) -> List[int]:
    """Parse comma-separated seed list. Empty => [default_seed]."""
    if seeds and str(seeds).strip():
        return [int(s.strip()) for s in str(seeds).split(',') if s.strip()]
    return [int(default_seed)]


def summarize_seed_metrics(seed_to_metrics: Dict[int, Dict[str, Any]]):
    """Compute mean/std/ci95 across seeds for numeric metrics."""
    if not seed_to_metrics:
        return {}, {}, {}
    keys = set()
    for m in seed_to_metrics.values():
        keys.update(m.keys())

    mean, std, ci95 = {}, {}, {}
    n = len(seed_to_metrics)
    for k in keys:
        vals = []
        for m in seed_to_metrics.values():
            v = m.get(k, None)
            if isinstance(v, (int, float, np.floating, np.integer)):
                try:
                    fv = float(v)
                    if np.isfinite(fv):
                        vals.append(fv)
                except Exception:
                    pass
        if vals:
            mu = float(np.mean(vals))
            sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            mean[k] = mu
            std[k] = sd
            ci95[k] = float(1.96 * sd / max(1, np.sqrt(n)))

    # keep non-numeric fields (from first seed) for completeness
    first = seed_to_metrics[next(iter(seed_to_metrics))]
    for k in keys:
        if k not in mean:
            mean[k] = first.get(k)

    return mean, std, ci95

def build_model(
    model_name: str,
    compute_precision: str,
    hmpe_operand_model: str = "legacy_activation_only",
):
    if model_name == "unet":
        base_model = UNet3D(n_channels=4, n_classes=4)
    else:
        if not HAS_UNETR:
            raise ImportError("MONAI is required for SwinUNETR. Please install monai.")
        base_model = SwinUNETR(img_size=64, in_channels=4, out_channels=4)

    if compute_precision == "FedMPQ_PROXY":
        return FedMPQProxy(base_model, default_bits=8)

    # Wrap with precision emulator
    return HMPEPrecisionEmulator(
        base_model,
        default_precision=compute_precision,
        quantize_weights=(hmpe_operand_model == "quantized_operands"),
    )


def build_criterion(strategy: str, policy: Dict[str, Any]) -> torch.nn.Module:
    if strategy == "FedEvi":
        return EvidentialDiceLoss3D(
            kl_weight=float(policy.get("kl_weight", 0.01)),
            annealing_step=int(policy.get("annealing_step", 10)),
        )
    if strategy == "FedCLAM":
        return FedCLAMDiceFIMLoss3D(
            classes=4,
            lambda_dice=float(policy.get("lambda_dice", 0.5)),
            lambda_fim=float(policy.get("lambda_fim", 0.5)),
            fim_warmup_rounds=int(policy.get("fim_warmup_rounds", 10)),
            pooled_size=int(policy.get("fim_pooled_size", 16)),
        )
    return DiceLoss(n_classes=4)


def print_sensitivity_summary(suite_metrics: dict):
    """打印λ/σ/B_th敏感性分析汇总表，用于论文Table生成"""
    print("\n" + "=" * 80)
    print("SENSITIVITY ANALYSIS SUMMARY")
    # CUDA状态检查
    import torch
    cuda_ok = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_ok else "CPU only"
    print(f"{'=' * 80}")
    print(f"[GPU] {device_name} | CUDA: {cuda_ok} | AMP: {cuda_ok}")
    if not cuda_ok:
        print("WARNING: CUDA不可用，将使用CPU训练，速度极慢！")
        print(
            "请检查: conda activate hmpe_acf_env && pip install torch --index-url https://download.pytorch.org/whl/cu118")
    print("=" * 80)

    # λ扫描汇总
    lambda_rows = {k: v for k, v in suite_metrics.items() if k.startswith("S1_lamda")}
    if lambda_rows:
        print(f"\n{'λ':>6} | {'Dice(%)':>10} | {'Latency(norm)':>14} | {'Energy(norm)':>13} | {'HR_theo':>8}")
        print("-" * 60)
        ref_lat = None
        ref_eng = None
        for name, data in sorted(lambda_rows.items()):
            m = data.get("metrics", {})
            lat = m.get("avg_latency_ms", 0.0)
            eng = m.get("avg_energy_mJ", 0.0)
            if ref_lat is None:
                ref_lat = lat if lat > 0 else 1.0
            if ref_eng is None:
                ref_eng = eng if eng > 0 else 1.0
            lam_val = name.split("_")[-1]
            dice = m.get("test_dice", m.get("accuracy", 0.0))
            hr = m.get("avg_hr_theoretical", m.get("avg_dp_hidden_ratio", 0.0))
            print(f"{lam_val:>6} | {dice*100:>10.2f} | {lat/ref_lat:>14.3f} | {eng/ref_eng:>13.3f} | {hr:>8.3f}")

    # σ扫描汇总
    sigma_rows = {k: v for k, v in suite_metrics.items() if k.startswith("S3_sigma")}
    if sigma_rows:
        print(f"\n{'σ':>6} | {'Dice(%)':>10} | {'Latency(norm)':>14} | {'ε(δ)':>12} | {'HR_BEU':>8}")
        print("-" * 60)
        ref_lat = None
        for name, data in sorted(sigma_rows.items()):
            m = data.get("metrics", {})
            lat = m.get("avg_latency_ms", 0.0)
            if ref_lat is None:
                ref_lat = lat if lat > 0 else 1.0
            sig_val = name.split("_")[-1]
            dice = m.get("test_dice", m.get("accuracy", 0.0))
            eps = m.get("final_epsilon", 0.0)
            hr = m.get("avg_dp_hidden_ratio", 0.0)
            print(f"{sig_val:>6} | {dice*100:>10.2f} | {lat/ref_lat:>14.3f} | {eps:>12.1f} | {hr:>8.3f}")

    print("=" * 80)



def run_full_pipeline(args):
    seeds = parse_seeds(getattr(args, "seeds", ""), args.seed)
    # 注意：split_seed 控制数据划分；这里 seeds 控制训练初始化/客户端采样随机性
    set_seed(seeds[0] if seeds else args.seed, deterministic=args.deterministic)
    hardware_profile_path = ensure_hardware_profile(args.hardware_profile)

    print("=" * 80)
    print(f" HMPE-ACF Full SOTA Matrix | Model: {args.model.upper()} | Rounds: {args.rounds}")
    print("=" * 80)

    output_root = Path(args.output_root)
    results_base = output_root / args.model
    for subdir in ["microarch", "scalability", "summaries", "sensitivity", "figures"]:
        (results_base / subdir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1-3. Hardware Benchmarks
    # ------------------------------------------------------------------
    if args.step in ["micro", "all"] and args.model == "unet":
        benchmark_microarchitecture(output_dir=f"{results_base}/microarch")
        test_pec_scalability(output_dir=f"{results_base}/scalability")
        generate_roofline_data(output_dir=f"{results_base}/microarch")

    # ------------------------------------------------------------------
    # 4. Training & evaluation
    # ------------------------------------------------------------------
    if args.step in ["train", "all"]:
        batch_size = args.batch_size if args.batch_size is not None else (2 if args.model == "unet" else 1)

        # === 核心对比矩阵 (SOTA Matrix) ===
        common_dp = {
            "enable": True,
            "simulate_hardware_beu": False,  # 默认 Software-DP
            # [P0] PEC 与 BEU 解耦：聚合侧是否加速由 simulate_hardware_pec 控制
            "simulate_hardware_pec": False,
            # [P0] DP 开销模型：legacy 保持旧结果；paper 更贴近 DP-SGD 语义
            "dp_cost_model": args.dp_cost_model,
            # 其它杂项开销（若需要可调），默认 0 不影响你现有结果
            "misc_overhead_ms": 0.0,
            "clip_norm": args.clip_norm,
            "noise_multiplier": args.noise_multiplier,
            "delta": args.delta,
            "batch_size": batch_size,
            "dp_mode": "soft",
            "enable_accounting": True,  # 可选，但建议开着
        }

        sota_scenarios = {
            # 0) Upper bound
            "FP32_noDP": {
                "dp": {**common_dp, "enable": False},
                "acf": {"compute": "FP32", "strategy": "FedAvg"},
                "comm_interval": 1,
            },

            # 1) Software DP baseline
            "FP32_softDP": {
                "dp": {**common_dp, "simulate_hardware_beu": False},
                "acf": {"compute": "FP32", "strategy": "FedAvg"},
                "comm_interval": 1,
            },

            # 2) FedBN baseline (算法侧对比)
            "FedBN": {
                "dp": {**common_dp, "simulate_hardware_beu": False},
                "acf": {"compute": "FP32", "strategy": "FedBN"},
                "comm_interval": 1,
            },

            "FedEvi": {
                "dp": {**common_dp, "simulate_hardware_beu": False},
                "acf": {
                    "compute": "FP32",
                    "strategy": "FedEvi",
                    "gamma": 1.0,
                    "kl_weight": 0.01,
                    "annealing_step": 10,
                    "adaptation_scope": (
                        "mechanism-preserving adapter in common 3D harness"
                    ),
                },
                "comm_interval": 1,
            },

            "FedCLAM": {
                "dp": {**common_dp, "simulate_hardware_beu": False},
                "acf": {
                    "compute": "FP32",
                    "strategy": "FedCLAM",
                    "alpha": 1.0,
                    "beta": 1.0,
                    "agg_lr": 1.0,
                    "zero_init": False,
                    "lambda_dice": 0.5,
                    "lambda_fim": 0.5,
                    "fim_warmup_rounds": 10,
                    "fim_pooled_size": 16,
                    "adaptation_scope": (
                        "mechanism-preserving adapter with pooled 3D foreground matching"
                    ),
                },
                "comm_interval": 1,
            },

            # 3) FedPAQ baseline (系统侧对比：周期聚合)
            "FedPAQ": {
                "dp": {**common_dp, "simulate_hardware_beu": False},
                "acf": {
                    "compute": "FP32",
                    "strategy": "FedPAQ",
                    "quantization_levels": 255,
                    "local_update_steps": 5,
                },
                "comm_interval": 1,
            },

            # 4) Mao et al. (硬件侧：BF16 + software DP)
            # Feasibility gate only; not a reportable FedMPQ result.
            "FedMPQ_proxy": {
                "dp": {**common_dp, "simulate_hardware_beu": False},
                "acf": {
                    "compute": "FedMPQ_PROXY",
                    "strategy": "FedMPQProxy",
                    "group_lasso_lambda": 0.001,
                    "pruning_threshold": 0.02,
                    "client_budgets": [
                        2, 2, 4, 4, 4, 6, 6, 6, 8, 8,
                        2, 2, 4, 4, 4, 6, 6, 6, 8, 8,
                    ],
                },
                "comm_interval": 1,
            },

            "Mao_etal": {
                "dp": {**common_dp, "simulate_hardware_beu": False},
                "acf": {"compute": "BF16", "strategy": "FedAvg"},
                "comm_interval": 1,
            },

            # 5) BitFusion (硬件侧：INT8，无 DP)
            "BitFusion": {
                "dp": {**common_dp, "enable": False},
                "acf": {"compute": "INT8", "strategy": "FedAvg"},
                "comm_interval": 1,
            },

            # 6) Ours w/o DP（用于证明：不是“没 DP 才快”）
            "HMPE-ACF_noDP": {
                "dp": {
                    **common_dp,
                    "enable": False,
                    "simulate_hardware_beu": True,
                    "simulate_hardware_pec": True,
                },
                "acf": {"compute": "Mixed", "strategy": "EntropyAware"},
                "comm_interval": 1,
            },

            # 7) Ours full
            "HMPE-ACF": {
                "dp": {**common_dp, "simulate_hardware_beu": True, "simulate_hardware_pec": True},
                "acf": {"compute": "Mixed", "strategy": "EntropyAware"},
                "comm_interval": 1,
            },
            # 8) HMPE + BEU, 静态FP8，无ACF动态调度
            # 用于证明ACF的独立贡献：对比此条与HMPE-ACF，
            # Dice差异来自ACF的熵感知调度
            "HMPE-BEU-noACF": {
                "dp": {**common_dp, "simulate_hardware_beu": True, "simulate_hardware_pec": False},
                "acf": {"compute": "FP8_E5M2", "strategy": "FedAvg"},
                "comm_interval": 1,
                },
            # 9) HMPE-only: 多精度FP8但软件DP，无BEU无ACF
            # 用于证明：多精度本身不能隐藏DP开销
            "HMPE-only": {
                "dp": {**common_dp, "simulate_hardware_beu": False},
                "acf": {"compute": "FP8_E5M2", "strategy": "FedAvg"},
                "comm_interval": 1,
            },

        }


        # === 消融实验 (Ablation) ===
        # 目标：分离 HMPE / BEU / PEC / ACF 各自带来的收益
        fedmpq_screen_scenarios = {
            "FedMPQ_proxy": sota_scenarios.pop("FedMPQ_proxy")
        }

        ablation_scenarios = {
            # A0: 纯软件基线 (FP32 + SoftDP + Software Aggregation)
            "A0_FP32_softDP_SW": {
                "dp": {**common_dp, "simulate_hardware_beu": False, "simulate_hardware_pec": False},
                "acf": {"compute": "FP32", "strategy": "FedAvg"},
                "comm_interval": 1,
            },
            # A1: 仅 HMPE（Static FP8）但仍是软件 DP + 软件聚合
            "A1_FP8_softDP_SW": {
                "dp": {**common_dp, "simulate_hardware_beu": False, "simulate_hardware_pec": False},
                "acf": {"compute": "FP8_E5M2", "strategy": "FedAvg"},
                "comm_interval": 1,
            },
            # A2: HMPE + BEU（DP 开销尽量被隐藏），聚合仍是软件
            "A2_FP8_softDP_BEU": {
                "dp": {**common_dp, "simulate_hardware_beu": True, "simulate_hardware_pec": False},
                "acf": {"compute": "FP8_E5M2", "strategy": "FedAvg"},
                "comm_interval": 1,
            },
            # A3: HMPE + BEU + PEC（聚合加速）
            "A3_FP8_softDP_BEU_PEC": {
                "dp": {**common_dp, "simulate_hardware_beu": True, "simulate_hardware_pec": True},
                "acf": {"compute": "FP8_E5M2", "strategy": "FedAvg"},
                "comm_interval": 1,
            },
            # A4: Full (HMPE + ACF + BEU + PEC)
            "A4_HMPE-ACF_full": {
                "dp": {**common_dp, "simulate_hardware_beu": True, "simulate_hardware_pec": True},
                "acf": {"compute": "Mixed", "strategy": "EntropyAware", "mode": "entropy_time", "lamda": 0.5, "budget_threshold": 0.0},
                "comm_interval": 1,
            },
        }

        # ACF mechanism evidence. Every scheduler field is explicit so the
        # run manifest is sufficient to reproduce precision assignments.
        acf_evidence_scenarios = {
            "ACF_static_BF16": {
                "dp": {
                    **common_dp,
                    "simulate_hardware_beu": True,
                    "simulate_hardware_pec": True,
                },
                "acf": {
                    "compute": "Mixed",
                    "strategy": "EntropyAware",
                    "mode": "static_high",
                    "lamda": 0.0,
                    "budget_threshold": 0.0,
                    "deterministic": True,
                    "scheduler_stream": "acf_scheduler:static_bf16",
                    "low_precision": "FP8_E5M2",
                    "high_precision": "BF16",
                },
                "comm_interval": 1,
            },
            "ACF_static_FP8": {
                "dp": {
                    **common_dp,
                    "simulate_hardware_beu": True,
                    "simulate_hardware_pec": True,
                },
                "acf": {
                    "compute": "Mixed",
                    "strategy": "EntropyAware",
                    "mode": "static_low",
                    "lamda": 0.0,
                    "budget_threshold": 0.0,
                    "deterministic": True,
                    "scheduler_stream": "acf_scheduler:static_fp8",
                    "low_precision": "FP8_E5M2",
                    "high_precision": "BF16",
                },
                "comm_interval": 1,
            },
            "ACF_progress_only": {
                "dp": {
                    **common_dp,
                    "simulate_hardware_beu": True,
                    "simulate_hardware_pec": True,
                },
                "acf": {
                    "compute": "Mixed",
                    "strategy": "EntropyAware",
                    "mode": "time_decay",
                    "lamda": 0.0,
                    "budget_threshold": 0.0,
                    "deterministic": False,
                    "scheduler_stream": "acf_scheduler:progress_only",
                    "low_precision": "FP8_E5M2",
                    "high_precision": "BF16",
                },
                "comm_interval": 1,
            },
            "ACF_entropy_only": {
                "dp": {
                    **common_dp,
                    "simulate_hardware_beu": True,
                    "simulate_hardware_pec": True,
                },
                "acf": {
                    "compute": "Mixed",
                    "strategy": "EntropyAware",
                    "mode": "entropy_only",
                    "lamda": 1.0,
                    "budget_threshold": 0.0,
                    "deterministic": False,
                    "scheduler_stream": "acf_scheduler:entropy_only",
                    "low_precision": "FP8_E5M2",
                    "high_precision": "BF16",
                },
                "comm_interval": 1,
            },
            "ACF_full": {
                "dp": {
                    **common_dp,
                    "simulate_hardware_beu": True,
                    "simulate_hardware_pec": True,
                },
                "acf": {
                    "compute": "Mixed",
                    "strategy": "EntropyAware",
                    "mode": "entropy_time",
                    "lamda": 0.5,
                    "budget_threshold": 0.0,
                    "deterministic": False,
                    "scheduler_stream": "acf_scheduler",
                    "low_precision": "FP8_E5M2",
                    "high_precision": "BF16",
                },
                "comm_interval": 1,
            },
            "ACF_full_noDP": {
                "dp": {
                    **common_dp,
                    "enable": False,
                    "simulate_hardware_beu": True,
                    "simulate_hardware_pec": True,
                },
                "acf": {
                    "compute": "Mixed",
                    "strategy": "EntropyAware",
                    "mode": "entropy_time",
                    "lamda": 0.5,
                    "budget_threshold": 0.0,
                    "deterministic": False,
                    "low_precision": "FP8_E5M2",
                    "high_precision": "BF16",
                },
                "comm_interval": 1,
            },
            "ACF_FedBN": {
                "dp": {
                    **common_dp,
                    "simulate_hardware_beu": False,
                    "simulate_hardware_pec": False,
                },
                "acf": {
                    "compute": "FP32",
                    "strategy": "FedBN",
                    "mode": "static_high",
                    "lamda": 0.0,
                    "budget_threshold": 0.0,
                    "deterministic": True,
                    "low_precision": "FP8_E5M2",
                    "high_precision": "BF16",
                },
                "comm_interval": 1,
            },
        }

        # === 敏感性实验 (Sensitivity) ===
        # 目标：验证 λ / 预算阈值 B_th / 噪声 σ 对 accuracy-latency-energy 的影响
        sensitivity_scenarios = {}

        # (S1) λ 扫描：0 表示完全时间项，1 表示完全空间项
        for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
            sensitivity_scenarios[f"S1_lamda_{lam:.2f}"] = {
                "dp": {**common_dp, "simulate_hardware_beu": True, "simulate_hardware_pec": True},
                "acf": {"compute": "Mixed", "strategy": "EntropyAware", "mode": "entropy_time", "lamda": float(lam), "budget_threshold": 0.0},
                "comm_interval": 1,
            }

        # (S2) 预算阈值 B_th 扫描（cycles）
        for bth in [0.0, 5e5, 1e6, 2e6]:
            sensitivity_scenarios[f"S2_budget_{int(bth)}"] = {
                "dp": {**common_dp, "simulate_hardware_beu": True, "simulate_hardware_pec": True},
                "acf": {"compute": "Mixed", "strategy": "EntropyAware", "mode": "entropy_time", "lamda": 0.5, "budget_threshold": float(bth)},
                "comm_interval": 1,
            }

        # (S3) 噪声 σ 扫描（Soft-DP）
        # (S3) 噪声 σ 扫描（Soft-DP -> 有意义DP区间）
        # 0.1: 当前默认(SoftDP工程口径)
        # 0.5/1.0/2.0: 进入有意义差分隐私区间，验证BEU在更强噪声下的隐藏能力
        for sig in [0.1, 0.5, 1.0, 2.0]:
            sensitivity_scenarios[f"S3_sigma_{sig:.2f}"] = {
                "dp": {**common_dp, "noise_multiplier": float(sig), "simulate_hardware_beu": True, "simulate_hardware_pec": True},
                "acf": {"compute": "Mixed", "strategy": "EntropyAware", "mode": "entropy_time", "lamda": 0.5, "budget_threshold": 0.0},
                "comm_interval": 1,
            }

        # -----------------------------
        # Scenario selection (SOTA / Ablation / Sensitivity)
        # -----------------------------
        scenario_groups = {
            "sota": sota_scenarios,
            "fedmpq_screen": fedmpq_screen_scenarios,
            "ablation": ablation_scenarios,
            "sensitivity": sensitivity_scenarios,
            "acf_evidence": acf_evidence_scenarios,
        }
        suite = getattr(args, "suite", "sota")
        if suite == "all":
            selected_scenarios = {}
            for g in scenario_groups.values():
                selected_scenarios.update(g)
        else:
            selected_scenarios = scenario_groups.get(suite, sota_scenarios)

        # optionally filter scenarios by name
        if args.scenarios:
            keep = set([s.strip() for s in args.scenarios.split(",") if s.strip()])
            selected_scenarios = {k: v for k, v in selected_scenarios.items() if k in keep}

        if not selected_scenarios:
            raise ValueError("No scenarios selected")

        client_schedules = {
            str(seed): build_client_schedule(
                num_clients=args.clients,
                rounds=args.rounds,
                client_fraction=args.client_fraction,
                seed=seed,
            )
            for seed in seeds
        }
        source_files = [
            "main_experiment.py",
            "dataset/dataset_loader.py",
            "models/precision_wrapper.py",
            "training/acf_scheduler.py",
            "training/aggregation.py",
            "training/dp_sgd.py",
            "training/federated_trainer.py",
            "simulator/acf_simulator.py",
            "experiments/scalability_test.py",
            "plot_beu_boundary.py",
            "scripts/build_paper_results.py",
            "scripts/generate_tpds_figures.py",
            "scripts/run_tpds_final_profile_smoke.ps1",
            "scripts/run_tpds_matched_five_seed.ps1",
            "scripts/run_tpds_sota_five_seed.ps1",
            "scripts/build_simulator_profile_from_fpga.py",
            "scripts/export_partition_evidence.py",
            "scripts/plot_non_iid_characterization.py",
            "scripts/build_acf_evidence.py",
            "utils/reproducibility.py",
            "visualization/plot_generator.py",
        ]
        run_manifest_path = output_root / "run_manifest.json"
        run_manifest = {
            "schema_version": 1,
            "status": "running",
            "command": [sys.executable, *sys.argv],
            "arguments": vars(args),
            "environment": collect_environment(),
            "dataset": dataset_inventory(Path(args.data_root)),
            "hardware_profile": {
                "path": str(hardware_profile_path),
                "sha256": sha256_file(hardware_profile_path),
            },
            "source_sha256": fingerprint_files(Path.cwd(), source_files),
            "methodology": {
                "local_update_unit": "epochs",
                "local_epochs": args.local_epochs,
                "client_fraction": args.client_fraction,
                "privacy_accounting": "per-client cumulative RDP; curve is max over all clients",
                "normalization": "paired by training seed against FedBN",
                "dp_cost_model": args.dp_cost_model,
                "amp_enabled": args.amp,
                "entropy_scope": args.entropy_scope,
                "partition_allocation_unit": "complete patient/volume",
                "global_split_seed": args.split_seed,
                "partition_seed": (
                    args.split_seed
                    if args.partition_seed is None
                    else args.partition_seed
                ),
                "partition_input": (
                    {
                        "path": str(Path(args.partition_file).resolve()),
                        "sha256": sha256_file(Path(args.partition_file)),
                    }
                    if args.partition_file
                    else None
                ),
                "min_client_samples": args.min_client_samples,
                "balance_client_sizes": args.balance_client_sizes,
                "partition_basis": args.partition_basis,
                "composition_bins": args.composition_bins,
                "aggregation_weight_basis": (
                    "participating-client local training-set sample count"
                ),
                "fedpaq_semantics": (
                    "each selected client performs tau=5 local optimizer "
                    "updates before every server communication; stochastic "
                    "quantized model increments; sample-weighted adapted baseline"
                ),
            },
            "seeds": seeds,
            "client_schedules": client_schedules,
            "scenario_configs": selected_scenarios,
        }
        if run_manifest_path.exists() and args.resume:
            with run_manifest_path.open("r", encoding="utf-8") as handle:
                existing_manifest = json.load(handle)
            critical_keys = [
                "model",
                "rounds",
                "clients",
                "data_root",
                "val_ratio",
                "test_ratio",
                "split_seed",
                "partition_seed",
                "partition_file",
                "iid",
                "alpha",
                "entropy_scope",
                "min_client_samples",
                "balance_client_sizes",
                "partition_basis",
                "composition_bins",
                "client_fraction",
                "local_epochs",
                "img_size",
                "batch_size",
                "lr",
                "noise_multiplier",
                "clip_norm",
                "delta",
                "dp_cost_model",
                "deterministic",
                "amp",
                "suite",
                "scenarios",
                "seeds",
                "hardware_profile",
            ]
            old_args = existing_manifest.get("arguments", {})
            changed = [
                key
                for key in critical_keys
                if old_args.get(key) != run_manifest["arguments"].get(key)
            ]
            if changed:
                raise ValueError(
                    f"Cannot resume because critical arguments changed: {changed}"
                )
            if existing_manifest.get("hardware_profile", {}).get("sha256") != (
                run_manifest["hardware_profile"]["sha256"]
            ):
                raise ValueError("Cannot resume because the hardware profile changed")
            if existing_manifest.get("source_sha256") != run_manifest["source_sha256"]:
                raise ValueError("Cannot resume because experiment source files changed")
            if existing_manifest.get("dataset") != run_manifest["dataset"]:
                raise ValueError("Cannot resume because the dataset inventory changed")
        write_json_atomic(run_manifest_path, run_manifest)

        print(f"\n [Suite]  {suite} | Scenarios: {len(selected_scenarios)} | Seeds: {seeds}")

        suite_metrics: Dict[str, Any] = {}

        for name, cfg in selected_scenarios.items():
            print(f"\n Running Scenario: {name}")

            seed_to_metrics: Dict[int, Dict[str, Any]] = {}

            for seed in seeds:
                print(f"   ↳ Seed: {seed}")
                set_seed(seed, deterministic=args.deterministic)

                if args.split_seed == 0:
                    out_dir = results_base / name.replace(" ", "_") / f"seed{seed}"
                else:
                    out_dir = results_base / name.replace(" ", "_") / f"split{args.split_seed}" / f"seed{seed}"
                out_dir.mkdir(parents=True, exist_ok=True)

                history_path = out_dir / "training_history.json"
                seed_manifest_path = out_dir / "run_manifest.json"
                if history_path.exists() and args.resume and seed_manifest_path.exists():
                    with seed_manifest_path.open("r", encoding="utf-8") as handle:
                        completed_manifest = json.load(handle)
                    if completed_manifest.get("status") == "completed":
                        if completed_manifest.get("scenario_config") != cfg:
                            raise ValueError(
                                f"Scenario config changed for {name}, seed {seed}"
                            )
                        if completed_manifest.get("client_schedule") != (
                            client_schedules[str(seed)]
                        ):
                            raise ValueError(
                                f"Client schedule changed for {name}, seed {seed}"
                            )
                        with history_path.open("r", encoding="utf-8") as handle:
                            completed_history = json.load(handle)
                        seed_to_metrics[int(seed)] = completed_history.get(
                            "metrics",
                            {},
                        )
                        print(f"   Resume: using completed {name}, seed {seed}")
                        continue

                if history_path.exists() and not (args.overwrite or args.resume):
                    raise FileExistsError(
                        f"Refusing to overwrite existing result: {history_path}. "
                        "Use a new --output_root or pass --overwrite."
                    )

                loaders, val_loader, test_loader, stats = get_federated_dataloaders(
                    num_clients=args.clients,
                    batch_size=batch_size,
                    img_size=args.img_size,
                    iid=args.iid,
                    alpha=args.alpha,
                    data_root=args.data_root,
                    val_ratio=args.val_ratio,
                    test_ratio=args.test_ratio,
                    split_seed=args.split_seed,
                    partition_seed=args.partition_seed,
                    partition_file=args.partition_file,
                    force_split=args.force_split,
                    loader_seed=seed,
                    allow_synthetic=args.allow_synthetic,
                    entropy_scope=args.entropy_scope,
                    min_client_samples=args.min_client_samples,
                    balance_client_sizes=args.balance_client_sizes,
                    partition_basis=args.partition_basis,
                    composition_bins=args.composition_bins,
                )

                partition_evidence = getattr(stats, "metadata", None)
                if partition_evidence:
                    evidence_path, distribution_csv_path = write_partition_evidence(
                        partition_evidence,
                        output_root,
                    )
                    evidence_record = {
                        "json_path": str(evidence_path.resolve()),
                        "json_sha256": sha256_file(evidence_path),
                        "csv_path": str(distribution_csv_path.resolve()),
                        "entropy_definition": partition_evidence[
                            "entropy_definition"
                        ],
                        "partition": partition_evidence["partition"],
                        "sample_count_summary": partition_evidence[
                            "sample_count_summary"
                        ],
                        "entropy_summary": partition_evidence[
                            "entropy_summary"
                        ],
                        "realized_heterogeneity": partition_evidence[
                            "realized_heterogeneity"
                        ],
                    }
                    previous_evidence = run_manifest.get("partition_evidence")
                    if (
                        previous_evidence
                        and previous_evidence.get("json_sha256")
                        != evidence_record["json_sha256"]
                    ):
                        raise ValueError(
                            "Data partition changed within one experiment root"
                        )
                    run_manifest["partition_evidence"] = evidence_record
                    write_json_atomic(run_manifest_path, run_manifest)

                model = build_model(
                    args.model,
                    cfg["acf"]["compute"],
                    hmpe_operand_model=args.hmpe_operand_model,
                )
                resolved_acf = {
                    "compute": cfg["acf"].get("compute"),
                    "strategy": cfg["acf"].get("strategy"),
                    "mode": cfg["acf"].get("mode", "entropy_time"),
                    "lamda": float(
                        cfg["acf"].get(
                            "lamda",
                            cfg["acf"].get("alpha", 0.5),
                        )
                    ),
                    "budget_threshold": float(
                        cfg["acf"].get("budget_threshold", 0.0)
                    ),
                    "deterministic_decision": bool(
                        cfg["acf"].get("deterministic", False)
                    ),
                    "decision_rule": (
                        "threshold_at_0.5"
                        if cfg["acf"].get("deterministic", False)
                        else "seeded_bernoulli"
                    ),
                    "low_precision": cfg["acf"].get(
                        "low_precision",
                        "FP8_E5M2",
                    ),
                    "high_precision": cfg["acf"].get(
                        "high_precision",
                        "BF16",
                    ),
                    "scheduler_seed_namespace": cfg["acf"].get(
                        "scheduler_stream",
                        "acf_scheduler",
                    ),
                    "scheduler_seed": derive_seed(
                        seed,
                        cfg["acf"].get("scheduler_stream", "acf_scheduler"),
                    ),
                }
                seed_manifest = {
                    "schema_version": 1,
                    "status": "running",
                    "scenario": name,
                    "scenario_config": cfg,
                    "seed": int(seed),
                    "global_split_seed": int(args.split_seed),
                    "partition_seed": int(
                        run_manifest.get("partition_evidence", {})
                        .get("partition", {})
                        .get(
                            "partition_seed",
                            args.split_seed
                            if args.partition_seed is None
                            else args.partition_seed,
                        )
                    ),
                    "partition_file": (
                        str(Path(args.partition_file).resolve())
                        if args.partition_file
                        else None
                    ),
                    "client_schedule": client_schedules[str(seed)],
                    "loader_seed": derive_seed(seed, "data_loader"),
                    "resolved_acf": resolved_acf,
                    "partition_evidence_sha256": (
                        run_manifest.get("partition_evidence", {})
                        .get("json_sha256")
                    ),
                    "parent_manifest": str(run_manifest_path.resolve()),
                }
                write_json_atomic(seed_manifest_path, seed_manifest)

                trainer = FederatedTrainer(
                    model=model,
                    optimizer_fn=lambda p: torch.optim.AdamW(p, lr=float(args.lr)),
                    criterion=build_criterion(
                        str(cfg["acf"].get("strategy", "FedAvg")),
                        cfg["acf"],
                    ),
                    dp_config=cfg["dp"],
                    acf_policy=cfg["acf"],
                    hw_profile_path=str(hardware_profile_path),
                    output_dir=str(out_dir),
                    comm_interval=cfg["comm_interval"],
                    run_seed=seed,
                    client_schedule=client_schedules[str(seed)],
                    use_amp=args.amp,
                )

                hist = trainer.run(
                    loaders,
                    val_loader,
                    test_loader,
                    stats,
                    args.rounds,
                    local_epochs=args.local_epochs,
                )
                seed_to_metrics[int(seed)] = hist.get("metrics", {})
                seed_manifest["status"] = "completed"
                seed_manifest["metrics"] = hist.get("metrics", {})
                write_json_atomic(seed_manifest_path, seed_manifest)

                # Save one representative full history for plotting (default: HMPE-ACF, first seed)
                if (name in ["HMPE-ACF", "A4_HMPE-ACF_full"]) and (seed == seeds[0]):
                    with open(results_base / "training_history.json", "w", encoding="utf-8") as f:
                        json.dump(hist, f, indent=2)

            mean_m, std_m, ci95_m = summarize_seed_metrics(seed_to_metrics)
            suite_metrics[name] = {
                "metrics": mean_m,
                "metrics_std": std_m,
                "metrics_ci95": ci95_m,
                "seeds": seed_to_metrics,
                "scenario_config": cfg,
            }

        # -----------------------------
        # Save suite results
        # -----------------------------
        out_path = results_base / "summaries" / f"{suite}_results.json"

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(suite_metrics, f, indent=2)
            # 打印敏感性汇总（方便直接从终端读数填表）
        if suite in ["sensitivity", "all"]:
            print_sensitivity_summary(suite_metrics)
        run_manifest["status"] = "completed"
        run_manifest["summary_path"] = str(out_path.resolve())
        write_json_atomic(run_manifest_path, run_manifest)
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # 5. Plotting
    # ------------------------------------------------------------------
    if args.step in ["plot", "all"]:
        print("\n[Plotting] Generating Figures...")
        figures_dir = (
            Path(args.figures_root)
            if args.figures_root
            else output_root / f"paper_figures_{args.model}"
        )
        pg = PlotGenerator(output_dir=str(figures_dir))

        # ✅ 统一入口：自动生成 Fig5~Fig9（以及存在就画）
        if hasattr(pg, "generate_all_figures"):
            pg.generate_all_figures(results_dir=str(results_base))
        else:
            # 兼容旧版 plot_generator：至少不让它直接崩
            if os.path.exists(f"{results_base}/training_history.json") and hasattr(pg, "plot_training_curves"):
                pg.plot_training_curves(
                    json.load(open(f"{results_base}/training_history.json", "r", encoding="utf-8")),
                    save_name="Fig5_Convergence"
                )
            if os.path.exists(f"{results_base}/summaries/sota_results.json") and hasattr(pg, "plot_sota_summary"):
                pg.plot_sota_summary(
                    json.load(open(f"{results_base}/summaries/sota_results.json", "r", encoding="utf-8")),
                    save_name="Fig6_SOTA_Comparison"
                )
            if os.path.exists(f"{results_base}/microarch/roofline_data.json") and hasattr(pg, "plot_roofline_model"):
                pg.plot_roofline_model(
                    json.load(open(f"{results_base}/microarch/roofline_data.json", "r", encoding="utf-8")),
                    save_name="Fig8_Roofline_Generality"
                )
            if os.path.exists(f"{results_base}/microarch/microarch_results.json") and hasattr(pg,
                                                                                              "plot_precision_comparison"):
                pg.plot_precision_comparison(
                    json.load(open(f"{results_base}/microarch/microarch_results.json", "r", encoding="utf-8")),
                    save_name="Fig9_Precision_Efficiency"
                )

    print(f"\n✅ All Done. Results in paper_figures_{args.model}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--step", default="all", choices=["all", "micro", "train", "plot"])
    parser.add_argument("--model", default="unet", choices=["unet", "unetr"])
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--clients", type=int, default=10)
    parser.add_argument("--data_root", type=str, default="./dataset/processed")
    parser.add_argument("--allow_synthetic", action="store_true")
    parser.add_argument("--output_root", type=str, default="./results")
    parser.add_argument(
        "--hardware_profile",
        type=str,
        default="./hardware_profile.json",
        help="Audited hardware profile used by the trace-based simulator.",
    )
    parser.add_argument("--figures_root", type=str, default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable CUDA AMP. Disabled by default for matched paper runs.",
    )

    # data split
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument(
        "--partition_seed",
        type=int,
        default=None,
        help=(
            "Client-partition seed. Defaults to --split_seed for backward "
            "compatibility, but should be varied independently in robustness runs."
        ),
    )
    parser.add_argument(
        "--partition_file",
        type=str,
        default="",
        help=(
            "Optional frozen partition_evidence.json. When provided, its "
            "client sample_indices are used directly instead of regeneration."
        ),
    )
    parser.add_argument("--force_split", action="store_true")

    # partition non-iid
    parser.add_argument("--iid", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument(
        "--entropy_scope",
        type=str,
        default=FULL_VOLUME_SCOPE,
        choices=[FULL_VOLUME_SCOPE, "legacy_center_slices"],
    )
    parser.add_argument(
        "--min_client_samples",
        type=int,
        default=1,
        help="Minimum number of complete patient volumes assigned to each client.",
    )
    parser.add_argument(
        "--balance_client_sizes",
        action="store_true",
        help=(
            "Constrain clients to differ by at most one patient volume; "
            "useful for isolating label skew from quantity skew."
        ),
    )
    parser.add_argument(
        "--partition_basis",
        choices=[
            "dominant_foreground_label",
            "foreground_composition_quantiles",
        ],
        default="dominant_foreground_label",
    )
    parser.add_argument(
        "--composition_bins",
        type=int,
        default=10,
    )
    parser.add_argument("--client_fraction", type=float, default=0.2)

    # training
    parser.add_argument("--local_epochs", type=int, default=2)
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--hmpe_operand_model",
        choices=["legacy_activation_only", "quantized_operands"],
        default="legacy_activation_only",
        help=(
            "Use legacy activation-only fake quantization for exact reproduction of "
            "frozen runs, or quantize both HMPE operands with FP32 accumulation for "
            "new hardware-aligned training runs."
        ),
    )

    # dp params (Soft-DP)
    parser.add_argument("--noise_multiplier", type=float, default=0.1)
    parser.add_argument("--clip_norm", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--dp_cost_model", choices=["paper", "legacy"], default="paper")

    # misc
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--seeds",
        type=str,
        default="",
        help="Comma-separated seed list (e.g., 0,1,2). Empty => use --seed only.",
    )
    parser.add_argument(
        "--suite",
        type=str,
        default="sota",
        choices=[
            "sota",
            "fedmpq_screen",
            "ablation",
            "sensitivity",
            "acf_evidence",
            "all",
        ],
        help="Which scenario suite to run.",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default="",
        help="Comma-separated scenario names to run (e.g., FP32_noDP,FP32_softDP,HMPE-ACF). Empty => run all.",
    )

    args = parser.parse_args()
    run_full_pipeline(args)
