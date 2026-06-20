# training/dp_sgd.py
# DP-SGD + (Soft) RDP accounting + BEU budget (hardware masking)

import torch
import torch.nn as nn
from torch.optim import Optimizer
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

# 尝试导入 Opacus 进行精确会计，如果没有则使用近似
try:
    from opacus.accountants.analysis import rdp as rdp_analysis

    HAS_OPACUS = True
except ImportError:
    HAS_OPACUS = False
    # 静默处理，避免干扰输出
    pass


LEGACY_RDP_ORDERS = (
    [1 + x / 10.0 for x in range(1, 100)]
    + list(range(12, 64))
)

# Low noise multipliers can place the optimum close to alpha=1.  The legacy
# grid started at 1.1 and produced a very loose epsilon bound for sigma=0.1.
DEFAULT_RDP_ORDERS = (
    [1 + x / 1000.0 for x in range(1, 100)]
    + [1 + x / 10.0 for x in range(1, 100)]
    + list(range(12, 128))
)


# =========================================================
# RDP Accountant (跨轮次累计)
# ---------------------------------------------------------
# 目标：
# 1) 修复原实现中 q=bs/10000 的硬编码问题
# 2) 修复“每轮重置 steps 导致 ε 只统计单轮”的问题
# 3) 提供对齐论文 Algorithm 2 (行 14–23) 的接口：
#    - 累计/预测 ε
#    - 在需要时支持“按目标 ε 反求噪声系数”的在线噪声自适应（可选）
# =========================================================


@dataclass
class RDPSegment:
    """一段具有相同采样率 q 与噪声系数 σ 的连续 DP-SGD 迭代。"""

    sample_rate: float
    noise_multiplier: float
    steps: int


class RDPAccountant:
    """
    轻量 RDP 会计器：以“分段累积”的方式跨轮次累计隐私损耗。

    - 每次本地训练结束，只需调用 add_segment(q, σ, steps)
    - 计算 ε 时对每个 segment 调一次 opacus 的 compute_rdp，再求和

    说明：
    - 这里的 q 使用近似 q = batch_size / |D_k| (Poisson/with-replacement 近似)
    - 与严格 DP-SGD (逐样本梯度裁剪) 不同：本仓库的训练端目前是“soft-DP”近似。
      但“会计器”本身仍按标准 RDP 公式累计，可用于论文/系统端的预算反馈接口对齐。
    """

    def __init__(self, orders: Optional[List[float]] = None):
        if orders is None:
            orders = DEFAULT_RDP_ORDERS
        self.orders = [float(o) for o in orders]
        self.segments: List[RDPSegment] = []

    @staticmethod
    def _safe_sample_rate(q: float) -> float:
        if not np.isfinite(q):
            return 0.0
        return float(min(1.0, max(0.0, q)))

    def add_segment(self, sample_rate: float, noise_multiplier: float, steps: int):
        """追加一段 (q, σ, steps)。若与最后一段参数一致则自动合并。"""
        steps = int(steps)
        if steps <= 0:
            return

        q = self._safe_sample_rate(sample_rate)
        sigma = float(noise_multiplier)
        if not np.isfinite(sigma) or sigma <= 0:
            # σ<=0 没有意义，直接丢弃，避免 opacus 崩溃
            return

        for segment in self.segments:
            if (
                abs(segment.sample_rate - q) < 1e-12
                and abs(segment.noise_multiplier - sigma) < 1e-12
            ):
                segment.steps += steps
                return
        self.segments.append(RDPSegment(q, sigma, steps))

    def total_steps(self) -> int:
        return int(sum(s.steps for s in self.segments))

    def get_epsilon_and_order(
        self,
        delta: float = 1e-5,
    ) -> Tuple[float, Optional[float]]:
        """Return cumulative epsilon and the selected Renyi order."""
        if not self.segments:
            return 0.0, None

        if not HAS_OPACUS:
            # 兜底：给出单调可用的近似值（仅用于不装 opacus 的演示环境）
            # ε 大致与 steps / σ^2 增长，这里只保证趋势而非严格数值
            s0 = self.segments[-1]
            t = self.total_steps()
            epsilon = float(
                (s0.sample_rate * np.sqrt(t))
                / max(s0.noise_multiplier, 1e-6)
            )
            return epsilon, None

        orders = self.orders
        rdp_total = np.zeros(len(orders), dtype=np.float64)
        for seg in self.segments:
            rdp_total += rdp_analysis.compute_rdp(
                q=seg.sample_rate,
                noise_multiplier=seg.noise_multiplier,
                steps=seg.steps,
                orders=orders,
            )
        epsilon, order = rdp_analysis.get_privacy_spent(
            orders=orders,
            rdp=rdp_total,
            delta=delta,
        )
        return float(epsilon), float(order)

    def get_epsilon(self, delta: float = 1e-5) -> float:
        """Return cumulative epsilon."""
        epsilon, _ = self.get_epsilon_and_order(delta)
        return epsilon

    def predict_epsilon(self,
                        sample_rate: float,
                        noise_multiplier: float,
                        steps: int,
                        delta: float = 1e-5) -> float:
        """预测：如果再追加一段 (q, σ, steps)，累计 ε 会是多少。"""
        shadow = RDPAccountant(self.orders)
        shadow.segments = [RDPSegment(s.sample_rate, s.noise_multiplier, s.steps) for s in self.segments]
        shadow.add_segment(sample_rate, noise_multiplier, steps)
        return shadow.get_epsilon(delta)

    def solve_noise_for_target_epsilon(self,
                                      target_epsilon: float,
                                      sample_rate: float,
                                      steps: int,
                                      delta: float = 1e-5,
                                      min_noise: float = 0.05,
                                      max_noise: float = 50.0,
                                      tol: float = 1e-3,
                                      max_iters: int = 40) -> float:
        """
        给定未来要执行的 (q, steps)，反求使得“累计 ε <= target_epsilon”的最小 σ。

        说明：
        - σ 越大，ε 越小（单调），因此可二分。
        - 该功能用于论文中“在线噪声自适应”接口对齐；默认可不开启。
        """
        target_epsilon = float(target_epsilon)
        if target_epsilon <= 0:
            return max_noise

        # 如果当前已经超预算，直接返回最大噪声（尽量把后续 ε 增量压小）
        cur_eps = self.get_epsilon(delta)
        if cur_eps >= target_epsilon:
            return max_noise

        lo, hi = float(min_noise), float(max_noise)
        # 若 hi 仍无法满足，则返回 hi
        if self.predict_epsilon(sample_rate, hi, steps, delta) > target_epsilon:
            return hi

        for _ in range(max_iters):
            mid = (lo + hi) / 2.0
            eps_mid = self.predict_epsilon(sample_rate, mid, steps, delta)
            if eps_mid <= target_epsilon:
                hi = mid
            else:
                lo = mid
            if (hi - lo) <= tol:
                break
        return hi

    def state_dict(self) -> Dict[str, Any]:
        return {
            'orders': list(self.orders),
            'segments': [(s.sample_rate, s.noise_multiplier, int(s.steps)) for s in self.segments],
        }

    def load_state_dict(self, state: Dict[str, Any]):
        orders = state.get('orders', None)
        if orders is not None:
            self.orders = [float(o) for o in orders]
        self.segments = []
        for q, sigma, steps in state.get('segments', []):
            self.add_segment(float(q), float(sigma), int(steps))





class BEUBudgetManager:
    """
    BEU hardware model: dual-threshold hysteresis control.

    设计目标（与论文 3.3 节/算法 2 对齐）：
      1) 追踪低精度计算释放的“时间盈余”预算（cycles）
      2) 将预算以 token-bucket 形式用于隐藏 DP 算子开销（clipping / noise）
      3) 使用双阈值迟滞控制稳定 Shadow-Execution 开/关

    兼容性说明：
      - 默认 cost_model="legacy"：保持你现有结果/行为不变
      - 可选 cost_model="paper"：按 DP-SGD 语义区分 per-example clipping 与 per-step noise
    """

    def __init__(
        self,
        hw_profile: dict,
        cost_model: str = "legacy",
        max_budget_cycles: float | None = None,
        max_overdraft_cycles: float | None = None,
    ):
        # 当前桶内预算（cycles）
        self.budget_cycles: float = 0.0

        # 统计量（用于论文中的 hidden-ratio / privacy tax 量化）
        self.total_dp_cost_cycles: float = 0.0
        self.total_penalty_cycles: float = 0.0

        # Read per-param security costs from profile
        sec = hw_profile["security_costs"]
        self.clip_cycles_per_param = float(sec["per_example_clipping"]["cycles_per_param"])
        self.noise_cycles_per_param = float(sec["noise_generation"]["cycles_per_param"])

        self.cost_model = str(cost_model).lower()

        # Dual-threshold hysteresis
        # NOTE: 这里保留你原有的阈值规模（以 1e6 参数为参考），避免“乱改”导致行为大幅漂移。
        legacy_cost_per_param = (self.clip_cycles_per_param + self.noise_cycles_per_param)
        base_cost = 1e6 * legacy_cost_per_param
        self.threshold_high = 1.5 * base_cost
        self.threshold_low = 0.5 * base_cost

        # token-bucket 容量上限（Bmax），以及允许的短期透支（FIFO buffer）
        self.max_budget_cycles = float(max_budget_cycles) if max_budget_cycles is not None else None
        self.max_overdraft_cycles = float(max_overdraft_cycles) if max_overdraft_cycles is not None else None

        # Shadow mode state
        self.in_shadow_mode: bool = False

    def reset_stats(self) -> None:
        self.total_dp_cost_cycles = 0.0
        self.total_penalty_cycles = 0.0

    def deposit_budget(self, saved_cycles: float) -> None:
        """Deposit saved cycles into token bucket (with optional capacity clamp)."""
        if not np.isfinite(saved_cycles):
            return
        self.budget_cycles += float(saved_cycles)
        if self.max_budget_cycles is not None:
            self.budget_cycles = min(self.budget_cycles, self.max_budget_cycles)

        # enter shadow mode if budget high enough
        if (not self.in_shadow_mode) and (self.budget_cycles > self.threshold_high):
            self.in_shadow_mode = True

    def deposit(self, saved_cycles: float) -> None:
        self.deposit_budget(saved_cycles)

    def _dp_cost_cycles(self, num_params: int, batch_size: int) -> float:
        """
        DP 算子开销模型（cycles）：
          - legacy: (clip + noise) * batch_size  （保持你原有实现）
          - paper:  clip * batch_size + noise    （更贴近 DP-SGD 语义）
        """
        num_params = float(num_params)
        batch_size = float(batch_size)

        if self.cost_model == "paper":
            return num_params * (self.clip_cycles_per_param * batch_size + self.noise_cycles_per_param)

        # default: legacy
        return num_params * (self.clip_cycles_per_param + self.noise_cycles_per_param) * batch_size

    def check_and_spend(self, num_params: int, batch_size: int) -> float:
        """
        Spend budget for DP operations.

        Returns:
            penalty_cycles: 预算不足导致的“阻塞性”额外开销（cycles）
            - Shadow mode：允许短期透支，penalty=0（开销被隐藏）
            - Blocking mode：预算不足，显式 penalty>0（privacy tax 外显）
        """
        cost = float(self._dp_cost_cycles(num_params, batch_size))
        self.total_dp_cost_cycles += cost

        penalty = 0.0

        if self.budget_cycles >= cost:
            self.budget_cycles -= cost
        else:
            if self.in_shadow_mode:
                # 允许短期透支（模拟 FIFO buffering / shadow execution）
                self.budget_cycles -= cost

                # 可选：限制透支深度，避免预算无限向负无穷漂移（不改变默认行为）
                if self.max_overdraft_cycles is not None:
                    self.budget_cycles = max(self.budget_cycles, -self.max_overdraft_cycles)

                if self.budget_cycles < self.threshold_low:
                    self.in_shadow_mode = False
            else:
                # Blocking mode：无法隐藏 -> 显式 penalty
                penalty = cost - max(0.0, self.budget_cycles)
                self.budget_cycles = 0.0

        self.total_penalty_cycles += penalty
        return float(penalty)

    def get_summary(self) -> Dict[str, Any]:
        """Export BEU state for logging/plotting."""
        hidden_ratio = 0.0
        if self.total_dp_cost_cycles > 0:
            hidden_ratio = float(1.0 - (self.total_penalty_cycles / self.total_dp_cost_cycles))
            hidden_ratio = float(max(0.0, min(1.0, hidden_ratio)))

        return {
            "budget_cycles": float(self.budget_cycles),
            "in_shadow_mode": bool(self.in_shadow_mode),
            "threshold_high": float(self.threshold_high),
            "threshold_low": float(self.threshold_low),
            "total_dp_cost_cycles": float(self.total_dp_cost_cycles),
            "total_penalty_cycles": float(self.total_penalty_cycles),
            "dp_hidden_ratio": hidden_ratio,
            "cost_model": self.cost_model,
        }


class DPSGDOptimizer(Optimizer):
    """
    DP-SGD Wrapper: 注入 (Clipping + Noise) + (SW/HW) overhead 模拟 + (可选)RDP 会计
    dp_mode:
      - "soft": RMS-clip + RMS-scaled noise（更贴近你现在论文里的 SoftDP 叙事，且不会把梯度剪废）
      - "strict": global-norm clip + sigma = nm * C / B（口径更像标准 DP-SGD，但这里仍是近似，因为没做 per-sample）
    """

    def __init__(
            self,
            optimizer,
            model,
            clip_norm,
            noise_multiplier,
            batch_size,
            beu_manager=None,
            dp_mode: str = "soft",
            sample_rate=None,
            accountant=None,
            delta: float = 1e-5,
            noise_generator: Optional[torch.Generator] = None,
    ):
        self.optimizer = optimizer
        self.model = model
        self.clip_norm = float(clip_norm)
        self.noise_multiplier = float(noise_multiplier)
        self.batch_size = int(batch_size)
        self.beu_manager = beu_manager

        # 新增：SoftDP / StrictDP
        self.dp_mode = (dp_mode or "soft").lower()

        # 用于 RDP 会计的采样率（可选）
        self.sample_rate = float(sample_rate) if sample_rate is not None else None

        # --- RDP 会计（跨轮次累计）---
        # accountant 由 FederatedTrainer 按 client_id 维护并传入
        self.accountant = accountant
        self.delta = float(delta)
        self.noise_generator = noise_generator

        # step bookkeeping
        self.steps = 0
        self.step_penalties = []  # 记录每一步的额外延迟

    # 代理标准 Optimizer 接口，避免各种 defaults/param_groups 崩溃
    def zero_grad(self, set_to_none: bool = False):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def __getattr__(self, name):
        return getattr(self.optimizer, name)

    def set_sample_rate(self, sample_rate: float):
        self.sample_rate = float(sample_rate)

    def set_noise_multiplier(self, noise_multiplier: float):
        self.noise_multiplier = float(noise_multiplier)

    @torch.no_grad()
    def _grad_stats(self):
        """返回 (grad_rms, grad_global_norm, total_elems)"""
        sq_sum = None
        count = 0
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.detach()
                if sq_sum is None:
                    sq_sum = g.pow(2).sum()
                else:
                    sq_sum = sq_sum + g.pow(2).sum()
                count += g.numel()

        if sq_sum is None or count <= 0:
            return 0.0, 0.0, 0

        grad_rms = float(torch.sqrt(sq_sum / float(count)))
        grad_norm = float(torch.sqrt(sq_sum))
        return grad_rms, grad_norm, count

    def step(self, closure=None):
        # 1) 模拟 DP 计算开销（硬件/软件）
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        if self.beu_manager:
            latency_penalty = self.beu_manager.check_and_spend(num_params, self.batch_size)
        else:
            # 软件基线开销：与论文式(13)对齐
            # clip: 0.5 cycles/param * batch_size（逐样本裁剪）
            # noise: 0.05 cycles/param（全局一次，不随batch_size增长）
            SW_CLIP_CYCLES_PER_PARAM = 0.5
            SW_NOISE_CYCLES_PER_PARAM = 0.05
            latency_penalty = float(num_params) * (
                    SW_CLIP_CYCLES_PER_PARAM * float(self.batch_size)
                    + SW_NOISE_CYCLES_PER_PARAM
            )
        self.step_penalties.append(latency_penalty)

        # 2) Clipping + Noise
        grad_rms, grad_norm, _ = self._grad_stats()

        # --- Clipping ---
        if self.clip_norm > 0:
            if self.dp_mode == "soft":
                # ✅ SoftDP：RMS clipping（避免“全局 norm=1”把梯度剪废）
                if grad_rms > self.clip_norm:
                    scale = self.clip_norm / (grad_rms + 1e-12)
                    for group in self.optimizer.param_groups:
                        for p in group["params"]:
                            if p.grad is not None:
                                p.grad.mul_(scale)
                    grad_rms = self.clip_norm
            else:
                # strict：保留全局 norm clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)

        # --- Noise ---
        if self.noise_multiplier > 0:
            if self.dp_mode == "soft":
                # ✅ SoftDP：噪声尺度跟随梯度 RMS（SNR≈1/noise_multiplier）
                sigma = self.noise_multiplier * grad_rms
            else:
                # strict-ish：sigma = nm * C / B（因为 pytorch loss 通常是 mean over batch）
                sigma = self.noise_multiplier * self.clip_norm / max(self.batch_size, 1)

            if sigma > 0:
                for group in self.optimizer.param_groups:
                    for p in group["params"]:
                        if p.grad is not None:
                            noise = torch.randn(
                                p.grad.shape,
                                dtype=p.grad.dtype,
                                device=p.grad.device,
                                generator=self.noise_generator,
                            )
                            p.grad.add_(noise * sigma)

        # 3) 真正更新
        loss = self.optimizer.step(closure)
        self.steps += 1

        # 4) RDP accounting（跨轮次累计）
        if self.accountant is not None and self.sample_rate is not None:
            self.accountant.add_segment(self.sample_rate, self.noise_multiplier, steps=1)

        return loss

    def get_total_dp_overhead_ms(self, clock_period_ns):
        return (sum(self.step_penalties) * float(clock_period_ns)) / 1e6

    def get_privacy_spent(self, delta=None) -> float:
        delta_val = self.delta if delta is None else float(delta)

        if self.accountant is not None:
            return float(self.accountant.get_epsilon(delta_val))

        # 兜底：如果没传 accountant，就按当前生命周期 steps 估算
        if not HAS_OPACUS or self.steps <= 0:
            q = float(self.sample_rate) if self.sample_rate is not None else (self.batch_size / 10000.0)
            return float((q * np.sqrt(max(self.steps, 1))) / max(self.noise_multiplier, 1e-6))

        q = float(self.sample_rate) if self.sample_rate is not None else (self.batch_size / 10000.0)
        orders = DEFAULT_RDP_ORDERS
        rdp = rdp_analysis.compute_rdp(
            q=q,
            noise_multiplier=self.noise_multiplier,
            steps=self.steps,
            orders=orders
        )
        eps, _ = rdp_analysis.get_privacy_spent(orders=orders, rdp=rdp, delta=delta_val)
        return float(eps)
