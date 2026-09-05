import numpy as np
from typing import Dict, List, Optional


class ACFScheduler:
    """
    [TCAD Core Component]
    自适应计算流调度器 (Adaptive Compute Flow Scheduler)

    - 支持论文 Eq.(11) 的时空协同概率模型（λ：空间/时间权衡系数）
    - 支持 BEU 预算闭环反馈（预算不足时强制低精度回血）
    - 兼容多种调度模式，用于消融/敏感性实验（Static / Time-Decay / Entropy-Only / Entropy+Time）
    """

    def __init__(
        self,
        total_rounds: int,
        client_stats: List[Dict],
        lamda: float = 0.5,
        budget_safety_threshold: float = 0.0,
        mode: str = "entropy_time",
        low_precision: str = "FP8_E5M2",
        high_precision: str = "BF16",
        deterministic: bool = False,
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        # backward-compat alias (旧版本使用 alpha)
        alpha: Optional[float] = None,
    ):
        """
        Args:
            total_rounds: 总通信轮次 T
            client_stats: 包含 'entropy' 的客户端元数据
            lamda: 时空权重系数 λ（论文符号；注意 python 关键字 lambda，故代码用 lamda）
            budget_safety_threshold: BEU 预算安全阈值 (cycles). 若低于此值，强制低精度回血
            mode: 调度模式
                - "entropy_time": Eq.(11) 时空协同（默认）
                - "entropy_only": 仅空间项（H_k）
                - "time_decay": 仅时间项（t/T）
                - "static_low": 恒定低精度（FP8）
                - "static_high": 恒定高精度（BF16）
            low_precision / high_precision: 低/高精度配置名（需被 HMPEPrecisionEmulator 支持）
            deterministic: 若 True，则用阈值 p_high>=0.5 做确定性决策（降低随机性，便于复现）
            alpha: 兼容旧参数名（若提供则覆盖 lamda）
        """
        self.total_rounds = int(total_rounds)

        # --- backward compatibility ---
        if alpha is not None:
            lamda = float(alpha)

        # clamp
        self.lamda = float(np.clip(lamda, 0.0, 1.0))
        self.budget_threshold = float(budget_safety_threshold)

        self.mode = str(mode).lower()
        self.low_precision = str(low_precision)
        self.high_precision = str(high_precision)
        self.deterministic = bool(deterministic)
        self.rng = rng if rng is not None else np.random.default_rng(seed)

        # 1) 预计算归一化熵 (Spatial Term)
        entropies = [float(stats.get("entropy", 0.5)) for stats in client_stats]
        max_h = max(entropies) if entropies and max(entropies) > 0 else 1.0
        self.normalized_entropies = [h / max_h for h in entropies]

        print(f"ACF Scheduler initialized for {len(client_stats)} clients.")
        print(f"   Max Entropy: {max_h:.4f} | Mode: {self.mode} | λ={self.lamda:.2f} | B_th={self.budget_threshold:.1f} cycles")

    def _p_high(self, client_id: int, current_round: int) -> float:
        """Compute p_high according to selected mode."""
        spatial_term = self.normalized_entropies[client_id]
        temporal_term = min(1.0, (current_round + 1) / max(1, self.total_rounds))

        if self.mode in ["entropy_time", "entropy+time", "eq11", "default"]:
            # Eq.(11)
            return float(self.lamda * spatial_term + (1.0 - self.lamda) * temporal_term)

        if self.mode in ["entropy_only", "entropy"]:
            return float(spatial_term)

        if self.mode in ["time_decay", "time"]:
            return float(temporal_term)

        if self.mode in ["static_low", "static_fp8", "low"]:
            return 0.0

        if self.mode in ["static_high", "static_bf16", "high"]:
            return 1.0

        raise ValueError(f"Unsupported precision scheduling mode: {self.mode}")

    def get_execution_plan(
        self,
        client_id: int,
        current_round: int,
        current_budget: float = float("inf"),
    ) -> Dict[str, str]:
        """
        决策函数：决定当前客户端在本轮的计算精度

        Returns:
            {'compute': <precision>, 'reason': <str>}
        """
        # [Closed-Loop Feedback] 若预算不足，强制低精度回血
        if float(current_budget) < self.budget_threshold:
            return {"compute": self.low_precision, "reason": "Budget_Recovery"}

        p_high = self._p_high(client_id, current_round)

        # 决策：随机采样 or 阈值确定
        if self.deterministic:
            is_high = (p_high >= 0.5)
        else:
            is_high = (self.rng.random() < p_high)

        if is_high:
            return {"compute": self.high_precision, "reason": f"High_Precision (p={p_high:.2f})"}
        return {"compute": self.low_precision, "reason": f"Low_Precision (p={p_high:.2f})"}
