import copy
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, Optional
from torch.amp import autocast, GradScaler  # AMP加速（新API）
import numpy as np
import torch
import torch.nn as nn

from .acf_scheduler import ACFScheduler
from .aggregation import (
    aggregate_fedpaq_deltas,
    batchnorm_state_keys,
    normalize_client_weights,
    weighted_reduce_states,
)
from .dp_sgd import BEUBudgetManager, DPSGDOptimizer, RDPAccountant
from .fedmpq_proxy import FedMPQProxyScheduler
from .sota_adapters import (
    aggregate_fedclam_states,
    aggregate_fedevi_states,
    calculate_overfitting_penalty,
    calculate_vlr,
)

from simulator.acf_simulator import ACFSimulator
from utils.reproducibility import derive_seed, make_torch_generator

try:
    from utils.layer_profiler import analyze_model_workload
except ImportError:
    analyze_model_workload = None


def json_default(o):
    # NumPy scalar -> Python scalar
    if isinstance(o, (np.integer, np.int32, np.int64)):
        return int(o)
    if isinstance(o, (np.floating, np.float32, np.float64)):
        return float(o)
    # NumPy array -> list
    if isinstance(o, np.ndarray):
        return o.tolist()
    # Torch tensor -> list/scalar
    if isinstance(o, torch.Tensor):
        if o.dim() == 0:
            return o.item()
        return o.detach().cpu().tolist()
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


class FederatedTrainer:
    """
    HMPE-ACF Federated Trainer (System + Algorithm)

    Key features kept:
      1) FedBN: decouple BN states
      2) FedPAQ: local periodic SGD + quantized model increments
      3) Hardware baselines: BitFusion(INT8), Mao(BF16) via precision emulator
      4) HMPE-ACF: mixed-precision + BEU masking of DP overhead (when DP enabled)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer_fn,
        criterion: nn.Module,
        device: str = "cuda",
        dp_config: Optional[Dict] = None,
        acf_policy: Optional[Dict] = None,
        hw_profile_path: Optional[str] = None,
        output_dir: str = "results",
        comm_interval: int = 1,
        run_seed: int = 0,
        client_schedule=None,
        use_amp: bool = False,
    ):
        self.device = device
        self.global_model = model.to(self.device)
        self.optimizer_fn = optimizer_fn
        self.criterion = criterion

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.dp_config = dp_config or {"enable": False}
        self.enable_dp = bool(self.dp_config.get("enable", False))
        # 是否启用 PEC（聚合协处理器）。
        # 兼容旧版本：若没有 simulate_hardware_pec，则沿用 simulate_hardware_beu 的行为。
        self.enable_pec = bool(self.dp_config.get("simulate_hardware_pec", self.dp_config.get("simulate_hardware_beu", False)))

        # cache: (precision, input_shape) -> dict{cycles_fp32, cycles_actual, energy..., surplus_cycles}
        self.acf_policy = acf_policy or {"compute": "FP32", "strategy": "FedAvg"}
        self.strategy = str(self.acf_policy.get("strategy", "FedAvg"))
        self.comm_interval = int(comm_interval)
        if self.comm_interval <= 0:
            raise ValueError("comm_interval must be positive")
        self.is_fedbn = self.strategy == "FedBN"
        self.is_fedpaq = self.strategy == "FedPAQ"
        self.is_fedmpq_proxy = self.strategy == "FedMPQProxy"
        self.is_fedevi = self.strategy == "FedEvi"
        self.is_fedclam = self.strategy == "FedCLAM"
        self.fedclam_momentum = {}
        if self.comm_interval != 1:
            raise ValueError(
                "Every recorded round is a server communication round; "
                "FedPAQ periodicity is expressed by local_update_steps"
            )
        self.fedpaq_levels = int(self.acf_policy.get("quantization_levels", 255))
        if self.is_fedpaq and self.fedpaq_levels <= 0:
            raise ValueError("FedPAQ quantization_levels must be positive")
        self.fedpaq_local_update_steps = int(
            self.acf_policy.get("local_update_steps", 5)
        )
        if self.is_fedpaq and self.fedpaq_local_update_steps <= 0:
            raise ValueError("FedPAQ local_update_steps must be positive")
        self.fedmpq_client_budgets = [
            int(value)
            for value in self.acf_policy.get("client_budgets", [])
        ]
        self.fedmpq_group_lasso_lambda = float(
            self.acf_policy.get("group_lasso_lambda", 0.01)
        )
        self.fedmpq_pruning_threshold = float(
            self.acf_policy.get("pruning_threshold", 0.03)
        )
        self.fedmpq_proxy_scheduler = None
        self.run_seed = int(run_seed)
        self.client_schedule = client_schedule
        self.use_amp = bool(use_amp)

        self.simulator = ACFSimulator(hw_profile_path)
        self.acf_scheduler: Optional[ACFScheduler] = None

        self.beu_managers = {}
        # RDP accountant: per-client, across rounds
        self.privacy_accountants: Dict[int, RDPAccountant] = {}
        self.client_bn_states = {}
        self._trace_cache = {}
        self.dp_noise_generators = {}
        self.quantization_generators = {}
        self.communication_quantization_generators = {}
        self.bn_state_keys = batchnorm_state_keys(self.global_model)

        self.best_val = 0.0
        self.best_state = None
        self.best_client_bn_states = None
    # AMP GradScler：每个scenario独立一个scaler
        self.scaler = GradScaler(
            'cuda',
            enabled=self.use_amp and torch.cuda.is_available(),
        )

    def _get_stream_generator(self, store: dict, stream: str, client_id: int):
        if client_id not in store:
            generator_device = (
                "cuda"
                if str(self.device).startswith("cuda") and torch.cuda.is_available()
                else "cpu"
            )
            store[client_id] = make_torch_generator(
                derive_seed(self.run_seed, stream, int(client_id)),
                device=generator_device,
            )
        return store[client_id]
    # --------------------------------------------------------------------------
    # Hardware surplus estimate (FP32 - actual precision)
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # Hardware step profile (FP32 vs target precision)
    # --------------------------------------------------------------------------

    def _cycles_to_ms(self, cycles: float) -> float:
        """Convert cycles to milliseconds using loaded hardware profile."""
        try:
            return (float(cycles) * float(self.simulator.clock_period_ns)) / 1e6
        except Exception:
            return 0.0

    def _get_hw_step_profile(self, model, input_shape, precision: str) -> Dict[str, float]:
        """
        Return a cached HW cost profile for one training step under a given precision.
        This does NOT change training behavior; it only produces latency/energy estimates
        for paper-quality reporting (TCAD).

        Returns keys:
          - cycles_fp32, cycles_actual, surplus_cycles
          - energy_fp32_mJ, energy_actual_mJ
        """
        cache_key = (precision, str(tuple(input_shape)))
        cached = self._trace_cache.get(cache_key, None)
        if isinstance(cached, dict):
            return cached

        profile = {
            "cycles_fp32": 0.0,
            "cycles_actual": 0.0,
            "energy_fp32_mJ": 0.0,
            "energy_actual_mJ": 0.0,
            "surplus_cycles": 0.0,
        }

        if analyze_model_workload is not None:
            workload = analyze_model_workload(model, input_shape)
            if workload:
                res_fp32 = self.simulator.simulate_model_training(workload, {"compute": "FP32"}, enable_beu=False)
                res_actual = self.simulator.simulate_model_training(workload, {"compute": precision}, enable_beu=False)

                try:
                    profile["cycles_fp32"] = float(res_fp32.get("cycles", 0.0))
                    profile["cycles_actual"] = float(res_actual.get("cycles", 0.0))
                    profile["energy_fp32_mJ"] = float(res_fp32.get("total_energy_mJ", 0.0))
                    profile["energy_actual_mJ"] = float(res_actual.get("total_energy_mJ", 0.0))
                except Exception:
                    pass

                profile["surplus_cycles"] = max(0.0, profile["cycles_fp32"] - profile["cycles_actual"])

        # cache
        self._trace_cache[cache_key] = profile
        return profile

    def _get_hw_surplus(self, model, input_shape, precision: str) -> float:
        """Backward-compatible wrapper: only return surplus cycles."""
        prof = self._get_hw_step_profile(model, input_shape, precision)
        return float(prof.get("surplus_cycles", 0.0))

    # --------------------------------------------------------------------------
    # Dice metric (BraTS: WT/TC/ET)
    # --------------------------------------------------------------------------

    def calculate_brats_dice(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """
        BraTS Dice metric:
          - WT = {1,2,3}
          - TC = {1,3}
          - ET = {3}
        We skip regions where both pred & target are empty.
        """
        # ---- hard assertions to avoid silent metric bugs ----
        assert pred.shape == target.shape, (
            f"[DiceAssert] pred/target shape mismatch: pred={tuple(pred.shape)} vs target={tuple(target.shape)}"
        )
        assert pred.ndim == 3, (
            f"[DiceAssert] expect per-sample 3D tensor (D,H,W), got pred.ndim={pred.ndim}"
        )
        assert target.ndim == 3, (
            f"[DiceAssert] expect per-sample 3D tensor (D,H,W), got target.ndim={target.ndim}"
        )
        # Values sanity (BraTS preprocessed should be 0..3)
        pmin, pmax = int(pred.min().item()), int(pred.max().item())
        tmin, tmax = int(target.min().item()), int(target.max().item())
        assert 0 <= pmin and pmax <= 3, f"[DiceAssert] pred out of range: min={pmin}, max={pmax} (expect 0..3)"
        assert 0 <= tmin and tmax <= 3, f"[DiceAssert] target out of range: min={tmin}, max={tmax} (expect 0..3)"

        smooth = 1e-5
        res = {}

        def dice_coef(p: torch.Tensor, t: torch.Tensor):
            inter = (p * t).sum()
            union = p.sum() + t.sum()
            if union.item() < 1:
                return None
            return ((2.0 * inter + smooth) / (union + smooth)).item()

        p_1 = (pred == 1).float()
        t_1 = (target == 1).float()
        p_2 = (pred == 2).float()
        t_2 = (target == 2).float()
        p_3 = (pred == 3).float()
        t_3 = (target == 3).float()

        d_wt = dice_coef(p_1 + p_2 + p_3, t_1 + t_2 + t_3)
        if d_wt is not None:
            res["WT"] = d_wt
        d_tc = dice_coef(p_1 + p_3, t_1 + t_3)
        if d_tc is not None:
            res["TC"] = d_tc
        d_et = dice_coef(p_3, t_3)
        if d_et is not None:
            res["ET"] = d_et

        return res

    # --------------------------------------------------------------------------
    # Evaluation (val/test)
    # --------------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, loader, model: Optional[nn.Module] = None) -> Dict[str, float]:
        evaluation_model = self.global_model if model is None else model
        evaluation_model.eval()

        # Always evaluate in FP32 (disable quantization hooks)
        if hasattr(evaluation_model, "update_policy"):
            evaluation_model.update_policy("FP32")

        metrics_sum = {"WT": [], "TC": [], "ET": []}

        for data, target in loader:
            data = data.to(self.device)
            target = target.to(self.device)

            output = evaluation_model(data)
            pred = torch.argmax(output, dim=1)

            # Only convert target if it is one-hot with class dimension
            # (B, C, D, H, W) -> (B, D, H, W)
            if target.ndim == output.ndim:
                target = torch.argmax(target, dim=1)

            # pred/target should both be (B, D, H, W)
            assert pred.shape == target.shape, (
                f"[EvalAssert] pred/target mismatch: pred={tuple(pred.shape)} target={tuple(target.shape)}"
            )

            for i in range(pred.shape[0]):
                scores = self.calculate_brats_dice(pred[i], target[i])
                for k, v in scores.items():
                    metrics_sum[k].append(v)

        avg_metrics = {k: float(np.mean(v) * 100.0) for k, v in metrics_sum.items() if len(v) > 0}
        avg_metrics["Mean"] = float(np.mean(list(avg_metrics.values()))) if avg_metrics else 0.0
        return avg_metrics

    def _initialize_fedbn_states(self, num_clients: int) -> None:
        if not self.is_fedbn:
            return
        global_state = self.global_model.state_dict()
        for client_id in range(int(num_clients)):
            self.client_bn_states[client_id] = {
                key: global_state[key].detach().cpu().clone()
                for key in self.bn_state_keys
            }

    def _build_fedbn_evaluation_model(
        self,
        client_num_samples,
    ) -> nn.Module:
        client_ids = sorted(self.client_bn_states)
        weights = normalize_client_weights(client_ids, client_num_samples)
        reduced_bn = weighted_reduce_states(
            [self.client_bn_states[client_id] for client_id in client_ids],
            weights,
        )
        evaluation_model = copy.deepcopy(self.global_model)
        evaluation_state = evaluation_model.state_dict()
        for key, value in reduced_bn.items():
            evaluation_state[key] = value.to(
                device=evaluation_state[key].device,
                dtype=evaluation_state[key].dtype,
            )
        evaluation_model.load_state_dict(evaluation_state, strict=True)
        return evaluation_model

    @torch.no_grad()
    def _evaluate_fedbn_personalized(
        self,
        client_eval_loaders,
    ) -> Dict[str, float]:
        metrics_sum = {"WT": [], "TC": [], "ET": []}
        evaluation_model = copy.deepcopy(self.global_model)
        for client_id, loader in enumerate(client_eval_loaders):
            if len(loader.dataset) == 0:
                continue
            evaluation_model.load_state_dict(
                self.global_model.state_dict(),
                strict=True,
            )
            evaluation_model.load_state_dict(
                self.client_bn_states[client_id],
                strict=False,
            )
            evaluation_model.eval()
            if hasattr(evaluation_model, "update_policy"):
                evaluation_model.update_policy("FP32")

            for data, target in loader:
                data = data.to(self.device)
                target = target.to(self.device)
                output = evaluation_model(data)
                pred = torch.argmax(output, dim=1)
                if target.ndim == output.ndim:
                    target = torch.argmax(target, dim=1)
                if pred.shape != target.shape:
                    raise AssertionError(
                        "Personalized FedBN evaluation shape mismatch"
                    )
                for sample_index in range(pred.shape[0]):
                    scores = self.calculate_brats_dice(
                        pred[sample_index],
                        target[sample_index],
                    )
                    for key, value in scores.items():
                        metrics_sum[key].append(value)
        del evaluation_model

        avg_metrics = {
            key: float(np.mean(values) * 100.0)
            for key, values in metrics_sum.items()
            if values
        }
        avg_metrics["Mean"] = (
            float(np.mean(list(avg_metrics.values())))
            if avg_metrics
            else 0.0
        )
        return avg_metrics

    def evaluate_for_strategy(
        self,
        loader,
        client_num_samples,
        client_eval_loaders=None,
    ) -> Dict[str, float]:
        if not self.is_fedbn:
            return self.evaluate(loader)
        if client_eval_loaders is not None:
            return self._evaluate_fedbn_personalized(client_eval_loaders)
        evaluation_model = self._build_fedbn_evaluation_model(
            client_num_samples
        )
        try:
            return self.evaluate(loader, model=evaluation_model)
        finally:
            del evaluation_model

    @torch.no_grad()
    def _mean_loss_on_loader(self, model, loader, round_idx: int) -> float:
        was_training = model.training
        model.eval()
        losses = []
        for data, target in loader:
            data = data.to(self.device)
            target = target.to(self.device)
            output = model(data)
            if target.ndim == output.ndim:
                target = target.argmax(dim=1)
            if self.is_fedevi:
                loss = self.criterion(output, target, round_idx + 1)
            elif self.is_fedclam:
                loss = self.criterion(output, target, data, round_idx)
            else:
                loss = self.criterion(output, target)
            losses.append(float(loss.item()))
        model.train(was_training)
        if not losses:
            raise ValueError("Client validation loader is empty")
        return float(np.mean(losses))

    # --------------------------------------------------------------------------
    # Client update
    # --------------------------------------------------------------------------

    def client_update(
        self,
        client_id,
        train_loader,
        epochs: int,
        precision: str,
        round_idx: int,
        max_steps: Optional[int] = None,
        eval_loader=None,
    ):
        # (1) init client model
        client_model = copy.deepcopy(self.global_model)
        if hasattr(client_model, "set_quantization_generator"):
            client_model.set_quantization_generator(
                self._get_stream_generator(
                    self.quantization_generators,
                    "quantization",
                    int(client_id),
                )
            )

        # FedBN restores private BN state after loading shared parameters.
        if self.is_fedbn and client_id in self.client_bn_states:
            client_model.load_state_dict(
                self.client_bn_states[client_id],
                strict=False,
            )

        client_model.train()
        if hasattr(client_model, "update_policy"):
            client_model.update_policy(precision)

        # (2) BEU manager
        if client_id not in self.beu_managers:
            # [P0] 仅增强统计/对齐公式，不改变默认行为（cost_model 默认 legacy）
            self.beu_managers[client_id] = BEUBudgetManager(
                self.simulator.hw_profile,
                cost_model=str(self.dp_config.get("dp_cost_model", "legacy")),
                max_budget_cycles=self.dp_config.get("beu_max_budget_cycles", None),
                max_overdraft_cycles=self.dp_config.get("beu_max_overdraft_cycles", None),
            )

        use_beu = self.enable_dp and bool(self.dp_config.get("simulate_hardware_beu", False))
        beu_mgr = self.beu_managers[client_id] if use_beu else None
        if beu_mgr is not None and hasattr(beu_mgr, "reset_stats"):
            # 每轮重置统计量，但不重置预算桶（预算需要跨轮累计）
            beu_mgr.reset_stats()

        initial_validation_loss = None
        if self.is_fedclam:
            if eval_loader is None:
                raise ValueError("FedCLAM requires a client validation loader")
            initial_validation_loss = self._mean_loss_on_loader(
                client_model, eval_loader, round_idx
            )

        # (3) optimizer
        base_opt = self.optimizer_fn(client_model.parameters())

        if self.enable_dp:
            # Important for RDP q estimate
            dataset_size = None
            try:
                dataset_size = len(train_loader.dataset)
            except Exception:
                dataset_size = None

            # --- DP wiring: define dp_delta / sample_rate / accountant / noise_mul ---

            # (a) delta
            dp_delta = float(self.dp_config.get("delta", 1e-5))

            # (b) sample rate q ~= batch_size / local_dataset_size
            sample_rate = None
            try:
                local_n = len(train_loader.dataset)
            except Exception:
                local_n = None

            if local_n is not None and local_n > 0:
                bs = int(self.dp_config.get("batch_size", 2))
                sample_rate = min(1.0, bs / float(local_n))

            # (c) accountant (per-client, across rounds)
            accountant = None
            if bool(self.dp_config.get("enable_accounting", True)):
                if client_id not in self.privacy_accountants:
                    self.privacy_accountants[client_id] = RDPAccountant()
                accountant = self.privacy_accountants[client_id]

            # (d) noise multiplier (optionally adapt via accountant)
            noise_mul = float(self.dp_config.get("noise_multiplier", 1.0))

            # Optional: online noise adaptation (only meaningful if accountant & epsilon_total are set)
            if accountant is not None and sample_rate is not None and bool(self.dp_config.get("adaptive_noise", False)):
                eps_total = self.dp_config.get("epsilon_total", None)
                if eps_total is not None:
                    # estimate local steps in this round
                    est_steps = int(max(1, len(train_loader) * int(epochs)))
                    if max_steps is not None:
                        est_steps = min(est_steps, int(max_steps))
                    noise_mul = accountant.solve_noise_for_target_epsilon(
                        target_epsilon=float(eps_total),
                        sample_rate=float(sample_rate),
                        steps=est_steps,
                        delta=float(dp_delta),
                        min_noise=max(0.05, noise_mul),
                        max_noise=float(self.dp_config.get("max_noise_multiplier", 50.0)),
                    )


            optimizer = DPSGDOptimizer(
                base_opt,
                client_model,
                self.dp_config.get("clip_norm", 1.0),
                noise_mul,  # ✅ 用自适应后的 noise_mul（否则 adaptive_noise 永远不生效）
                self.dp_config.get("batch_size", 2),
                beu_manager=beu_mgr,
                dp_mode=self.dp_config.get("dp_mode", "soft"),
                sample_rate=sample_rate,
                accountant=accountant,  # ✅ 真正接会计器
                delta=dp_delta,  # ✅ 真正传 delta
                noise_generator=self._get_stream_generator(
                    self.dp_noise_generators,
                    "dp_noise",
                    int(client_id),
                ),
            )
        else:
            optimizer = base_opt

        # (4) train
        losses = []
        batch_surplus = 0.0

        # Probe input shape once for hardware surplus cache
        try:
            s_batch = next(iter(train_loader))[0]
            in_shape = s_batch.shape
        except Exception:
            in_shape = None


        # [P0] 估算每 step 的硬件 cycles / energy（仅用于论文指标，不影响训练）
        step_prof = None
        cycles_per_step = 0.0
        energy_per_step_mJ = 0.0
        if in_shape is not None:
            profile_precision = precision
            if (
                isinstance(precision, dict)
                and hasattr(client_model, "profile_precision")
            ):
                profile_precision = client_model.profile_precision()
            step_prof = self._get_hw_step_profile(
                client_model,
                in_shape,
                profile_precision,
            )
            try:
                batch_surplus = float(step_prof.get("surplus_cycles", 0.0))
                cycles_per_step = float(step_prof.get("cycles_actual", 0.0))
                energy_per_step_mJ = float(step_prof.get("energy_actual_mJ", 0.0))
            except Exception:
                pass

        num_steps = 0
        num_examples_processed = 0
        privacy_events = []
        stop_training = False
        for _ in range(int(epochs)):
            for data, target in train_loader:
                data = data.to(self.device)
                target = target.to(self.device)

                num_steps += 1
                current_batch_size = int(data.shape[0])
                num_examples_processed += current_batch_size
                if self.enable_dp:
                    optimizer.batch_size = current_batch_size
                    if local_n is not None and local_n > 0:
                        optimizer.set_sample_rate(
                            min(1.0, current_batch_size / float(local_n))
                        )

                if beu_mgr is not None and in_shape is not None:
                    beu_mgr.deposit(batch_surplus)

                optimizer.zero_grad()

                # AMP autocast：前向用FP16，反向自动缩放
                with autocast(
                    'cuda',
                    enabled=self.use_amp and torch.cuda.is_available(),
                ):
                    output = client_model(data)

                    # if target is one-hot:
                    if target.ndim == output.ndim:
                        target = torch.argmax(target, dim=1)

                    if self.is_fedevi:
                        loss = self.criterion(output, target, round_idx + 1)
                    elif self.is_fedclam:
                        loss = self.criterion(output, target, data, round_idx)
                    else:
                        loss = self.criterion(output, target)
                    if self.is_fedmpq_proxy:
                        loss = loss + (
                            self.fedmpq_group_lasso_lambda
                            * client_model.bit_group_lasso()
                        )

                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite loss for client {client_id}, "
                        f"round {round_idx}, precision {precision}"
                    )

                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
                losses.append(float(loss.item()))
                if self.enable_dp:
                    privacy_events.append({
                        "sample_rate": float(optimizer.sample_rate),
                        "noise_multiplier": float(optimizer.noise_multiplier),
                    })
                if max_steps is not None and num_steps >= int(max_steps):
                    stop_training = True
                    break
            if stop_training:
                break

        assigned_bit_widths = None
        pruned_bit_widths = None
        bit_width_delta = None
        if self.is_fedmpq_proxy:
            assigned_bit_widths = {
                name: int(str(value).upper().replace("INT", ""))
                for name, value in precision.items()
            }
            pruned_bit_widths = client_model.prune_msb(
                self.fedmpq_pruning_threshold
            )
            bit_width_delta = {
                name: (
                    int(assigned_bit_widths[name])
                    - int(pruned_bit_widths[name])
                )
                for name in assigned_bit_widths
            }

        # (5) save client states
        if self.is_fedbn:
            self.client_bn_states[client_id] = {
                k: v.cpu().clone()
                for k, v in client_model.state_dict().items()
                if k in self.bn_state_keys
            }

        # (6) metrics —— 必须在 comm_interval 块外，所有场景都要执行
        compute_latency_ms = float(num_steps) * self._cycles_to_ms(cycles_per_step)
        energy_mJ = float(num_steps) * float(energy_per_step_mJ)

        metrics = {
            "loss": float(np.mean(losses) if losses else 0.0),
            "compute_latency_ms": float(compute_latency_ms),
            "energy_mJ": float(energy_mJ),
            "precision": str(precision),
            "num_steps": int(num_steps),
            "num_examples_processed": int(num_examples_processed),
            "privacy_events": privacy_events,
        }
        if self.is_fedclam:
            trained_validation_loss = self._mean_loss_on_loader(
                client_model, eval_loader, round_idx
            )
            metrics.update({
                "initial_validation_loss": float(initial_validation_loss),
                "trained_validation_loss": float(trained_validation_loss),
                "val_loss_ratio": calculate_vlr(
                    initial_validation_loss,
                    trained_validation_loss,
                    float(self.acf_policy.get("beta", 1.0)),
                ),
                "overfitting_penalty": calculate_overfitting_penalty(
                    metrics["loss"],
                    trained_validation_loss,
                    float(self.acf_policy.get("alpha", 1.0)),
                ),
            })
        if self.is_fedmpq_proxy:
            metrics["average_bit_width"] = float(
                client_model.average_bit_width()
            )
            metrics["fedmpq_assigned_bit_widths"] = assigned_bit_widths
            metrics["fedmpq_pruned_bit_widths"] = pruned_bit_widths
            metrics["fedmpq_bit_width_delta"] = bit_width_delta

        # ── 论文命题支撑：计算 ΔC(p) 与 Cpriv 的比值，验证 HRpriv=1.0 主张 ──
        delta_c_cycles = float(step_prof.get("surplus_cycles", 0.0)) * num_steps if step_prof else 0.0
        sec_profile = self.simulator.hw_profile.get("security_costs", {})
        clip_cpp = float(sec_profile.get("per_example_clipping", {}).get("cycles_per_param", 0.5))
        noise_cpp = float(sec_profile.get("noise_generation", {}).get("cycles_per_param", 0.05))
        num_params_est = sum(p.numel() for p in client_model.parameters() if p.requires_grad)
        c_priv_cycles = num_params_est * (
            clip_cpp * float(num_examples_processed)
            + noise_cpp * float(num_steps)
        )
        hr_theoretical = float(min(1.0, delta_c_cycles / max(c_priv_cycles, 1.0)))

        if self.enable_dp:
            dp_overhead_ms = float(optimizer.get_total_dp_overhead_ms(self.simulator.clock_period_ns))
            dp_total_ms = dp_overhead_ms
            dp_hidden_ratio = 0.0
            budget_cycles = 0.0

            if beu_mgr is not None:
                summ = beu_mgr.get_summary()
                budget_cycles = float(summ.get("budget_cycles", 0.0))
                dp_hidden_ratio = float(summ.get("dp_hidden_ratio", 0.0))
                dp_total_cycles = float(summ.get("total_dp_cost_cycles", 0.0))
                dp_total_ms = float(self._cycles_to_ms(dp_total_cycles))

            metrics.update({
                "epsilon": float(optimizer.get_privacy_spent(delta=dp_delta)),
                "dp_overhead_ms": float(dp_overhead_ms),
                "dp_total_ms": float(dp_total_ms),
                "dp_hidden_ratio": float(dp_hidden_ratio),
                "budget_cycles": float(budget_cycles),
                "budget_utilization": float(dp_hidden_ratio),
                "delta_c_cycles": float(delta_c_cycles),
                "c_priv_cycles": float(c_priv_cycles),
                "hr_theoretical": float(hr_theoretical),
            })
        else:
            metrics.update({
                "epsilon": 0.0,
                "dp_overhead_ms": 0.0,
                "dp_total_ms": 0.0,
                "dp_hidden_ratio": 0.0,
                "budget_cycles": 0.0,
                "budget_utilization": 0.0,
                "delta_c_cycles": float(delta_c_cycles),
                "c_priv_cycles": float(c_priv_cycles),
                "hr_theoretical": float(hr_theoretical),
            })

        return client_model.state_dict(), metrics

    # --------------------------------------------------------------------------
    # Run training
    # --------------------------------------------------------------------------

    def run(self, client_loaders, val_loader, test_loader, client_stats, rounds: int, local_epochs: int = 1):
        client_num_samples = [
            int(stats.get("num_samples", 0))
            for stats in client_stats
        ]
        if len(client_num_samples) != len(client_loaders):
            raise ValueError("client_stats and client_loaders must have equal length")
        if self.is_fedmpq_proxy:
            if len(self.fedmpq_client_budgets) != len(client_loaders):
                raise ValueError(
                    "FedMPQ proxy requires one bit budget per client"
                )
            self.fedmpq_proxy_scheduler = FedMPQProxyScheduler(
                self.global_model.layer_parameter_counts(),
                self.fedmpq_client_budgets,
            )
        self._initialize_fedbn_states(len(client_loaders))
        fedbn_val_loaders = getattr(
            client_stats,
            "client_val_loaders",
            None,
        )
        fedbn_test_loaders = getattr(
            client_stats,
            "client_test_loaders",
            None,
        )
        fedbn_evaluation_mode = None
        if self.is_fedbn:
            fedbn_evaluation_mode = (
                "personalized local BN on disjoint simulated-client holdout "
                "partitions"
                if fedbn_val_loaders is not None
                else (
                    "sample-weighted BN snapshot for central holdout evaluation "
                    "only; the snapshot is not distributed back to clients"
                )
            )

        if self.acf_scheduler is None:
            # [P0] ACF 超参从 acf_policy 透传（用于消融/敏感性实验）
            lamda = float(self.acf_policy.get("lamda", self.acf_policy.get("alpha", 0.5)))
            budget_th = float(self.acf_policy.get("budget_threshold", 0.0))
            mode = str(self.acf_policy.get("mode", "entropy_time"))
            deterministic = bool(self.acf_policy.get("deterministic", False))
            # 若用户仍用旧字段 alpha，这里不做破坏式改动：lamda 会被 alpha 覆盖
            self.acf_scheduler = ACFScheduler(
                total_rounds=rounds,
                client_stats=client_stats,
                lamda=lamda,
                budget_safety_threshold=budget_th,
                mode=mode,
                low_precision=str(
                    self.acf_policy.get("low_precision", "FP8_E5M2")
                ),
                high_precision=str(
                    self.acf_policy.get("high_precision", "BF16")
                ),
                deterministic=deterministic,
                seed=derive_seed(self.run_seed, "acf_scheduler"),
                alpha=self.acf_policy.get("alpha", None),
            )

        history = {
            "client_num_samples": client_num_samples,
            "local_epochs": int(local_epochs),
            "client_entropies": [
                float(stats.get("entropy", 0.0))
                for stats in client_stats
            ],
            "acf_policy": {
                "compute": self.acf_policy.get("compute"),
                "strategy": self.acf_policy.get("strategy"),
                "mode": self.acf_scheduler.mode,
                "lamda": float(self.acf_scheduler.lamda),
                "budget_threshold": float(
                    self.acf_scheduler.budget_threshold
                ),
                "deterministic_decision": bool(
                    self.acf_scheduler.deterministic
                ),
                "low_precision": self.acf_scheduler.low_precision,
                "high_precision": self.acf_scheduler.high_precision,
                "scheduler_seed": int(
                    derive_seed(self.run_seed, "acf_scheduler")
                ),
            },
            "round": [],
            "train_loss": [],
            "val_dice": [],
            "epsilon": [],
            "epsilon_participating_max": [],
            "epsilon_per_client": [],
            "scheduled_clients": [],
            "participating_clients": [],
            "participating_client_num_samples": [],
            "client_optimizer_steps": [],
            "client_examples_processed": [],
            "client_privacy_events": [],
            "client_precisions": [],
            "is_aggregation_round": [],
            "aggregation_cohort": [],
            "aggregation_weights": [],
            "aggregation_weight_sum": [],
            "aggregation_method": [],
            "aggregation_excluded_key_count": [],
            "fedpaq_quantization": [],

            # system-level latency breakdown (ms)
            "client_latency_ms": [],  # max_k (compute + dp_overhead)
            "compute_latency_ms": [],  # compute of the round straggler
            "dp_overhead_ms": [],      # uncovered DP of the same straggler
            "dp_total_ms": [],
            "dp_background_ms": [],
            "latency_ms": [],          # client + comm + agg (+misc)
            "agg_latency_ms": [],
            "comm_latency_ms": [],
            "misc_latency_ms": [],
            "latency_residual_ms": [],
            "straggler_client_id": [],

            # energy (mJ) & DP masking ratio
            "energy_mJ": [],
            "local_training_energy_mJ": [],
            "softdp_energy_mJ": [],
            "beu_aux_energy_mJ": [],
            "communication_energy_mJ": [],
            "server_aggregation_energy_mJ": [],
            "energy_scope": (
                "modeled local-training compute/memory energy; "
                "SoftDP, BEU auxiliary, communication, and server energy excluded"
            ),
            "dp_hidden_ratio": [],
            "budget_utilization": [],  # backward-compat alias
            # ΔC/Cpriv 理论隐藏比率追踪（论文命题验证）
            "hr_theoretical": [],
            "delta_c_cycles": [],
            "c_priv_cycles": [],
            "val_dice_detailed": [],
        }
        history["aggregation_config"] = {
            "weight_basis": "participating-client local training-set sample count",
            "strategy": self.strategy,
            "batchnorm_private_key_count": int(len(self.bn_state_keys)),
            "fedbn_evaluation_mode": (
                fedbn_evaluation_mode
            ),
            "fedpaq_local_update_steps": (
                int(self.fedpaq_local_update_steps)
                if self.is_fedpaq
                else None
            ),
            "fedpaq_quantizer": (
                "stochastic unbiased QSGD-style L2 quantizer"
                if self.is_fedpaq
                else None
            ),
            "fedpaq_quantization_levels": (
                int(self.fedpaq_levels) if self.is_fedpaq else None
            ),
        }
        history["energy_component_status"] = {
            "local_training_compute_memory": "modeled",
            "softdp_operator": "not modeled",
            "beu_auxiliary": "not modeled",
            "communication": "not modeled",
            "server_aggregation": "not modeled",
        }

        print(
            f"\nStart | Strategy: {self.acf_policy['strategy']} | "
            f"FedPAQ local steps: "
            f"{self.fedpaq_local_update_steps if self.is_fedpaq else 'n/a'}"
        )
        print(f"{'Rnd':>3} | {'Loss':>7} | {'Val Dice':>8} | {'Overhead':>8} | {'Latency':>8}")
        print("-" * 65)

        for r in range(int(rounds)):
            num_clients = len(client_loaders)
            if self.client_schedule is None:
                raise ValueError("A fixed client_schedule is required for reproducible runs")
            if r >= len(self.client_schedule):
                raise ValueError("client_schedule is shorter than the requested rounds")
            scheduled_cids = [int(cid) for cid in self.client_schedule[r]]
            cids = scheduled_cids
            is_agg = True
            if not cids or any(cid < 0 or cid >= num_clients for cid in cids):
                raise ValueError(f"Invalid client schedule at round {r}: {cids}")

            states, metrics = [], []
            round_precisions = []

            for cid in cids:
                budget = self.beu_managers[cid].budget_cycles if cid in self.beu_managers else 0.0
                policy = self.acf_policy["compute"]
                if self.is_fedmpq_proxy:
                    prec = self.fedmpq_proxy_scheduler.policy_for(cid)
                elif policy == "Mixed":
                    prec = self.acf_scheduler.get_execution_plan(cid, r, budget)["compute"]
                else:
                    prec = policy
                round_precisions.append(str(prec))

                s, m = self.client_update(
                    cid,
                    client_loaders[cid],
                    local_epochs,
                    prec,
                    round_idx=r,
                    max_steps=(
                        self.fedpaq_local_update_steps
                        if self.is_fedpaq
                        else None
                    ),
                    eval_loader=(
                        fedbn_val_loaders[cid]
                        if (self.is_fedclam and fedbn_val_loaders is not None)
                        else None
                    ),
                )
                states.append(s)
                metrics.append(m)

            agg_lat = 0.0
            comm_lat = 0.0
            aggregation_weights = []
            aggregation_method = None
            aggregation_excluded_key_count = 0
            fedpaq_quantization = []

            if is_agg:
                if self.is_fedclam:
                    aggregation_weights = [1.0 / len(cids)] * len(cids)
                elif self.is_fedmpq_proxy:
                    raw_weights = [
                        float(client_num_samples[cid])
                        * float(self.fedmpq_client_budgets[cid])
                        for cid in cids
                    ]
                    denominator = float(sum(raw_weights))
                    aggregation_weights = [
                        value / denominator for value in raw_weights
                    ]
                else:
                    aggregation_weights = normalize_client_weights(
                        cids,
                        client_num_samples,
                    )
                if self.is_fedpaq:
                    round_base_state = copy.deepcopy(
                        self.global_model.state_dict()
                    )
                    generators = [
                        self._get_stream_generator(
                            self.communication_quantization_generators,
                            "fedpaq_communication_quantization",
                            int(cid),
                        )
                        for cid in cids
                    ]
                    avg_state, quantization_stats = aggregate_fedpaq_deltas(
                        round_base_state,
                        states,
                        aggregation_weights,
                        levels=self.fedpaq_levels,
                        generators=generators,
                    )
                    fedpaq_quantization = [
                        {
                            "client_id": int(cid),
                            "quantizer_seed": int(
                                derive_seed(
                                    self.run_seed,
                                    "fedpaq_communication_quantization",
                                    int(cid),
                                )
                            ),
                            **stats,
                        }
                        for cid, stats in zip(cids, quantization_stats)
                    ]
                    aggregation_method = "sample_weighted_adapted_fedpaq"
                elif self.is_fedevi:
                    if fedbn_val_loaders is None:
                        raise ValueError("FedEvi requires client validation loaders")
                    avg_state, aggregation_weights, fedevi_scores = aggregate_fedevi_states(
                        self.global_model,
                        states,
                        aggregation_weights,
                        [fedbn_val_loaders[cid] for cid in cids],
                        self.device,
                        gamma=float(self.acf_policy.get("gamma", 1.0)),
                    )
                    aggregation_method = "adapted_fedevi_uncertainty_weighted"
                    for metric, score in zip(metrics, fedevi_scores):
                        metric["fedevi_distributional_uncertainty"] = float(score[0])
                        metric["fedevi_data_uncertainty"] = float(score[1])
                elif self.is_fedclam:
                    avg_state, self.fedclam_momentum = aggregate_fedclam_states(
                        self.global_model.state_dict(),
                        states,
                        cids,
                        metrics,
                        self.fedclam_momentum,
                        r,
                        agg_lr=float(self.acf_policy.get("agg_lr", 1.0)),
                        zero_init=bool(self.acf_policy.get("zero_init", False)),
                    )
                    aggregation_method = "adapted_fedclam_client_momentum"
                else:
                    excluded_keys = self.bn_state_keys if self.is_fedbn else set()
                    avg_state = weighted_reduce_states(
                        states,
                        aggregation_weights,
                        reference_state=self.global_model.state_dict(),
                        excluded_keys=excluded_keys,
                    )
                    if self.is_fedmpq_proxy:
                        aggregation_method = (
                            "budget_and_sample_weighted_fedmpq_proxy"
                        )
                    else:
                        aggregation_method = (
                            "sample_weighted_fedbn_shared_state"
                            if self.is_fedbn
                            else "sample_weighted_fedavg"
                        )
                    aggregation_excluded_key_count = len(excluded_keys)

                self.global_model.load_state_dict(avg_state, strict=True)
                if self.is_fedmpq_proxy:
                    self.fedmpq_proxy_scheduler.update(
                        cids,
                        [
                            metric["fedmpq_pruned_bit_widths"]
                            for metric in metrics
                        ],
                        [
                            metric["fedmpq_bit_width_delta"]
                            for metric in metrics
                        ],
                        aggregation_weights,
                    )

                model_numel = sum(
                    parameter.numel()
                    for parameter in self.global_model.parameters()
                )
                payload_bits_per_value = 32.0
                if self.is_fedpaq:
                    payload_bits_per_value = (
                        int(math.ceil(math.log2(self.fedpaq_levels + 1))) + 1
                    )
                elif self.is_fedmpq_proxy:
                    payload_bits_per_value = float(
                        np.average(
                            [
                                metric["average_bit_width"]
                                for metric in metrics
                            ],
                            weights=aggregation_weights,
                        )
                    )
                payload_ratio = float(payload_bits_per_value / 32.0)
                model_size_mb = (
                    model_numel * 4.0 * payload_ratio / (1024.0 * 1024.0)
                )
                # [P0] PEC 与 BEU 解耦：聚合加速是否启用由 simulate_hardware_pec 控制
                method = "PEC" if self.enable_pec else "Software"
                agg_lat = float(self.simulator.simulate_aggregation(len(cids), model_size_mb, method))
                comm_lat = float(
                    self.dp_config.get("communication_latency_ms_fp32", 100.0)
                ) * payload_ratio

            # validation (on agg rounds or last round)
            if is_agg or (r == rounds - 1):
                val_stats = self.evaluate_for_strategy(
                    val_loader,
                    client_num_samples,
                    fedbn_val_loaders,
                )
                val_mean = float(val_stats.get("Mean", 0.0))

                if val_mean > self.best_val:
                    self.best_val = val_mean
                    self.best_state = copy.deepcopy(self.global_model.state_dict())
                    if self.is_fedbn:
                        self.best_client_bn_states = copy.deepcopy(
                            self.client_bn_states
                        )
            else:
                val_mean = history["val_dice"][-1] if history["val_dice"] else 0.0
                val_stats = {"Mean": val_mean, "WT": 0.0, "TC": 0.0, "ET": 0.0}

            avg_loss = float(np.mean([m["loss"] for m in metrics]) if metrics else 0.0)
            avg_hidden_ratio = float(np.mean([m.get("dp_hidden_ratio", 0.0) for m in metrics]) if metrics else 0.0)

            # Select one straggler and take every critical-path component from it.
            client_total = [
                float(m.get("compute_latency_ms", 0.0))
                + float(m.get("dp_overhead_ms", 0.0))
                for m in metrics
            ]
            straggler_index = int(np.argmax(client_total)) if client_total else -1
            straggler_metrics = metrics[straggler_index] if straggler_index >= 0 else {}
            straggler_client_id = int(cids[straggler_index]) if straggler_index >= 0 else -1
            straggler_compute = float(straggler_metrics.get("compute_latency_ms", 0.0))
            straggler_dp_visible = float(straggler_metrics.get("dp_overhead_ms", 0.0))
            straggler_dp_total = float(straggler_metrics.get("dp_total_ms", 0.0))
            straggler_dp_background = max(0.0, straggler_dp_total - straggler_dp_visible)
            max_client_total = straggler_compute + straggler_dp_visible

            # energy: sum over participating clients (clients run in parallel but energy adds up)
            total_energy_mJ = float(np.sum([float(m.get("energy_mJ", 0.0)) for m in metrics])) if metrics else 0.0

            misc_lat = float(self.dp_config.get("misc_overhead_ms", 0.0))
            total_lat = max_client_total + float(comm_lat) + float(agg_lat) + misc_lat
            latency_residual = total_lat - (
                straggler_compute
                + straggler_dp_visible
                + float(comm_lat)
                + float(agg_lat)
                + misc_lat
            )
            if abs(latency_residual) > 1e-9:
                raise AssertionError(
                    f"Latency breakdown does not close at round {r}: {latency_residual}"
                )

            delta = float(self.dp_config.get("delta", 1e-5))
            epsilon_per_client = [
                float(self.privacy_accountants[cid].get_epsilon(delta))
                if cid in self.privacy_accountants
                else 0.0
                for cid in range(num_clients)
            ]
            epsilon_max = max(epsilon_per_client) if epsilon_per_client else 0.0
            epsilon_participating_max = max(
                (epsilon_per_client[cid] for cid in cids),
                default=0.0,
            )

            history["round"].append(r)
            history["train_loss"].append(avg_loss)
            history["val_dice"].append(val_mean)
            history["scheduled_clients"].append(scheduled_cids)
            history["participating_clients"].append(cids)
            history["participating_client_num_samples"].append(
                [int(client_num_samples[cid]) for cid in cids]
            )
            history["client_optimizer_steps"].append(
                [int(metric["num_steps"]) for metric in metrics]
            )
            history["client_examples_processed"].append(
                [
                    int(metric["num_examples_processed"])
                    for metric in metrics
                ]
            )
            history["client_privacy_events"].append(
                [metric["privacy_events"] for metric in metrics]
            )
            history["client_precisions"].append(round_precisions)
            history["is_aggregation_round"].append(bool(is_agg))
            history["aggregation_cohort"].append(cids if is_agg else [])
            history["aggregation_weights"].append(aggregation_weights)
            history["aggregation_weight_sum"].append(
                float(sum(aggregation_weights)) if aggregation_weights else 0.0
            )
            history["aggregation_method"].append(aggregation_method)
            history["aggregation_excluded_key_count"].append(
                int(aggregation_excluded_key_count)
            )
            history["fedpaq_quantization"].append(fedpaq_quantization)

            history["client_latency_ms"].append(max_client_total)
            history["compute_latency_ms"].append(straggler_compute)
            history["dp_overhead_ms"].append(straggler_dp_visible)
            history["dp_total_ms"].append(straggler_dp_total)
            history["dp_background_ms"].append(straggler_dp_background)
            history["dp_hidden_ratio"].append(avg_hidden_ratio)
            history["budget_utilization"].append(avg_hidden_ratio)  # alias
            history["hr_theoretical"].append(
                float(np.mean([m.get("hr_theoretical", 0.0) for m in metrics])) if metrics else 0.0
            )
            history["delta_c_cycles"].append(
                float(np.mean([m.get("delta_c_cycles", 0.0) for m in metrics])) if metrics else 0.0
            )
            history["c_priv_cycles"].append(
                float(np.mean([m.get("c_priv_cycles", 0.0) for m in metrics])) if metrics else 0.0
            )
            history["agg_latency_ms"].append(float(agg_lat))
            history["comm_latency_ms"].append(float(comm_lat))
            history["misc_latency_ms"].append(misc_lat)
            history["latency_ms"].append(float(total_lat))
            history["latency_residual_ms"].append(float(latency_residual))
            history["straggler_client_id"].append(straggler_client_id)

            history["energy_mJ"].append(total_energy_mJ)
            history["local_training_energy_mJ"].append(total_energy_mJ)
            history["softdp_energy_mJ"].append(None)
            history["beu_aux_energy_mJ"].append(None)
            history["communication_energy_mJ"].append(None)
            history["server_aggregation_energy_mJ"].append(None)

            history["val_dice_detailed"].append(val_stats)
            history["epsilon"].append(epsilon_max)
            history["epsilon_participating_max"].append(epsilon_participating_max)
            history["epsilon_per_client"].append(epsilon_per_client)

            print(
                f"{r:3d} | {avg_loss:7.4f} | {val_mean:8.2f} | "
                f"{straggler_dp_visible:8.1f} | {total_lat:8.1f}"
            )

        # final test evaluation on best-val checkpoint
        if self.best_state is not None:
            self.global_model.load_state_dict(self.best_state, strict=False)
        if self.is_fedbn and self.best_client_bn_states is not None:
            self.client_bn_states = copy.deepcopy(self.best_client_bn_states)
        test_stats = self.evaluate_for_strategy(
            test_loader,
            client_num_samples,
            fedbn_test_loaders,
        )
        test_mean = float(test_stats.get("Mean", 0.0))


        # [P2] Time-to-Accuracy (wall-clock proxy): 达到 0.9*best_val 所需累计延迟
        t2a_ratio = float(self.dp_config.get("t2a_ratio", 0.90))
        t2a_target = float(self.best_val) * t2a_ratio
        t2a_round = -1
        t2a_ms = 0.0
        if history.get("val_dice") and history.get("latency_ms"):
            cum_lat = np.cumsum(history["latency_ms"])
            for ridx, v in enumerate(history["val_dice"]):
                if float(v) >= t2a_target:
                    t2a_round = int(ridx)
                    t2a_ms = float(cum_lat[ridx])
                    break
            if t2a_round < 0:
                # never reached target within training budget -> use total time
                t2a_round = int(len(history["val_dice"]) - 1)
                t2a_ms = float(np.sum(history["latency_ms"]))

        epsilon_diffs = np.diff(history["epsilon"]) if len(history["epsilon"]) > 1 else []
        if len(epsilon_diffs) and float(np.min(epsilon_diffs)) < -1e-9:
            raise AssertionError("Cumulative max-client epsilon must be non-decreasing")
        for is_aggregation_round, weights in zip(
            history["is_aggregation_round"],
            history["aggregation_weights"],
        ):
            if is_aggregation_round and abs(sum(weights) - 1.0) > 1e-12:
                raise AssertionError("Aggregation weights must sum to one")
            if (not is_aggregation_round) and weights:
                raise AssertionError(
                    "Non-aggregation rounds must not record aggregation weights"
                )

        precision_totals = {
            client_id: {"high": 0, "total": 0}
            for client_id in range(len(client_loaders))
        }
        high_precision = self.acf_scheduler.high_precision
        for client_ids, precisions in zip(
            history["participating_clients"],
            history["client_precisions"],
        ):
            for client_id, precision in zip(client_ids, precisions):
                precision_totals[int(client_id)]["total"] += 1
                if str(precision) == high_precision:
                    precision_totals[int(client_id)]["high"] += 1
        total_assignments = sum(
            counts["total"] for counts in precision_totals.values()
        )
        high_assignments = sum(
            counts["high"] for counts in precision_totals.values()
        )
        high_rate_by_client = [
            (
                float(counts["high"] / counts["total"])
                if counts["total"]
                else None
            )
            for counts in precision_totals.values()
        ]

        history["metrics"] = {
            "best_val_dice": float(self.best_val),
            "test_dice": float(test_mean),
            "accuracy": float(test_mean),  # backward-compatible alias

            # privacy
            "final_epsilon": float(history["epsilon"][-1]) if history.get("epsilon") else 0.0,
            "privacy_accounting_scope": "cumulative maximum over all clients",

            # time-to-accuracy (TCAD-friendly system metric)
            "t2a_ratio": float(t2a_ratio),
            "t2a_target_val_dice": float(t2a_target),
            "t2a_round": int(t2a_round),
            "t2a_ms": float(t2a_ms),
            "total_time_ms": float(np.sum(history["latency_ms"])) if history["latency_ms"] else 0.0,
            "total_energy_mJ": float(np.sum(history["energy_mJ"])) if history["energy_mJ"] else 0.0,
            "total_local_training_energy_mJ": (
                float(np.sum(history["local_training_energy_mJ"]))
                if history["local_training_energy_mJ"]
                else 0.0
            ),
            "energy_scope": history["energy_scope"],
            "aggregation_weight_basis": (
                "participating-client local training-set sample count"
            ),
            "fedbn_evaluation_mode": (
                history["aggregation_config"]["fedbn_evaluation_mode"]
            ),
            "fedpaq_quantization_levels": (
                int(self.fedpaq_levels) if self.is_fedpaq else None
            ),
            "fedpaq_local_update_steps": (
                int(self.fedpaq_local_update_steps)
                if self.is_fedpaq
                else None
            ),

            # latency / energy summary (mean over rounds)
            "avg_latency_ms": float(np.mean(history["latency_ms"])) if history["latency_ms"] else 0.0,
            "avg_client_latency_ms": float(np.mean(history["client_latency_ms"])) if history["client_latency_ms"] else 0.0,
            "avg_compute_latency_ms": float(np.mean(history["compute_latency_ms"])) if history["compute_latency_ms"] else 0.0,
            "avg_agg_latency_ms": float(np.mean(history["agg_latency_ms"])) if history["agg_latency_ms"] else 0.0,
            "avg_comm_latency_ms": float(np.mean(history["comm_latency_ms"])) if history["comm_latency_ms"] else 0.0,
            "avg_misc_latency_ms": float(np.mean(history["misc_latency_ms"])) if history["misc_latency_ms"] else 0.0,

            "avg_dp_overhead_ms": float(np.mean(history["dp_overhead_ms"])) if history["dp_overhead_ms"] else 0.0,
            "avg_dp_total_ms": float(np.mean(history["dp_total_ms"])) if history["dp_total_ms"] else 0.0,
            "avg_dp_background_ms": float(np.mean(history["dp_background_ms"])) if history["dp_background_ms"] else 0.0,
            "avg_dp_hidden_ratio": float(np.mean(history["dp_hidden_ratio"])) if history["dp_hidden_ratio"] else 0.0,
            "max_abs_latency_residual_ms": float(
                np.max(np.abs(history["latency_residual_ms"]))
            ) if history["latency_residual_ms"] else 0.0,

            "avg_energy_mJ": float(np.mean(history["energy_mJ"])) if history["energy_mJ"] else 0.0,
            "avg_local_training_energy_mJ": (
                float(np.mean(history["local_training_energy_mJ"]))
                if history["local_training_energy_mJ"]
                else 0.0
            ),

            "avg_budget_utilization": float(np.mean(history["budget_utilization"])) if history["budget_utilization"] else 0.0,
            # 论文命题：ΔC(p) vs Cpriv 均值（用于Section 4.3的数字支撑）
            "avg_hr_theoretical": float(np.mean(history["hr_theoretical"])) if history["hr_theoretical"] else 0.0,
            "avg_delta_c_cycles": float(np.mean(history["delta_c_cycles"])) if history["delta_c_cycles"] else 0.0,
            "avg_c_priv_cycles": float(np.mean(history["c_priv_cycles"])) if history["c_priv_cycles"] else 0.0,
            "high_precision_name": high_precision,
            "high_precision_assignment_rate": (
                float(high_assignments / total_assignments)
                if total_assignments
                else 0.0
            ),
            "high_precision_assignment_rate_by_client": high_rate_by_client,


        }

        with open(self.output_dir / "training_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, default=json_default)

        return history
