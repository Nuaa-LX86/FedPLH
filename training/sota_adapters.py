from __future__ import annotations

import copy
import math
from collections import OrderedDict
from typing import Dict, Mapping, MutableMapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.losses import DiceLoss
from .aggregation import StateDict, weighted_reduce_states


class EvidentialDiceLoss3D(nn.Module):
    """FedEvi evidential Dice loss generalized to n-D segmentation."""

    def __init__(self, kl_weight: float = 0.01, annealing_step: int = 10):
        super().__init__()
        self.kl_weight = float(kl_weight)
        self.annealing_step = int(annealing_step)
        self.smooth = 1e-5

    @staticmethod
    def _kl_divergence(alpha: torch.Tensor) -> torch.Tensor:
        ones = torch.ones_like(alpha)
        strength = alpha.sum(dim=1, keepdim=True)
        first = (
            torch.lgamma(strength)
            - torch.lgamma(alpha).sum(dim=1, keepdim=True)
            - torch.lgamma(ones.sum(dim=1, keepdim=True))
        )
        second = ((alpha - ones) * (
            torch.digamma(alpha) - torch.digamma(strength)
        )).sum(dim=1, keepdim=True)
        return (first + second).mean()

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        round_number: int,
    ) -> torch.Tensor:
        classes = logits.shape[1]
        if target.ndim == logits.ndim:
            target = target[:, 0] if target.shape[1] == 1 else target.argmax(dim=1)
        evidence = torch.exp(torch.clamp_max(logits, 80.0))
        alpha = evidence + 1.0
        probabilities = alpha / alpha.sum(dim=1, keepdim=True)
        one_hot = F.one_hot(target.long(), classes).movedim(-1, 1).to(logits.dtype)

        # FedEvi reduces over the batch and spatial axes before averaging
        # class-wise Dice.  Keeping the batch axis here would define a
        # different loss whenever the local batch contains multiple volumes.
        reduction_axes = (0, *range(2, logits.ndim))
        intersection = (probabilities * one_hot).sum(dim=reduction_axes)
        denominator = (
            probabilities.square().sum(dim=reduction_axes)
            + one_hot.sum(dim=reduction_axes)
        )
        dice = ((2.0 * intersection + self.smooth) /
                (denominator + self.smooth)).mean()

        annealing = min(1.0, float(round_number) / float(self.annealing_step))
        kl_alpha = (alpha - 1.0) * (1.0 - one_hot) + 1.0
        return (1.0 - dice) + self.kl_weight * annealing * self._kl_divergence(kl_alpha)


class ForegroundIntensityMatchingLoss3D(nn.Module):
    """Memory-bounded 3D adaptation of FedCLAM foreground matching."""

    def __init__(self, pooled_size: int = 16, smooth: float = 1e-5):
        super().__init__()
        self.pooled_size = int(pooled_size)
        self.smooth = float(smooth)

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        image: torch.Tensor,
    ) -> torch.Tensor:
        if target.ndim == logits.ndim:
            target = target[:, 0] if target.shape[1] == 1 else target.argmax(dim=1)
        foreground_probability = 1.0 - torch.softmax(logits, dim=1)[:, :1]
        foreground_mask = (target > 0).to(logits.dtype).unsqueeze(1)

        intensity = image.mean(dim=1, keepdim=True)
        reduce_dims = tuple(range(2, intensity.ndim))
        minimum = intensity.amin(dim=reduce_dims, keepdim=True)
        maximum = intensity.amax(dim=reduce_dims, keepdim=True)
        intensity = (intensity - minimum) / (maximum - minimum + self.smooth)

        output_size = (self.pooled_size,) * (logits.ndim - 2)
        prediction = F.adaptive_avg_pool3d(
            foreground_probability * intensity, output_size
        )
        ground_truth = F.adaptive_avg_pool3d(
            foreground_mask * intensity, output_size
        )
        prediction = torch.sort(prediction.flatten(start_dim=1), dim=1).values
        ground_truth = torch.sort(ground_truth.flatten(start_dim=1), dim=1).values
        return torch.sqrt(F.mse_loss(prediction, ground_truth) + self.smooth)


class FedCLAMDiceFIMLoss3D(nn.Module):
    def __init__(
        self,
        classes: int,
        lambda_dice: float = 0.5,
        lambda_fim: float = 0.5,
        fim_warmup_rounds: int = 10,
        pooled_size: int = 16,
    ):
        super().__init__()
        self.dice = DiceLoss(classes)
        self.fim = ForegroundIntensityMatchingLoss3D(pooled_size=pooled_size)
        self.lambda_dice = float(lambda_dice)
        self.lambda_fim = float(lambda_fim)
        self.fim_warmup_rounds = int(fim_warmup_rounds)

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        image: torch.Tensor,
        round_index: int,
    ) -> torch.Tensor:
        dice = self.dice(logits, target)
        if int(round_index) < self.fim_warmup_rounds:
            return dice
        return self.lambda_dice * dice + self.lambda_fim * self.fim(
            logits, target, image
        )


def calculate_vlr(initial_loss: float, trained_loss: float, beta: float) -> float:
    argument = -float(beta) * (float(initial_loss) - float(trained_loss)) / (
        float(trained_loss) + 1e-6
    )
    argument = max(-60.0, min(60.0, argument))
    return float(1.0 / (1.0 + math.exp(argument)))


def calculate_overfitting_penalty(
    train_loss: float,
    validation_loss: float,
    alpha: float,
) -> float:
    ratio = max(0.0, float(train_loss) / (float(validation_loss) + 1e-6))
    return float(1.0 - min(ratio ** float(alpha), 1.0))


@torch.no_grad()
def fedevi_uncertainty_scores(
    global_model: nn.Module,
    local_model: nn.Module,
    loader,
    device: str,
) -> tuple[float, float]:
    global_model.eval()
    local_model.eval()
    distributional = []
    data_uncertainty = []
    for image, _ in loader:
        image = image.to(device)
        global_alpha = torch.exp(torch.clamp_max(global_model(image), 80.0)) + 1.0
        global_strength = global_alpha.sum(dim=1, keepdim=True)
        global_probability = global_alpha / global_strength
        entropy = -(global_probability * torch.log(global_probability)).sum(dim=1)
        global_data = (
            global_probability
            * (torch.digamma(global_strength + 1.0) - torch.digamma(global_alpha + 1.0))
        ).sum(dim=1)
        distributional.append((entropy - global_data).mean())

        local_alpha = torch.exp(torch.clamp_max(local_model(image), 80.0)) + 1.0
        local_strength = local_alpha.sum(dim=1, keepdim=True)
        local_probability = local_alpha / local_strength
        local_data = (
            local_probability
            * (torch.digamma(local_strength + 1.0) - torch.digamma(local_alpha + 1.0))
        ).sum(dim=1)
        data_uncertainty.append(local_data.mean())

    if not distributional or not data_uncertainty:
        raise ValueError("FedEvi requires non-empty client validation loaders")
    return (
        float(torch.stack(distributional).mean().item()),
        float(torch.stack(data_uncertainty).mean().item()),
    )


def fedevi_adjust_weights(
    base_weights: Sequence[float],
    scores: Sequence[tuple[float, float]],
    gamma: float = 1.0,
) -> list[float]:
    if len(base_weights) != len(scores):
        raise ValueError("FedEvi weights and scores must have equal length")
    adjusted = [
        float(weight) + float(gamma) * max(0.0, u_dis) / max(u_data, 1e-12)
        for weight, (u_dis, u_data) in zip(base_weights, scores)
    ]
    denominator = sum(adjusted)
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("FedEvi produced invalid aggregation weights")
    return [float(value / denominator) for value in adjusted]


def aggregate_fedevi_states(
    global_model: nn.Module,
    client_states: Sequence[StateDict],
    base_weights: Sequence[float],
    validation_loaders: Sequence,
    device: str,
    gamma: float = 1.0,
) -> tuple[OrderedDict[str, torch.Tensor], list[float], list[tuple[float, float]]]:
    surrogate_state = weighted_reduce_states(client_states, base_weights)
    surrogate = copy.deepcopy(global_model).to(device)
    surrogate.load_state_dict(surrogate_state, strict=True)
    scores = []
    for state, loader in zip(client_states, validation_loaders):
        local = copy.deepcopy(global_model).to(device)
        local.load_state_dict(state, strict=True)
        scores.append(fedevi_uncertainty_scores(surrogate, local, loader, device))
        del local
    adjusted = fedevi_adjust_weights(base_weights, scores, gamma=gamma)
    result = weighted_reduce_states(client_states, adjusted)
    del surrogate
    return result, adjusted, scores


def aggregate_fedclam_states(
    base_state: StateDict,
    client_states: Sequence[StateDict],
    client_ids: Sequence[int],
    client_metrics: Sequence[Mapping[str, float]],
    momentum: MutableMapping[int, Dict[str, torch.Tensor]],
    round_index: int,
    agg_lr: float = 1.0,
    zero_init: bool = False,
) -> tuple[OrderedDict[str, torch.Tensor], MutableMapping[int, Dict[str, torch.Tensor]]]:
    if not (len(client_states) == len(client_ids) == len(client_metrics)):
        raise ValueError("FedCLAM cohort inputs must have equal length")
    equal = [1.0 / len(client_states)] * len(client_states)
    fedavg = weighted_reduce_states(client_states, equal)
    result: OrderedDict[str, torch.Tensor] = OrderedDict()

    for key, base_tensor in base_state.items():
        if not (key.endswith("weight") or key.endswith("bias")):
            result[key] = fedavg[key].clone()
            continue
        pseudo_gradient = base_tensor.to(torch.float32) - fedavg[key].to(torch.float32)
        speeds = []
        for client_id, metrics in zip(client_ids, client_metrics):
            client_memory = momentum.setdefault(int(client_id), {})
            previous = client_memory.get(key)
            if previous is None:
                previous = torch.zeros_like(pseudo_gradient)
            if int(round_index) == 0:
                updated = torch.zeros_like(pseudo_gradient) if zero_init else pseudo_gradient
            else:
                coefficient = max(0.0, min(1.0, float(metrics["val_loss_ratio"])))
                dampening = 1.0 - float(metrics["overfitting_penalty"])
                updated = coefficient * previous + dampening * pseudo_gradient
            client_memory[key] = updated.detach().clone()
            speeds.append(updated)
        average_speed = torch.stack(speeds).mean(dim=0)
        result[key] = (
            base_tensor.to(torch.float32) - float(agg_lr) * average_speed
        ).to(base_tensor.dtype)
    return result, momentum
