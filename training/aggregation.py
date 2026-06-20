from __future__ import annotations

import copy
import math
from collections import OrderedDict
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn


StateDict = Mapping[str, torch.Tensor]


def normalize_client_weights(
    client_ids: Sequence[int],
    client_num_samples: Sequence[int],
) -> List[float]:
    if not client_ids:
        raise ValueError("At least one participating client is required")
    if len(set(int(client_id) for client_id in client_ids)) != len(client_ids):
        raise ValueError("Participating client IDs must be unique")

    sample_counts = []
    for client_id in client_ids:
        client_id = int(client_id)
        if client_id < 0 or client_id >= len(client_num_samples):
            raise IndexError(f"Client ID out of range: {client_id}")
        sample_count = int(client_num_samples[client_id])
        if sample_count <= 0:
            raise ValueError(
                f"Client {client_id} has non-positive training sample count"
            )
        sample_counts.append(sample_count)

    denominator = float(sum(sample_counts))
    weights = [float(sample_count / denominator) for sample_count in sample_counts]
    if abs(sum(weights) - 1.0) > 1e-12:
        raise AssertionError("Normalized client weights do not sum to one")
    return weights


def batchnorm_state_keys(model: nn.Module) -> Set[str]:
    keys: Set[str] = set()
    for module_name, module in model.named_modules():
        if not isinstance(module, nn.modules.batchnorm._BatchNorm):
            continue
        prefix = f"{module_name}." if module_name else ""
        for state_name in module.state_dict().keys():
            keys.add(f"{prefix}{state_name}")
    return keys


def _validate_states(states: Sequence[StateDict]) -> List[str]:
    if not states:
        raise ValueError("At least one client state is required")
    reference_keys = list(states[0].keys())
    reference_key_set = set(reference_keys)
    for index, state in enumerate(states[1:], start=1):
        if set(state.keys()) != reference_key_set:
            raise ValueError(f"Client state {index} has a different key set")
    return reference_keys


def _weighted_tensor(
    tensors: Sequence[torch.Tensor],
    weights: Sequence[float],
) -> torch.Tensor:
    reference = tensors[0]
    if reference.is_floating_point() or reference.is_complex():
        accumulation_dtype = (
            torch.float32
            if reference.dtype in (torch.float16, torch.bfloat16)
            else reference.dtype
        )
        result = torch.zeros_like(reference, dtype=accumulation_dtype)
        for weight, tensor in zip(weights, tensors):
            result.add_(tensor.to(dtype=accumulation_dtype), alpha=float(weight))
        return result.to(dtype=reference.dtype)

    if all(torch.equal(reference, tensor) for tensor in tensors[1:]):
        return reference.clone()

    result = torch.zeros_like(reference, dtype=torch.float64)
    for weight, tensor in zip(weights, tensors):
        result.add_(tensor.to(dtype=torch.float64), alpha=float(weight))
    if reference.dtype == torch.bool:
        return (result >= 0.5).to(dtype=reference.dtype)
    return torch.round(result).to(dtype=reference.dtype)


def weighted_reduce_states(
    states: Sequence[StateDict],
    weights: Sequence[float],
    *,
    reference_state: Optional[StateDict] = None,
    excluded_keys: Optional[Iterable[str]] = None,
) -> "OrderedDict[str, torch.Tensor]":
    keys = _validate_states(states)
    if len(states) != len(weights):
        raise ValueError("Number of states and weights must match")
    if any((not math.isfinite(float(weight))) or float(weight) < 0 for weight in weights):
        raise ValueError("Aggregation weights must be finite and non-negative")
    if abs(sum(float(weight) for weight in weights) - 1.0) > 1e-12:
        raise ValueError("Aggregation weights must sum to one")

    excluded = set(excluded_keys or ())
    if reference_state is None:
        reference_state = states[0]

    reduced: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for key in keys:
        if key in excluded:
            reduced[key] = reference_state[key].clone()
            continue
        reduced[key] = _weighted_tensor(
            [state[key] for state in states],
            weights,
        )
    return reduced


def fedpaq_qsgd_quantize(
    vector: torch.Tensor,
    levels: int,
    *,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if levels <= 0:
        raise ValueError("FedPAQ quantization levels must be positive")
    if not vector.is_floating_point():
        raise TypeError("FedPAQ quantization requires a floating-point tensor")
    if vector.numel() == 0:
        return vector.clone()

    norm = torch.linalg.vector_norm(vector)
    if float(norm.item()) == 0.0:
        return torch.zeros_like(vector)

    scaled = vector.abs().div(norm).mul(float(levels))
    lower = torch.floor(scaled)
    probability = scaled - lower
    random_values = torch.rand(
        probability.shape,
        dtype=probability.dtype,
        device=probability.device,
        generator=generator,
    )
    quantized_level = lower + (random_values < probability).to(lower.dtype)
    return vector.sign().mul(norm).mul(quantized_level.div(float(levels)))


def aggregate_fedpaq_deltas(
    base_state: StateDict,
    client_states: Sequence[StateDict],
    weights: Sequence[float],
    *,
    levels: int,
    generators: Optional[Sequence[Optional[torch.Generator]]] = None,
) -> Tuple["OrderedDict[str, torch.Tensor]", List[Dict[str, float]]]:
    keys = _validate_states(client_states)
    if set(base_state.keys()) != set(keys):
        raise ValueError("Base state and client states have different key sets")
    if len(client_states) != len(weights):
        raise ValueError("Number of client states and weights must match")
    if abs(sum(float(weight) for weight in weights) - 1.0) > 1e-12:
        raise ValueError("Aggregation weights must sum to one")
    if generators is None:
        generators = [None] * len(client_states)
    if len(generators) != len(client_states):
        raise ValueError("Number of generators and client states must match")

    floating_keys = [
        key for key in keys if base_state[key].is_floating_point()
    ]
    shapes = [base_state[key].shape for key in floating_keys]
    sizes = [base_state[key].numel() for key in floating_keys]

    accumulated_delta = {
        key: torch.zeros_like(
            base_state[key],
            dtype=(
                torch.float32
                if base_state[key].dtype in (torch.float16, torch.bfloat16)
                else base_state[key].dtype
            ),
        )
        for key in floating_keys
    }
    quantization_stats: List[Dict[str, float]] = []

    for state, weight, generator in zip(client_states, weights, generators):
        flattened_delta = torch.cat([
            (state[key] - base_state[key]).reshape(-1)
            for key in floating_keys
        ])
        quantized = fedpaq_qsgd_quantize(
            flattened_delta,
            levels,
            generator=generator,
        )
        offset = 0
        for key, shape, size in zip(floating_keys, shapes, sizes):
            tensor_delta = quantized[offset:offset + size].reshape(shape)
            accumulated_delta[key].add_(
                tensor_delta.to(dtype=accumulated_delta[key].dtype),
                alpha=float(weight),
            )
            offset += size

        error = quantized - flattened_delta
        quantization_stats.append({
            "update_l2_norm": float(
                torch.linalg.vector_norm(flattened_delta).item()
            ),
            "quantization_error_l2_norm": float(
                torch.linalg.vector_norm(error).item()
            ),
            "num_quantized_values": int(flattened_delta.numel()),
        })

    reduced_non_floating = weighted_reduce_states(
        client_states,
        weights,
        reference_state=base_state,
        excluded_keys=floating_keys,
    )
    result = copy.deepcopy(reduced_non_floating)
    for key in floating_keys:
        result[key] = (
            base_state[key].to(dtype=accumulated_delta[key].dtype)
            + accumulated_delta[key]
        ).to(dtype=base_state[key].dtype)
    return result, quantization_stats
