from __future__ import annotations

from typing import Dict, Mapping, Union

import torch
import torch.nn as nn
from torch.nn.utils import parametrize


BitPolicy = Union[str, Mapping[str, Union[int, str]]]


def _parse_bits(value: Union[int, str]) -> int:
    if isinstance(value, int):
        bits = value
    else:
        text = str(value).upper().strip()
        if text == "FP32":
            return 32
        if text.startswith("INT"):
            text = text[3:]
        bits = int(text)
    if bits not in {1, 2, 3, 4, 5, 6, 7, 8, 32}:
        raise ValueError(f"Unsupported FedMPQ proxy bit width: {bits}")
    return bits


def _fake_quantize_symmetric(tensor: torch.Tensor, bits: int) -> torch.Tensor:
    if bits >= 16 or tensor.numel() == 0:
        return tensor
    if bits == 1:
        scale = tensor.detach().abs().amax().clamp(min=1e-8)
        quantized = torch.where(
            tensor >= 0,
            scale.expand_as(tensor),
            -scale.expand_as(tensor),
        )
        return tensor + (quantized - tensor).detach()
    qmax = float((1 << (bits - 1)) - 1)
    scale = tensor.detach().abs().amax().clamp(min=1e-8) / qmax
    quantized = torch.clamp(torch.round(tensor / scale), -qmax, qmax) * scale
    return tensor + (quantized - tensor).detach()


class _WeightQuantizer(nn.Module):
    def __init__(self, bits: int = 8):
        super().__init__()
        self.bits = int(bits)

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        return _fake_quantize_symmetric(weight, self.bits)


class _BinaryPlanesSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight: torch.Tensor, bits: int):
        bits = int(bits)
        qmax = (1 << bits) - 1
        scale = 2.0 * weight.detach().abs().amax().clamp(min=1e-8)
        zero_point = float(1 << (bits - 1))
        integer = torch.clamp(
            torch.round(weight / scale * float(qmax) + zero_point),
            0,
            qmax,
        ).to(torch.int64)
        shifts = torch.arange(
            bits - 1,
            -1,
            -1,
            device=weight.device,
            dtype=torch.int64,
        )
        shape = (bits,) + (1,) * weight.ndim
        planes = (
            (integer.unsqueeze(0) >> shifts.view(shape)) & 1
        ).to(weight.dtype)
        ctx.bits = bits
        return planes

    @staticmethod
    def backward(ctx, grad_planes: torch.Tensor):
        bits = int(ctx.bits)
        qmax = float((1 << bits) - 1)
        powers = torch.tensor(
            [1 << shift for shift in range(bits - 1, -1, -1)],
            dtype=grad_planes.dtype,
            device=grad_planes.device,
        )
        shape = (bits,) + (1,) * (grad_planes.ndim - 1)
        grad_weight = torch.sum(
            grad_planes * powers.view(shape),
            dim=0,
        ) / qmax
        return grad_weight, None


def _binary_planes(weight: torch.Tensor, bits: int) -> torch.Tensor:
    return _BinaryPlanesSTE.apply(weight, int(bits))


class FedMPQProxy(nn.Module):
    """
    Clean-room functional adaptation of the FedMPQ training path.

    The master tensors remain floating point so the method can share the
    existing 3D U-Net and SoftDP optimizer. Forward execution uses STE
    fixed-point weights, activations remain at 4 bits as in the paper,
    group-Lasso acts on binary weight planes, and local MSB pruning updates
    layer bit widths. This is an adapted baseline rather than a byte-equivalent
    reproduction of the authors' integer optimizer.
    """

    def __init__(
        self,
        model: nn.Module,
        default_bits: int = 8,
        activation_bits: int = 4,
    ):
        super().__init__()
        self.model = model
        self.default_bits = _parse_bits(default_bits)
        self.activation_bits = _parse_bits(activation_bits)
        self.quantization_enabled = True
        self.layer_bits: Dict[str, int] = {}
        self._quantizers: Dict[str, _WeightQuantizer] = {}
        self._quantized_modules: Dict[str, nn.Module] = {}

        for name, module in self.model.named_modules():
            if not isinstance(
                module,
                (nn.Conv3d, nn.ConvTranspose3d, nn.Linear),
            ):
                continue
            quantizer = _WeightQuantizer(self.default_bits)
            parametrize.register_parametrization(
                module,
                "weight",
                quantizer,
            )
            self._quantizers[name] = quantizer
            self._quantized_modules[name] = module
            self.layer_bits[name] = self.default_bits

        if not self._quantized_modules:
            raise ValueError("FedMPQProxy found no quantizable layers")

    def update_policy(self, policy: BitPolicy) -> None:
        if isinstance(policy, str):
            bits = _parse_bits(policy)
            self.quantization_enabled = bits < 16
            for name in self.layer_bits:
                self.layer_bits[name] = bits
                self._quantizers[name].bits = bits
            return

        unknown = set(policy) - set(self.layer_bits)
        if unknown:
            raise KeyError(f"Unknown FedMPQ proxy layers: {sorted(unknown)}")
        self.quantization_enabled = True
        for name in self.layer_bits:
            bits = _parse_bits(policy.get(name, self.default_bits))
            self.layer_bits[name] = bits
            self._quantizers[name].bits = bits

    def set_quantization_generator(self, generator: torch.Generator) -> None:
        # The feasibility proxy uses deterministic rounding.
        del generator

    def layer_parameter_counts(self) -> Dict[str, int]:
        return {
            name: int(module.parametrizations.weight.original.numel())
            for name, module in self._quantized_modules.items()
        }

    def average_bit_width(self) -> float:
        counts = self.layer_parameter_counts()
        denominator = float(sum(counts.values()))
        return float(
            sum(counts[name] * self.layer_bits[name] for name in counts)
            / denominator
        )

    def profile_precision(self) -> str:
        # The available hardware profile contains INT4 and INT8 only. Mapping
        # lower/intermediate budgets upward is conservative.
        return "INT4" if self.average_bit_width() <= 4.0 else "INT8"

    @torch.no_grad()
    def quantization_feedback(self) -> Dict[str, float]:
        feedback: Dict[str, float] = {}
        for name, module in self._quantized_modules.items():
            weight = module.parametrizations.weight.original.detach()
            quantized = _fake_quantize_symmetric(
                weight,
                self.layer_bits[name],
            )
            error = torch.mean((quantized - weight) ** 2)
            signal = torch.mean(weight ** 2).clamp(min=1e-12)
            feedback[name] = float((error / signal).item())
        return feedback

    def bit_group_lasso(self) -> torch.Tensor:
        counts = self.layer_parameter_counts()
        total_parameters = float(sum(counts.values()))
        regularizer = None
        for name, module in self._quantized_modules.items():
            bits = self.layer_bits[name]
            if bits >= 16:
                continue
            weight = module.parametrizations.weight.original
            planes = _binary_planes(weight, bits)
            group_norms = torch.sqrt(
                torch.sum(
                    planes * planes,
                    dim=tuple(range(1, planes.ndim)),
                )
                + 1e-8
            )
            layer_term = (
                float(counts[name]) / total_parameters
            ) * torch.sum(group_norms)
            regularizer = (
                layer_term
                if regularizer is None
                else regularizer + layer_term
            )
        if regularizer is None:
            return next(self.parameters()).new_zeros(())
        return regularizer

    @torch.no_grad()
    def prune_msb(self, threshold: float) -> Dict[str, int]:
        for name, module in self._quantized_modules.items():
            bits = int(self.layer_bits[name])
            weight = module.parametrizations.weight.original.detach()
            while bits > 1:
                planes = _binary_planes(weight, bits)
                if float(planes[0].mean().item()) > float(threshold):
                    break
                bits -= 1
            self.layer_bits[name] = bits
            self._quantizers[name].bits = bits
        return dict(self.layer_bits)

    def _activation_pre_hook(self, layer_name: str):
        def hook(module, inputs):
            del module
            if not self.quantization_enabled:
                return inputs
            bits = self.activation_bits
            return tuple(
                _fake_quantize_symmetric(value, bits)
                if isinstance(value, torch.Tensor)
                and value.is_floating_point()
                else value
                for value in inputs
            )

        return hook

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if not self.quantization_enabled:
            return self.model(x, *args, **kwargs)

        hooks = [
            module.register_forward_pre_hook(
                self._activation_pre_hook(name)
            )
            for name, module in self._quantized_modules.items()
        ]
        try:
            return self.model(x, *args, **kwargs)
        finally:
            for hook in hooks:
                hook.remove()

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)
