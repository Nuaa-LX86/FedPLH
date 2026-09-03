# models/precision_wrapper.py
# for numerical stability, following common practice in quantized training.
import torch
import torch.nn as nn
from typing import Dict, Union

try:
    from torch.func import functional_call as _functional_call

    def functional_call(module, replacements, args, kwargs):
        return _functional_call(
            module, replacements, args=args, kwargs=kwargs, strict=False
        )
except ImportError:
    from torch.nn.utils.stateless import functional_call as _functional_call

    def functional_call(module, replacements, args, kwargs):
        return _functional_call(module, replacements, args, kwargs, strict=False)

PrecisionMode = Union[str, Dict[str, str]]


class HMPEPrecisionEmulator(nn.Module):
    """
    HMPE 硬件精度模拟器（更稳健版本）
    - 对 FP8/INT8 使用对数量化 / 线性量化 + SR + STE
    - 对 BF16/FP16 采用原生 dtype cast（更接近真实硬件，也更稳定）
    - 默认不量化最后一层 logits (例如 UNet 的 outc)，避免 Dice 彻底崩盘
    """

    def __init__(
        self,
        model: nn.Module,
        default_precision: str = 'FP32',
        quantize_weights: bool = False,
    ):
        super().__init__()
        self.model = model
        self.default_precision = default_precision
        self.quantize_weights = bool(quantize_weights)
        self.layer_policies: Dict[str, str] = {}
        self.quantization_generator = None

        # (Exponent Bits, Mantissa Bits, Max Value, Min Normal)
        self.format_specs = {
            'FP32': (8, 23, 3.4e38, 1.18e-38),
            'BF16': (8, 7, 3.4e38, 1.18e-38),
            'FP16': (5, 10, 65504, 6.10e-5),
            'FP8_E5M2': (5, 2, 57344, 6.10e-5),
            'FP8_E4M3': (4, 3, 448, 1.95e-3),
            'INT8': (0, 7, 127, 0)
        }

        # 默认不量化的层名后缀（可以按需扩展）
        self.blacklist_suffix = ['outc']  # 最后一层输出层

    def update_policy(self, policy: PrecisionMode):
        if isinstance(policy, str):
            self.default_precision = policy
            self.layer_policies = {}
        elif isinstance(policy, dict):
            self.layer_policies = policy
        else:
            raise ValueError("Policy must be str or dict")

    def set_quantization_generator(self, generator: torch.Generator):
        self.quantization_generator = generator

    # --------- 核心：按 layer 决定是否量化 & 用什么模式 ---------

    def _should_quantize(self, name: str, module: nn.Module) -> bool:
        # 1) 跳过黑名单层（例如 outc）
        if any(name.endswith(suf) for suf in self.blacklist_suffix):
            return False
        # 2) 只量化 Conv / Linear / MHA
        return isinstance(
            module,
            (
                nn.Conv2d,
                nn.Conv3d,
                nn.ConvTranspose2d,
                nn.ConvTranspose3d,
                nn.Linear,
                nn.MultiheadAttention,
            ),
        )

    def _get_mode_for_layer(self, name: str) -> str:
        # per-layer 策略 > default
        return self.layer_policies.get(name, self.default_precision)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # FP32 + 无策略时直接透传
        if self.default_precision == 'FP32' and not self.layer_policies:
            return self.model(x, *args, **kwargs)

        hooks = []

        # 这里用闭包把 layer 名字捕获进去
        def make_pre_hook(layer_name: str):
            def quantize_input_hook(module, inputs):
                mode = self._get_mode_for_layer(layer_name)
                # FP32 不量化
                if mode == 'FP32':
                    return inputs

                new_inputs = []
                for inp in inputs:
                    if isinstance(inp, torch.Tensor) and inp.is_floating_point():
                        new_inputs.append(self._fake_quantize(inp, mode))
                    else:
                        new_inputs.append(inp)
                return tuple(new_inputs)
            return quantize_input_hook

        # 为每个需要量化的层注册 hook
        for name, module in self.model.named_modules():
            if self._should_quantize(name, module):
                hooks.append(module.register_forward_pre_hook(make_pre_hook(name)))

        replacements = {}
        if self.quantize_weights:
            for name, module in self.model.named_modules():
                if not self._should_quantize(name, module):
                    continue
                mode = self._get_mode_for_layer(name)
                if mode == 'FP32':
                    continue
                for parameter_name, parameter in module.named_parameters(recurse=False):
                    if "weight" not in parameter_name or not parameter.is_floating_point():
                        continue
                    qualified_name = (
                        f"{name}.{parameter_name}" if name else parameter_name
                    )
                    replacements[qualified_name] = self._fake_quantize(parameter, mode)

        try:
            call_args = (x, *args)
            if replacements:
                out = functional_call(self.model, replacements, call_args, kwargs)
            else:
                out = self.model(*call_args, **kwargs)
        finally:
            for h in hooks:
                h.remove()

        return out

    @property
    def operand_model(self) -> str:
        return (
            "quantized_activations_and_weights_fp32_accumulation"
            if self.quantize_weights
            else "legacy_activation_only_fake_quantization"
        )

    # --------- 伪量化实现 ---------

    def _fake_quantize(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        """核心量化模拟逻辑 (FP8/INT8: 手写量化; BF16/FP16: dtype cast)"""

        # 1) BF16 / FP16：使用原生 dtype cast，最稳妥
        if mode == 'BF16':
            x_bf16 = x.to(torch.bfloat16)
            x_quant = x_bf16.to(torch.float32)
            return (x_quant - x).detach() + x

        if mode == 'FP16':
            x_fp16 = x.to(torch.float16)
            x_quant = x_fp16.to(torch.float32)
            return (x_quant - x).detach() + x

        # 2) 其它模式：按自定义格式量化
        if mode == 'FP32':
            return x

        spec = self.format_specs.get(mode, None)
        if spec is None:
            # 未知模式直接透传
            return x

        exp_bits, man_bits, max_val, min_normal = spec

        # 2.1 截断动态范围，模拟 overflow
        x_clamped = torch.clamp(x, -max_val, max_val)

        # 2.2 INT 模式：线性对称量化
        if 'INT' in mode:
            if x_clamped.numel() == 0:
                return x_clamped
            scale = max_val / x_clamped.abs().max().clamp(min=1e-8)
            x_int = torch.round(x_clamped * scale)
            x_dequant = x_int / scale
            return (x_dequant - x_clamped).detach() + x_clamped

        # 2.3 FP8 等模式：对数刻度 + SR + STE
        if man_bits > 0:
            # 加上极小值防止 log2(0)
            abs_x = x_clamped.abs() + 1e-20

            exponent = torch.floor(torch.log2(abs_x))
            bias = 2 ** (exp_bits - 1) - 1
            min_exp = 1 - bias
            exponent = torch.clamp(exponent, min=min_exp)

            step = 2.0 ** (exponent - man_bits)

            if self.training:
                noise = torch.rand(
                    x_clamped.shape,
                    dtype=x_clamped.dtype,
                    device=x_clamped.device,
                    generator=self.quantization_generator,
                )
                x_quant = torch.floor(x_clamped / step + noise) * step
            else:
                x_quant = torch.round(x_clamped / step) * step

            return (x_quant - x_clamped).detach() + x_clamped

        return x_clamped

    # --------- 透传底层模型属性 ---------

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)
