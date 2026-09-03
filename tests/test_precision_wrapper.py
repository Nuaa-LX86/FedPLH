import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.precision_wrapper import HMPEPrecisionEmulator


class PrecisionWrapperTests(unittest.TestCase):
    def test_bf16_operand_model_quantizes_activation_and_weight(self):
        convolution = nn.Conv3d(1, 1, kernel_size=1, bias=False)
        with torch.no_grad():
            convolution.weight.fill_(0.12345)
        emulator = HMPEPrecisionEmulator(
            convolution,
            default_precision="BF16",
            quantize_weights=True,
        ).eval()
        input_tensor = torch.tensor([[[[[0.23456]]]]], dtype=torch.float32)
        actual = emulator(input_tensor)
        expected = F.conv3d(
            input_tensor.to(torch.bfloat16).to(torch.float32),
            convolution.weight.to(torch.bfloat16).to(torch.float32),
        )
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(
            emulator.operand_model,
            "quantized_activations_and_weights_fp32_accumulation",
        )

    def test_legacy_mode_leaves_weight_unquantized(self):
        convolution = nn.Conv3d(1, 1, kernel_size=1, bias=False)
        with torch.no_grad():
            convolution.weight.fill_(0.12345)
        legacy = HMPEPrecisionEmulator(
            convolution,
            default_precision="BF16",
            quantize_weights=False,
        ).eval()
        aligned = HMPEPrecisionEmulator(
            convolution,
            default_precision="BF16",
            quantize_weights=True,
        ).eval()
        input_tensor = torch.tensor([[[[[0.23456]]]]], dtype=torch.float32)
        self.assertFalse(torch.equal(legacy(input_tensor), aligned(input_tensor)))
        self.assertEqual(
            legacy.operand_model,
            "legacy_activation_only_fake_quantization",
        )

    def test_quantized_weight_uses_ste_gradient(self):
        convolution = nn.Conv3d(1, 1, kernel_size=1, bias=False)
        emulator = HMPEPrecisionEmulator(
            convolution,
            default_precision="FP8_E5M2",
            quantize_weights=True,
        ).train()
        emulator.set_quantization_generator(torch.Generator().manual_seed(7))
        input_tensor = torch.ones((1, 1, 1, 1, 1), dtype=torch.float32)
        emulator(input_tensor).sum().backward()
        self.assertIsNotNone(convolution.weight.grad)
        self.assertGreater(float(convolution.weight.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
