import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from training.federated_trainer import FederatedTrainer


class TinySegmentationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(4, 4, kernel_size=1, bias=False),
            nn.BatchNorm3d(4),
            nn.ReLU(),
        )
        self.out = nn.Conv3d(4, 4, kernel_size=1)

    def forward(self, inputs):
        return self.out(self.encoder(inputs))


def make_loader(num_samples, offset=0):
    generator = torch.Generator().manual_seed(100 + offset)
    inputs = torch.randn(
        num_samples,
        4,
        4,
        4,
        4,
        generator=generator,
    )
    targets = torch.arange(4 * 4 * 4).reshape(4, 4, 4) % 4
    targets = targets.unsqueeze(0).repeat(num_samples, 1, 1, 1)
    return DataLoader(
        TensorDataset(inputs, targets.long()),
        batch_size=1,
        shuffle=False,
    )


class FederatedSemanticSmokeTests(unittest.TestCase):
    def _run(self, strategy, interval, schedule):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        output_dir = Path(temporary_directory.name)
        client_loaders = [
            make_loader(2, 0),
            make_loader(1, 1),
            make_loader(3, 2),
        ]
        client_stats = [
            {"num_samples": 2, "entropy": 0.2},
            {"num_samples": 1, "entropy": 0.5},
            {"num_samples": 3, "entropy": 0.8},
        ]
        trainer = FederatedTrainer(
            model=TinySegmentationModel(),
            optimizer_fn=lambda parameters: torch.optim.SGD(
                parameters,
                lr=0.01,
            ),
            criterion=nn.CrossEntropyLoss(),
            device="cpu",
            dp_config={"enable": False},
            acf_policy={
                "compute": "FP32",
                "strategy": strategy,
                "quantization_levels": 255,
                "local_update_steps": 2,
            },
            hw_profile_path="hardware_profile.json",
            output_dir=str(output_dir),
            comm_interval=interval,
            run_seed=7,
            client_schedule=schedule,
            use_amp=False,
        )
        trainer._get_hw_step_profile = lambda *args, **kwargs: {
            "cycles_fp32": 0.0,
            "cycles_actual": 0.0,
            "energy_fp32_mJ": 0.0,
            "energy_actual_mJ": 0.0,
            "surplus_cycles": 0.0,
        }
        history = trainer.run(
            client_loaders,
            make_loader(1, 10),
            make_loader(1, 11),
            client_stats,
            rounds=len(schedule),
            local_epochs=1,
        )
        self.assertTrue((output_dir / "training_history.json").exists())
        return history, trainer

    def test_fedavg_logs_sample_weighted_reduction(self):
        history, _ = self._run(
            "FedAvg",
            1,
            [[0, 1]],
        )
        self.assertEqual(
            history["aggregation_method"],
            ["sample_weighted_fedavg"],
        )
        self.assertEqual(history["aggregation_weights"], [[2 / 3, 1 / 3]])
        self.assertAlmostEqual(history["aggregation_weight_sum"][0], 1.0)
        self.assertEqual(
            history["energy_component_status"]["communication"],
            "not modeled",
        )
        self.assertEqual(
            history["energy_mJ"],
            history["local_training_energy_mJ"],
        )

    def test_fedbn_keeps_all_bn_state_private(self):
        history, trainer = self._run(
            "FedBN",
            1,
            [[0, 1]],
        )
        self.assertEqual(len(trainer.bn_state_keys), 5)
        self.assertEqual(history["aggregation_excluded_key_count"], [5])
        self.assertEqual(
            history["aggregation_method"],
            ["sample_weighted_fedbn_shared_state"],
        )
        self.assertIn("central holdout", history["metrics"]["fedbn_evaluation_mode"])

    def test_fedpaq_uses_local_period_and_quantized_delta(self):
        history, _ = self._run(
            "FedPAQ",
            1,
            [[0, 1], [1, 2]],
        )
        self.assertEqual(history["scheduled_clients"], [[0, 1], [1, 2]])
        self.assertEqual(history["participating_clients"], [[0, 1], [1, 2]])
        self.assertEqual(history["is_aggregation_round"], [True, True])
        self.assertEqual(history["aggregation_weights"][0], [2 / 3, 1 / 3])
        self.assertEqual(history["aggregation_weights"][1], [1 / 4, 3 / 4])
        self.assertEqual(
            history["aggregation_method"][0],
            "sample_weighted_adapted_fedpaq",
        )
        self.assertEqual(len(history["fedpaq_quantization"][0]), 2)
        self.assertEqual(
            history["aggregation_config"]["fedpaq_local_update_steps"],
            2,
        )
        self.assertTrue(
            all(
                "quantizer_seed" in row
                for row in history["fedpaq_quantization"][0]
            )
        )


if __name__ == "__main__":
    unittest.main()
