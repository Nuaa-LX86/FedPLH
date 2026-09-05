import unittest
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from dataset.dataset_loader import (
    FULL_VOLUME_SCOPE,
    MedicalSegmentationDataset,
    build_partition_evidence,
    load_frozen_holdout_partitions,
    load_frozen_partitions,
    partition_holdout_by_training_profile,
    partition_data,
)
from models.precision_wrapper import HMPEPrecisionEmulator
from models.unet3d import UNet3D
from plot_beu_boundary import (
    compute_boundary,
    compute_credit_factor_sensitivity,
)
from simulator.acf_simulator import ACFSimulator
from training.acf_scheduler import ACFScheduler
from training.aggregation import (
    aggregate_fedpaq_deltas,
    batchnorm_state_keys,
    normalize_client_weights,
    weighted_reduce_states,
)
from training.dp_sgd import RDPAccountant
from utils.reproducibility import build_client_schedule


class IntegrityTests(unittest.TestCase):
    def test_participating_client_weights_use_training_sample_counts(self):
        weights = normalize_client_weights(
            [3, 1],
            [5, 20, 7, 40],
        )
        self.assertEqual(weights, [40 / 60, 20 / 60])
        self.assertAlmostEqual(sum(weights), 1.0, places=12)

    def test_weighted_reduction_and_excluded_state(self):
        states = [
            {
                "weight": torch.tensor([1.0, 3.0]),
                "counter": torch.tensor(2, dtype=torch.long),
                "private": torch.tensor([10.0]),
            },
            {
                "weight": torch.tensor([5.0, 7.0]),
                "counter": torch.tensor(6, dtype=torch.long),
                "private": torch.tensor([20.0]),
            },
        ]
        reference = {
            "weight": torch.zeros(2),
            "counter": torch.tensor(0, dtype=torch.long),
            "private": torch.tensor([99.0]),
        }
        reduced = weighted_reduce_states(
            states,
            [0.25, 0.75],
            reference_state=reference,
            excluded_keys={"private"},
        )
        self.assertTrue(
            torch.allclose(reduced["weight"], torch.tensor([4.0, 6.0]))
        )
        self.assertEqual(int(reduced["counter"]), 5)
        self.assertEqual(float(reduced["private"]), 99.0)

    def test_batchnorm_state_discovery_covers_parameters_and_buffers(self):
        model = HMPEPrecisionEmulator(
            UNet3D(n_channels=4, n_classes=4),
            default_precision="FP32",
        )
        keys = batchnorm_state_keys(model)
        self.assertEqual(len(keys), 70)
        self.assertIn("model.inc.double_conv.1.weight", keys)
        self.assertIn("model.inc.double_conv.1.bias", keys)
        self.assertIn("model.inc.double_conv.1.running_mean", keys)
        self.assertIn("model.inc.double_conv.1.running_var", keys)
        self.assertIn("model.inc.double_conv.1.num_batches_tracked", keys)

    def test_fedpaq_aggregates_quantized_model_deltas(self):
        base = {
            "weight": torch.tensor([0.0]),
            "counter": torch.tensor(0, dtype=torch.long),
        }
        clients = [
            {
                "weight": torch.tensor([1.0]),
                "counter": torch.tensor(2, dtype=torch.long),
            },
            {
                "weight": torch.tensor([3.0]),
                "counter": torch.tensor(6, dtype=torch.long),
            },
        ]
        result, stats = aggregate_fedpaq_deltas(
            base,
            clients,
            [0.25, 0.75],
            levels=255,
            generators=[
                torch.Generator().manual_seed(1),
                torch.Generator().manual_seed(2),
            ],
        )
        self.assertTrue(
            torch.allclose(result["weight"], torch.tensor([2.5]))
        )
        self.assertEqual(int(result["counter"]), 5)
        self.assertEqual(len(stats), 2)
        self.assertTrue(
            all(row["num_quantized_values"] == 1 for row in stats)
        )

    def test_full_volume_cache_counts_every_voxel(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_dir = root / "train" / "images"
            mask_dir = root / "train" / "masks"
            image_dir.mkdir(parents=True)
            mask_dir.mkdir(parents=True)
            np.save(
                image_dir / "sample.npy",
                np.zeros((4, 2, 2, 2), dtype=np.float32),
            )
            mask = np.array(
                [
                    [[0, 0], [0, 1]],
                    [[1, 2], [3, 0]],
                ],
                dtype=np.uint8,
            )
            np.save(mask_dir / "sample.npy", mask)

            dataset = MedicalSegmentationDataset(
                str(root),
                class_count_scope=FULL_VOLUME_SCOPE,
            )
            self.assertEqual(
                dataset.get_sample_class_counts(0),
                {0: 4, 1: 2, 2: 1, 3: 1},
            )
            cache_path = (
                root / "train"
            ).parent / "_class_counts_full_volume_v2_train.json"
            with cache_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual(payload["scope"], FULL_VOLUME_SCOPE)

    def test_partition_minimum_size_and_evidence(self):
        class FakeDataset:
            def __init__(self):
                self.counts = {}
                for sample_id in range(60):
                    dominant = sample_id % 3 + 1
                    counts = {0: 1000, 1: 10, 2: 10, 3: 10}
                    counts[dominant] = 100
                    self.counts[sample_id] = counts

            def get_sample_class_counts(self, sample_id):
                return self.counts[sample_id]

        dataset = FakeDataset()
        partitions, metadata = partition_data(
            dataset,
            range(60),
            num_clients=6,
            alpha=0.05,
            seed=7,
            min_client_samples=5,
            return_metadata=True,
        )
        self.assertGreaterEqual(min(map(len, partitions)), 5)
        evidence = build_partition_evidence(dataset, partitions, metadata)
        self.assertTrue(evidence.metadata["entropy_definition"]["background_excluded"])
        self.assertEqual(
            sum(client["num_samples"] for client in evidence),
            60,
        )

        balanced, _ = partition_data(
            dataset,
            range(60),
            num_clients=6,
            alpha=0.05,
            seed=7,
            min_client_samples=5,
            balance_client_sizes=True,
            return_metadata=True,
        )
        self.assertEqual(sorted(map(len, balanced)), [10] * 6)

        composition_balanced, metadata = partition_data(
            dataset,
            range(60),
            num_clients=6,
            alpha=0.05,
            seed=7,
            min_client_samples=5,
            balance_client_sizes=True,
            partition_basis="foreground_composition_quantiles",
            composition_bins=6,
            return_metadata=True,
        )
        self.assertEqual(sorted(map(len, composition_balanced)), [10] * 6)
        self.assertEqual(
            metadata["partition_type"],
            "dirichlet_patient_composition_strata",
        )

    def test_partition_seed_is_recorded_independently(self):
        class FakeDataset:
            def get_sample_class_counts(self, sample_id):
                label = int(sample_id) % 3 + 1
                return {0: 100, 1: 1, 2: 1, 3: 1, label: 20}

        _, first_metadata = partition_data(
            FakeDataset(),
            range(30),
            num_clients=3,
            alpha=0.5,
            seed=17,
            return_metadata=True,
        )
        _, second_metadata = partition_data(
            FakeDataset(),
            range(30),
            num_clients=3,
            alpha=0.5,
            seed=23,
            return_metadata=True,
        )
        self.assertEqual(first_metadata["partition_seed"], 17)
        self.assertEqual(second_metadata["partition_seed"], 23)

    def test_frozen_partition_requires_exact_training_coverage(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            partition_path = Path(temporary_directory) / "partition.json"
            partition_path.write_text(
                json.dumps({
                    "partition": {"partition_seed": 9, "alpha": 0.5},
                    "clients": [
                        {"client_id": 0, "sample_indices": [0, 2]},
                        {"client_id": 1, "sample_indices": [1, 3]},
                    ],
                }),
                encoding="utf-8",
            )
            partitions, metadata = load_frozen_partitions(
                partition_path,
                [0, 1, 2, 3],
                2,
            )
            self.assertEqual(partitions, [[0, 2], [1, 3]])
            self.assertEqual(metadata["partition_seed"], 9)
            self.assertEqual(
                metadata["partition_source"],
                "frozen_partition_evidence",
            )

    def test_frozen_holdout_requires_disjoint_exact_coverage(self):
        payload = {
            "evaluation_partitions": {
                "validation": [
                    {"client_id": 0, "sample_indices": [10, 12]},
                    {"client_id": 1, "sample_indices": [11]},
                ]
            }
        }
        partitions = load_frozen_holdout_partitions(
            payload,
            "validation",
            [10, 11, 12],
            2,
        )
        self.assertEqual(partitions, [[10, 12], [11]])
        with self.assertRaises(ValueError):
            load_frozen_holdout_partitions(
                payload,
                "validation",
                [10, 11, 13],
                2,
            )

    def test_personalized_holdout_partitions_are_disjoint_and_complete(self):
        class FakeDataset:
            def get_sample_class_counts(self, sample_id):
                label = int(sample_id) % 3 + 1
                counts = {0: 100, 1: 1, 2: 1, 3: 1}
                counts[label] = 20
                return counts

        client_stats = [
            {
                "num_samples": 10,
                "dominant_label_sample_counts": {
                    "0": 0, "1": 8, "2": 1, "3": 1,
                },
            },
            {
                "num_samples": 10,
                "dominant_label_sample_counts": {
                    "0": 0, "1": 1, "2": 8, "3": 1,
                },
            },
            {
                "num_samples": 10,
                "dominant_label_sample_counts": {
                    "0": 0, "1": 1, "2": 1, "3": 8,
                },
            },
        ]
        partitions = partition_holdout_by_training_profile(
            FakeDataset(),
            range(30, 42),
            client_stats,
            seed=5,
        )
        flattened = [value for partition in partitions for value in partition]
        self.assertEqual(sorted(flattened), list(range(30, 42)))
        self.assertEqual(len(flattened), len(set(flattened)))

    def test_client_schedule_is_deterministic(self):
        first = build_client_schedule(20, 80, 0.2, 3)
        second = build_client_schedule(20, 80, 0.2, 3)
        self.assertEqual(first, second)
        self.assertTrue(all(len(round_clients) == 4 for round_clients in first))

    def test_acf_rng_does_not_use_global_numpy_state(self):
        stats = [{"entropy": 1.0}, {"entropy": 0.5}]
        first = ACFScheduler(10, stats, seed=123)
        np.random.seed(999)
        plans_first = [
            first.get_execution_plan(0, round_index)["compute"]
            for round_index in range(10)
        ]

        second = ACFScheduler(10, stats, seed=123)
        np.random.seed(1)
        plans_second = [
            second.get_execution_plan(0, round_index)["compute"]
            for round_index in range(10)
        ]
        self.assertEqual(plans_first, plans_second)

    def test_rdp_accountant_is_monotonic(self):
        accountant = RDPAccountant()
        epsilon = []
        for _ in range(5):
            accountant.add_segment(0.1, 1.0, 1)
            epsilon.append(accountant.get_epsilon(1e-5))
        self.assertTrue(
            all(current >= previous for previous, current in zip(epsilon, epsilon[1:]))
        )

    def test_boundary_uses_explicit_30x_point(self):
        result = compute_boundary(1.254e9, 5.188e7, 1408.0, 35.0)
        self.assertAlmostEqual(
            result["coverage_threshold_multiplier"],
            24.171164225134926,
        )
        self.assertAlmostEqual(result["visible_cost_ms_at_30x"], 214.77272727272728)

    def test_credit_factor_sensitivity_uses_round_aligned_histories(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            paths = []
            for seed, ratios in enumerate(([0.05, 0.10], [0.15, 0.20])):
                path = root / f"seed{seed}" / "training_history.json"
                path.parent.mkdir(parents=True)
                payload = {
                    "round": [0, 1],
                    "delta_c_cycles": [100.0, 100.0],
                    "c_priv_cycles": [ratio * 100.0 for ratio in ratios],
                }
                path.write_text(json.dumps(payload), encoding="utf-8")
                paths.append(path)

            result = compute_credit_factor_sensitivity(paths)
            original_hash = result["inputs"][0]["sha256"]
            payload = json.loads(paths[0].read_text(encoding="utf-8"))
            payload["unrelated_metric"] = [999.0, 999.0]
            paths[0].write_text(json.dumps(payload), encoding="utf-8")
            projected_result = compute_credit_factor_sensitivity(paths)

        self.assertEqual(result["seed_count"], 2)
        self.assertEqual(result["record_count"], 4)
        self.assertAlmostEqual(result["median_required_credit_factor"], 0.125)
        self.assertAlmostEqual(result["maximum_required_credit_factor"], 0.20)
        self.assertEqual(len(result["inputs"]), 2)
        self.assertTrue(all(len(item["sha256"]) == 64 for item in result["inputs"]))
        self.assertEqual(projected_result["inputs"][0]["sha256"], original_hash)

    def test_credit_factor_sensitivity_accepts_public_neutral_fields(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "seed0" / "training_history.json"
            path.parent.mkdir(parents=True)
            path.write_text(
                json.dumps(
                    {
                        "round": [0, 1],
                        "profiled_timing_slack_cycles": [100.0, 200.0],
                        "operator_cost_cycles": [5.0, 20.0],
                    }
                ),
                encoding="utf-8",
            )
            result = compute_credit_factor_sensitivity([path])

        self.assertEqual(result["record_count"], 2)
        self.assertAlmostEqual(result["minimum_required_credit_factor"], 0.05)
        self.assertAlmostEqual(result["maximum_required_credit_factor"], 0.10)

    def test_credit_factor_sensitivity_rejects_invalid_cycles(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "seed0" / "training_history.json"
            path.parent.mkdir(parents=True)
            path.write_text(
                json.dumps(
                    {
                        "round": [0],
                        "delta_c_cycles": [0.0],
                        "c_priv_cycles": [1.0],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "non-positive"):
                compute_credit_factor_sensitivity([path])

    def test_credit_factor_sensitivity_rejects_missing_fields(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "seed0" / "training_history.json"
            path.parent.mkdir(parents=True)
            path.write_text(
                json.dumps({"round": [0], "delta_c_cycles": [1.0]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "missing required fields"):
                compute_credit_factor_sensitivity([path])

    def test_credit_factor_sensitivity_rejects_unexpected_seed_or_round_count(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "seed0" / "training_history.json"
            path.parent.mkdir(parents=True)
            path.write_text(
                json.dumps(
                    {
                        "round": [0, 1],
                        "delta_c_cycles": [100.0, 100.0],
                        "c_priv_cycles": [5.0, 5.0],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Expected 2 seed histories"):
                compute_credit_factor_sensitivity(
                    [path],
                    expected_seed_count=2,
                    expected_round_count=2,
                )
            with self.assertRaisesRegex(ValueError, "Expected 3 rounds"):
                compute_credit_factor_sensitivity(
                    [path],
                    expected_seed_count=1,
                    expected_round_count=3,
                )

    def test_fixed_resource_aggregation_scales_with_clients(self):
        simulator = ACFSimulator("hardware_profile.json")
        latency_10 = simulator.simulate_aggregation(10, 50, "PEC")
        latency_20 = simulator.simulate_aggregation(20, 50, "PEC")
        pec = simulator.hw_profile["federation_costs"]["PEC_hardware"]
        server_clock_mhz = float(
            pec.get("clock_frequency_MHz", simulator.server_clock_freq_mhz)
        )
        pipeline_ms = float(pec["pipeline_depth"]) / (server_clock_mhz * 1e6) * 1e3
        self.assertAlmostEqual(
            latency_20 - pipeline_ms,
            2.0 * (latency_10 - pipeline_ms),
        )

    def test_sac_uses_profile_derived_bottleneck(self):
        simulator = ACFSimulator("hardware_profile.json")
        pec = simulator.hw_profile["federation_costs"]["PEC_hardware"]
        server_clock_mhz = float(
            pec.get("clock_frequency_MHz", simulator.server_clock_freq_mhz)
        )
        lane_bandwidth_gbps = (
            float(pec["throughput_bytes_per_cycle"])
            * int(pec.get("parallel_lanes", 1))
            * server_clock_mhz
            / 1000.0
        )
        effective_bandwidth_gbps = min(lane_bandwidth_gbps, simulator.mem_bw_gbps)
        expected_ms = (
            10 * 1000 * 1024 * 1024 / (effective_bandwidth_gbps * 1e9) * 1e3
            + float(pec["pipeline_depth"]) / (server_clock_mhz * 1e6) * 1e3
        )
        actual_ms = simulator.simulate_aggregation(1000, 10, "PEC")
        self.assertAlmostEqual(actual_ms, expected_ms, places=12)

    def test_software_conversion_factor_respects_sac_bottleneck(self):
        simulator = ACFSimulator("hardware_profile.json")
        software = simulator.hw_profile["federation_costs"]["software_baseline"]
        self.assertEqual(software["software_conversion_factor"], 12.0)
        self.assertNotIn("instruction_overhead_per_byte", software)
        pec = simulator.hw_profile["federation_costs"]["PEC_hardware"]
        server_clock_mhz = float(
            pec.get("clock_frequency_MHz", simulator.server_clock_freq_mhz)
        )
        lane_bandwidth_gbps = (
            float(pec["throughput_bytes_per_cycle"])
            * int(pec.get("parallel_lanes", 1))
            * server_clock_mhz
            / 1000.0
        )
        expected_ratio = 12.0 * min(
            lane_bandwidth_gbps, simulator.mem_bw_gbps
        ) / simulator.mem_bw_gbps
        sac_ms = simulator.simulate_aggregation(1000, 10, "PEC")
        cpu_ms = simulator.simulate_aggregation(1000, 10, "Software")
        self.assertAlmostEqual(cpu_ms / sac_ms, expected_ratio, delta=0.001)


if __name__ == "__main__":
    unittest.main()
