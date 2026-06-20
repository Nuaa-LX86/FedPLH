import csv
import json
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from utils.reproducibility import derive_seed, make_torch_generator


CLASS_COUNT_SCHEMA_VERSION = 2
FULL_VOLUME_SCOPE = "full_volume_foreground"
LEGACY_CENTER_SCOPE = "legacy_center_slices"


def calculate_shannon_entropy(
    class_counts: Dict[int, int],
    classes: Optional[Sequence[int]] = None,
    eps: float = 1e-12,
) -> float:
    if classes is None:
        classes = sorted(class_counts)
    total = float(sum(float(class_counts.get(int(c), 0)) for c in classes))
    if total <= 0:
        return 0.0
    entropy = 0.0
    for class_id in classes:
        probability = float(class_counts.get(int(class_id), 0)) / total
        if probability > 0:
            entropy -= probability * np.log2(probability + eps)
    return float(entropy)


def _cache_path(data_root: str, split: str, scope: str) -> Path:
    if scope == FULL_VOLUME_SCOPE:
        return (
            Path(data_root)
            / f"_class_counts_full_volume_v{CLASS_COUNT_SCHEMA_VERSION}_{split}.json"
        )
    if scope == LEGACY_CENTER_SCOPE:
        return Path(data_root) / f"_class_counts_cache_{split}.json"
    raise ValueError(f"Unsupported class-count scope: {scope}")


def _load_cache(cache_file: Path) -> Optional[Dict]:
    if not cache_file.exists():
        return None
    try:
        with cache_file.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def _save_cache(cache_file: Path, data: Dict) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_file.with_suffix(cache_file.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle)
    temporary.replace(cache_file)


class SyntheticMedicalDataset(Dataset):
    def __init__(
        self,
        n=200,
        img_size=64,
        n_classes=4,
        seed=0,
        alpha_dirichlet=0.5,
    ):
        self.n = int(n)
        self.img_size = int(img_size)
        self.n_classes = int(n_classes)
        self.rng = np.random.RandomState(seed)
        weights = self.rng.dirichlet(
            alpha_dirichlet * np.ones(3),
            size=self.n,
        )
        self.dominant_labels = np.array(
            [int(np.argmax(weight)) + 1 for weight in weights],
            dtype=np.int64,
        )
        size = self.img_size
        margin = size // 6
        self.tumor_centers = self.rng.randint(
            size // 2 - margin,
            size // 2 + margin + 1,
            size=(self.n, 3),
        )
        self.outer_radii = self.rng.uniform(0.20 * size, 0.28 * size, size=self.n)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        rng = np.random.RandomState(idx)
        size = self.img_size
        label = np.zeros((size, size, size), dtype=np.int64)
        cx, cy, cz = self.tumor_centers[idx]
        outer_radius = float(self.outer_radii[idx])
        shell_radius = outer_radius * rng.uniform(0.55, 0.65)
        core_radius = outer_radius * rng.uniform(0.30, 0.40)
        axes = rng.uniform(0.7, 1.3, size=3)
        zz, yy, xx = np.mgrid[0:size, 0:size, 0:size]
        distance = np.sqrt(
            ((zz - cz) / axes[0]) ** 2
            + ((yy - cy) / axes[1]) ** 2
            + ((xx - cx) / axes[2]) ** 2
        )
        label[distance <= outer_radius] = 2
        label[distance <= shell_radius] = 3
        label[distance <= core_radius] = 1
        noisy_margin = (
            (distance > outer_radius * 0.85)
            & (distance < outer_radius * 1.15)
        )
        label[noisy_margin] = rng.randint(0, 3, size=noisy_margin.sum())

        channels = []
        for background, ncr, edema, enhancing in [
            (0, 0.6, 0.3, 0.8),
            (0, 0.2, 0.4, 1),
            (0, 0.5, 1, 0.7),
            (0, 0.4, 1, 0.6),
        ]:
            channel = np.full(
                (size, size, size),
                background,
                dtype=np.float32,
            )
            channel[label == 1] = ncr
            channel[label == 2] = edema
            channel[label == 3] = enhancing
            channel += rng.randn(size, size, size).astype(np.float32) * 0.25
            channel = (channel - channel.mean()) / (channel.std() + 1e-6)
            channels.append(channel)
        return (
            torch.from_numpy(np.stack(channels, 0)).float(),
            torch.from_numpy(label).long(),
        )

    def get_sample_class_counts(self, idx: int) -> Dict[int, int]:
        dominant = int(self.dominant_labels[idx])
        total = self.img_size ** 3
        return {
            0: int(total * 0.78),
            1: int(total * 0.07) if dominant == 1 else int(total * 0.03),
            2: int(total * 0.10) if dominant == 2 else int(total * 0.05),
            3: int(total * 0.05) if dominant == 3 else int(total * 0.02),
        }


class MedicalSegmentationDataset(Dataset):
    """BraTS dataset with versioned per-volume class-count caching."""

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        img_size: int = 64,
        class_count_scope: str = FULL_VOLUME_SCOPE,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.img_size = int(img_size)
        self.class_count_scope = str(class_count_scope)

        image_dir = self.data_root / split / "images"
        mask_dir = self.data_root / split / "masks"
        if not image_dir.exists() or not mask_dir.exists():
            raise FileNotFoundError(f"Not found: {image_dir} or {mask_dir}")

        self.image_paths = sorted(image_dir.glob("*.npy"))
        self.mask_paths = [mask_dir / path.name for path in self.image_paths]
        for mask_path in self.mask_paths:
            if not mask_path.exists():
                raise FileNotFoundError(f"Missing mask: {mask_path}")

        self._cc_cache: Dict[int, Dict[int, int]] = {}
        cache_file = _cache_path(
            str(data_root),
            split,
            self.class_count_scope,
        )
        cached = _load_cache(cache_file)
        cached_counts = None
        if (
            self.class_count_scope == FULL_VOLUME_SCOPE
            and isinstance(cached, dict)
            and cached.get("schema_version") == CLASS_COUNT_SCHEMA_VERSION
            and cached.get("scope") == FULL_VOLUME_SCOPE
            and cached.get("sample_names") == [p.name for p in self.mask_paths]
        ):
            cached_counts = cached.get("counts", {})
        elif (
            self.class_count_scope == LEGACY_CENTER_SCOPE
            and isinstance(cached, dict)
        ):
            cached_counts = cached

        if cached_counts is not None and len(cached_counts) == len(self.image_paths):
            for key, values in cached_counts.items():
                self._cc_cache[int(key)] = {
                    int(class_id): int(count)
                    for class_id, count in values.items()
                }
            print(
                f"[Cache] Loaded {self.class_count_scope} counts for "
                f"{len(self._cc_cache)} samples"
            )
        else:
            self._build_class_count_cache(cache_file)

    def _build_class_count_cache(self, cache_file: Path) -> None:
        total = len(self.mask_paths)
        print(
            f"[Cache] Scanning {total} masks for "
            f"{self.class_count_scope} counts..."
        )
        for index, mask_path in enumerate(self.mask_paths):
            if index % 100 == 0:
                print(f"        {index}/{total}...", end="\r", flush=True)
            mask = np.load(mask_path, mmap_mode="r")
            if self.class_count_scope == FULL_VOLUME_SCOPE:
                counts = np.bincount(
                    np.asarray(mask).reshape(-1),
                    minlength=4,
                )
                nonzero_labels = np.flatnonzero(counts)
                if len(nonzero_labels) and int(nonzero_labels[-1]) > 3:
                    raise ValueError(
                        f"Mask {mask_path} contains labels outside 0..3"
                    )
                self._cc_cache[index] = {
                    class_id: int(counts[class_id])
                    for class_id in range(4)
                }
            else:
                depth = int(mask.shape[0])
                centers = [
                    max(0, depth // 2 - 1),
                    depth // 2,
                    min(depth - 1, depth // 2 + 1),
                ]
                sample_counts: Dict[int, int] = {}
                for slice_index in centers:
                    labels, counts = np.unique(
                        mask[slice_index],
                        return_counts=True,
                    )
                    for label, count in zip(labels, counts):
                        sample_counts[int(label)] = (
                            sample_counts.get(int(label), 0) + int(count)
                        )
                self._cc_cache[index] = sample_counts

        serialized = {
            str(index): {
                str(class_id): int(count)
                for class_id, count in counts.items()
            }
            for index, counts in self._cc_cache.items()
        }
        if self.class_count_scope == FULL_VOLUME_SCOPE:
            payload = {
                "schema_version": CLASS_COUNT_SCHEMA_VERSION,
                "scope": FULL_VOLUME_SCOPE,
                "classes": [0, 1, 2, 3],
                "sample_names": [p.name for p in self.mask_paths],
                "counts": serialized,
            }
        else:
            payload = serialized
        _save_cache(cache_file, payload)
        print(f"\n[Cache] Saved {cache_file.name}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_array = np.load(self.image_paths[idx], mmap_mode="r")
        mask_array = np.load(self.mask_paths[idx], mmap_mode="r")
        image = torch.from_numpy(np.array(image_array, copy=True)).float()
        mask = torch.from_numpy(np.array(mask_array, copy=True)).long()
        if int(mask.max().item()) > 3:
            raise ValueError("Mask label > 3; remap label 4 to 3 first.")

        image = F.interpolate(
            image.unsqueeze(0),
            size=(self.img_size,) * 3,
            mode="trilinear",
            align_corners=False,
        ).squeeze(0)
        mask = F.interpolate(
            mask.unsqueeze(0).unsqueeze(0).float(),
            size=(self.img_size,) * 3,
            mode="nearest",
        ).squeeze(0).squeeze(0).long()
        return image, mask

    def get_sample_class_counts(self, idx: int) -> Dict[int, int]:
        return dict(self._cc_cache.get(idx, {0: 1}))


def _dominant_tumor_label(class_counts: Dict[int, int]) -> int:
    foreground = {
        class_id: int(class_counts.get(class_id, 0))
        for class_id in (1, 2, 3)
    }
    if sum(foreground.values()) <= 0:
        return 0
    return int(max(foreground, key=foreground.get))


def _composition_quantile_labels(dataset, indices, num_bins):
    indices = [int(sample_id) for sample_id in indices]
    features = []
    for sample_id in indices:
        counts = dataset.get_sample_class_counts(sample_id)
        foreground = np.array(
            [float(counts.get(class_id, 0)) for class_id in (1, 2, 3)],
            dtype=np.float64,
        )
        foreground /= max(float(foreground.sum()), 1e-12)
        features.append(foreground)
    features = np.asarray(features, dtype=np.float64)
    candidates = {
        "label_1_fraction": features[:, 0],
        "label_2_fraction": features[:, 1],
        "label_3_fraction": features[:, 2],
        "label_2_minus_label_1": features[:, 1] - features[:, 0],
        "label_3_minus_label_1": features[:, 2] - features[:, 0],
        "label_3_minus_label_2": features[:, 2] - features[:, 1],
    }
    projection_name = max(
        candidates,
        key=lambda name: float(np.var(candidates[name])),
    )
    scores = candidates[projection_name]
    if np.allclose(scores, scores[0]):
        scores = np.arange(len(indices), dtype=np.float64)
        projection_name = "stable_sample_order_fallback"

    num_bins = max(2, min(int(num_bins), len(indices)))
    order = np.argsort(scores, kind="mergesort")
    labels = {}
    score_ranges = []
    for bin_id, positions in enumerate(np.array_split(order, num_bins)):
        bin_scores = scores[positions]
        score_ranges.append({
            "bin_id": int(bin_id),
            "num_samples": int(len(positions)),
            "score_min": float(np.min(bin_scores)),
            "score_max": float(np.max(bin_scores)),
        })
        for position in positions:
            labels[indices[int(position)]] = int(bin_id)
    return labels, {
        "projection": projection_name,
        "score_ranges": score_ranges,
    }


def _rebalance_minimum_size(
    partitions,
    sample_labels,
    target_class_counts,
    min_client_samples,
    balance_client_sizes,
):
    current_counts = np.zeros_like(target_class_counts, dtype=np.int64)
    for client_id, partition in enumerate(partitions):
        for sample_id in partition:
            current_counts[sample_labels[int(sample_id)], client_id] += 1

    if balance_client_sizes:
        total_samples = sum(len(partition) for partition in partitions)
        base_size, remainder = divmod(total_samples, len(partitions))
        target_sizes = np.array(
            [
                base_size + (1 if client_id < remainder else 0)
                for client_id in range(len(partitions))
            ],
            dtype=np.int64,
        )
    else:
        target_sizes = np.full(
            len(partitions),
            int(min_client_samples),
            dtype=np.int64,
        )

    moves = 0
    while any(
        len(partition) < int(target_sizes[client_id])
        for client_id, partition in enumerate(partitions)
    ):
        deficits = [
            int(target_sizes[client_id]) - len(partition)
            for client_id, partition in enumerate(partitions)
        ]
        receiver = int(np.argmax(deficits))
        donors = [
            client_id
            for client_id, partition in enumerate(partitions)
            if len(partition) > int(target_sizes[client_id])
        ]
        if not donors:
            raise RuntimeError("Unable to satisfy client-size constraints")

        best_candidate = None
        for donor in donors:
            for position, sample_id in enumerate(partitions[donor]):
                label = sample_labels[int(sample_id)]
                receiver_deficit = (
                    target_class_counts[label, receiver]
                    - current_counts[label, receiver]
                )
                donor_surplus = (
                    current_counts[label, donor]
                    - target_class_counts[label, donor]
                )
                candidate = (
                    float(receiver_deficit + donor_surplus),
                    len(partitions[donor]),
                    -int(sample_id),
                    donor,
                    position,
                    label,
                )
                if best_candidate is None or candidate > best_candidate:
                    best_candidate = candidate

        _, _, _, donor, position, label = best_candidate
        sample_id = partitions[donor].pop(position)
        partitions[receiver].append(sample_id)
        current_counts[label, donor] -= 1
        current_counts[label, receiver] += 1
        moves += 1

    return partitions, moves


def partition_data(
    dataset,
    indices,
    num_clients,
    iid=False,
    alpha=0.5,
    seed=0,
    min_client_samples=1,
    balance_client_sizes=False,
    partition_basis="dominant_foreground_label",
    composition_bins=10,
    return_metadata=False,
):
    rng = np.random.RandomState(seed)
    indices = np.array(list(indices), dtype=np.int64)
    min_client_samples = int(min_client_samples)
    if min_client_samples < 1:
        raise ValueError("min_client_samples must be at least 1")
    if min_client_samples * int(num_clients) > len(indices):
        raise ValueError(
            "min_client_samples * num_clients exceeds the training-set size"
        )

    composition_bin_ranges = None
    if partition_basis == "foreground_composition_quantiles":
        sample_labels, composition_bin_ranges = _composition_quantile_labels(
            dataset,
            indices,
            composition_bins,
        )
    elif partition_basis == "dominant_foreground_label":
        sample_labels = {}
        for sample_id in indices:
            if hasattr(dataset, "dominant_labels"):
                label = int(dataset.dominant_labels[int(sample_id)])
            else:
                label = _dominant_tumor_label(
                    dataset.get_sample_class_counts(int(sample_id))
                )
            sample_labels[int(sample_id)] = label
    else:
        raise ValueError(f"Unsupported partition_basis: {partition_basis}")

    if iid:
        rng.shuffle(indices)
        partitions = [
            split.tolist()
            for split in np.array_split(indices, num_clients)
        ]
        metadata = {
            "partition_type": "iid",
            "partition_seed": int(seed),
            "alpha": None,
            "min_client_samples": min_client_samples,
            "balance_client_sizes": True,
            "partition_basis": str(partition_basis),
            "rebalance_moves": 0,
        }
        return (partitions, metadata) if return_metadata else partitions

    labels = np.array(
        [sample_labels[int(sample_id)] for sample_id in indices],
        dtype=np.int64,
    )
    num_partition_classes = int(labels.max()) + 1
    indices_by_class = [
        indices[labels == class_id]
        for class_id in range(num_partition_classes)
    ]
    for class_indices in indices_by_class:
        rng.shuffle(class_indices)

    client_indices = [[] for _ in range(num_clients)]
    target_class_counts = np.zeros(
        (num_partition_classes, num_clients),
        dtype=np.float64,
    )
    for class_id, class_indices in enumerate(indices_by_class):
        if len(class_indices) == 0:
            continue
        proportions = rng.dirichlet(alpha * np.ones(num_clients))
        raw_counts = proportions * len(class_indices)
        counts = np.floor(raw_counts).astype(int)
        remainder = int(len(class_indices) - counts.sum())
        if remainder:
            order = np.argsort(-(raw_counts - counts))
            counts[order[:remainder]] += 1
        target_class_counts[class_id] = raw_counts

        start = 0
        for client_id, count in enumerate(counts):
            take = int(count)
            if take:
                client_indices[client_id].extend(
                    class_indices[start:start + take].tolist()
                )
            start += take

    client_indices, rebalance_moves = _rebalance_minimum_size(
        client_indices,
        sample_labels,
        target_class_counts,
        min_client_samples,
        bool(balance_client_sizes),
    )
    for partition in client_indices:
        rng.shuffle(partition)

    metadata = {
        "partition_type": (
            "dirichlet_patient_composition_strata"
            if partition_basis == "foreground_composition_quantiles"
            else "dirichlet_dominant_foreground_label"
        ),
        "partition_seed": int(seed),
        "alpha": float(alpha),
        "min_client_samples": min_client_samples,
        "balance_client_sizes": bool(balance_client_sizes),
        "partition_basis": str(partition_basis),
        "composition_bins": (
            int(composition_bins)
            if partition_basis == "foreground_composition_quantiles"
            else None
        ),
        "composition_bin_score_ranges": composition_bin_ranges,
        "rebalance_moves": int(rebalance_moves),
    }
    return (client_indices, metadata) if return_metadata else client_indices


def _jensen_shannon_bits(first, second):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first /= max(float(first.sum()), 1e-12)
    second /= max(float(second.sum()), 1e-12)
    midpoint = 0.5 * (first + second)

    def _kl(values, reference):
        mask = values > 0
        return float(
            np.sum(values[mask] * np.log2(values[mask] / reference[mask]))
        )

    return 0.5 * _kl(first, midpoint) + 0.5 * _kl(second, midpoint)


class ClientStats(list):
    def __init__(
        self,
        values,
        metadata,
        client_val_loaders=None,
        client_test_loaders=None,
    ):
        super().__init__(values)
        self.metadata = metadata
        self.client_val_loaders = client_val_loaders
        self.client_test_loaders = client_test_loaders


def partition_holdout_by_training_profile(
    dataset,
    holdout_indices,
    client_stats,
    seed,
):
    num_clients = len(client_stats)
    rng = np.random.default_rng(int(seed))
    grouped = {class_id: [] for class_id in range(4)}
    for sample_id in holdout_indices:
        dominant_label = _dominant_tumor_label(
            dataset.get_sample_class_counts(int(sample_id))
        )
        grouped[dominant_label].append(int(sample_id))

    partitions = [[] for _ in range(num_clients)]
    for class_id, sample_ids in grouped.items():
        if not sample_ids:
            continue
        rng.shuffle(sample_ids)
        profile = np.asarray(
            [
                float(
                    client["dominant_label_sample_counts"].get(
                        str(class_id),
                        0,
                    )
                )
                for client in client_stats
            ],
            dtype=np.float64,
        )
        if float(profile.sum()) <= 0:
            profile = np.asarray(
                [
                    float(client.get("num_samples", 0))
                    for client in client_stats
                ],
                dtype=np.float64,
            )
        profile /= max(float(profile.sum()), 1e-12)
        raw_counts = profile * len(sample_ids)
        counts = np.floor(raw_counts).astype(np.int64)
        remainder = int(len(sample_ids) - counts.sum())
        if remainder:
            order = np.argsort(-(raw_counts - counts), kind="mergesort")
            counts[order[:remainder]] += 1

        offset = 0
        for client_id, count in enumerate(counts):
            take = int(count)
            partitions[client_id].extend(sample_ids[offset:offset + take])
            offset += take

    flattened = [
        sample_id for partition in partitions for sample_id in partition
    ]
    if len(flattened) != len(set(flattened)):
        raise AssertionError("Holdout partition contains duplicate samples")
    if set(flattened) != set(int(index) for index in holdout_indices):
        raise AssertionError("Holdout partition does not preserve the sample set")
    return partitions


def build_partition_evidence(dataset, partitions, partition_metadata):
    clients = []
    global_counts = np.zeros(4, dtype=np.int64)
    distributions = []

    for client_id, sample_ids in enumerate(partitions):
        counts = np.zeros(4, dtype=np.int64)
        dominant_counts = np.zeros(4, dtype=np.int64)
        empty_foreground_samples = 0
        for sample_id in sample_ids:
            sample_counts = dataset.get_sample_class_counts(int(sample_id))
            sample_array = np.array(
                [
                    int(sample_counts.get(class_id, 0))
                    for class_id in range(4)
                ],
                dtype=np.int64,
            )
            counts += sample_array
            if int(sample_array[1:].sum()) == 0:
                empty_foreground_samples += 1
                dominant_label = 0
            else:
                dominant_label = int(np.argmax(sample_array[1:]) + 1)
            dominant_counts[dominant_label] += 1

        global_counts += counts
        foreground_counts = counts[1:]
        foreground_total = int(foreground_counts.sum())
        if foreground_total:
            proportions = foreground_counts / foreground_total
        else:
            proportions = np.zeros(3, dtype=np.float64)
        distributions.append(proportions)
        total_voxels = int(counts.sum())

        clients.append({
            "client_id": int(client_id),
            "entropy": float(
                calculate_shannon_entropy(
                    {
                        class_id: int(counts[class_id])
                        for class_id in range(4)
                    },
                    classes=[1, 2, 3],
                )
            ),
            "num_samples": int(len(sample_ids)),
            "class_counts": {
                str(class_id): int(counts[class_id])
                for class_id in range(4)
            },
            "foreground_proportions": {
                str(class_id): float(proportions[class_id - 1])
                for class_id in (1, 2, 3)
            },
            "background_fraction": (
                float(counts[0] / total_voxels) if total_voxels else 0.0
            ),
            "empty_foreground_samples": int(empty_foreground_samples),
            "empty_foreground_fraction": (
                float(empty_foreground_samples / len(sample_ids))
                if sample_ids
                else 0.0
            ),
            "dominant_label_sample_counts": {
                str(class_id): int(dominant_counts[class_id])
                for class_id in range(4)
            },
            "sample_indices": [int(sample_id) for sample_id in sample_ids],
        })

    global_foreground = global_counts[1:].astype(np.float64)
    global_foreground /= max(float(global_foreground.sum()), 1e-12)
    client_global_jsd = [
        _jensen_shannon_bits(distribution, global_foreground)
        for distribution in distributions
    ]
    pairwise_jsd = [
        _jensen_shannon_bits(distributions[first], distributions[second])
        for first in range(len(distributions))
        for second in range(first + 1, len(distributions))
    ]
    sample_counts = np.array(
        [len(partition) for partition in partitions],
        dtype=np.int64,
    )
    entropies = np.array(
        [client["entropy"] for client in clients],
        dtype=np.float64,
    )

    evidence = {
        "schema_version": 1,
        "entropy_definition": {
            "scope": "complete 3D mask",
            "labels": (
                "original mutually exclusive segmentation labels 1, 2, and 3"
            ),
            "background_excluded": True,
            "aggregation_unit": (
                "foreground voxels over all local patient volumes"
            ),
            "empty_foreground_handling": (
                "entropy=0; empty samples counted separately"
            ),
        },
        "partition": dict(partition_metadata),
        "num_clients": int(len(partitions)),
        "num_train_samples": int(sample_counts.sum()),
        "sample_count_summary": {
            "min": int(sample_counts.min()),
            "mean": float(sample_counts.mean()),
            "max": int(sample_counts.max()),
            "std": float(sample_counts.std(ddof=0)),
        },
        "entropy_summary": {
            "min": float(entropies.min()),
            "mean": float(entropies.mean()),
            "max": float(entropies.max()),
            "std": float(entropies.std(ddof=0)),
        },
        "global_class_counts": {
            str(class_id): int(global_counts[class_id])
            for class_id in range(4)
        },
        "global_foreground_proportions": {
            str(class_id): float(global_foreground[class_id - 1])
            for class_id in (1, 2, 3)
        },
        "realized_heterogeneity": {
            "mean_client_global_jsd_bits": float(
                np.mean(client_global_jsd)
            ),
            "max_client_global_jsd_bits": float(
                np.max(client_global_jsd)
            ),
            "mean_pairwise_jsd_bits": (
                float(np.mean(pairwise_jsd)) if pairwise_jsd else 0.0
            ),
            "max_pairwise_jsd_bits": (
                float(np.max(pairwise_jsd)) if pairwise_jsd else 0.0
            ),
        },
        "clients": clients,
    }
    return ClientStats(clients, evidence)


def write_partition_evidence(evidence, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "partition_evidence.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(evidence, handle, indent=2)

    csv_path = output_dir / "client_distribution.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "client_id",
                "num_samples",
                "entropy",
                "foreground_class_1",
                "foreground_class_2",
                "foreground_class_3",
                "background_fraction",
                "empty_foreground_fraction",
            ],
        )
        writer.writeheader()
        for client in evidence["clients"]:
            writer.writerow({
                "client_id": client["client_id"],
                "num_samples": client["num_samples"],
                "entropy": client["entropy"],
                "foreground_class_1": (
                    client["foreground_proportions"]["1"]
                ),
                "foreground_class_2": (
                    client["foreground_proportions"]["2"]
                ),
                "foreground_class_3": (
                    client["foreground_proportions"]["3"]
                ),
                "background_fraction": client["background_fraction"],
                "empty_foreground_fraction": (
                    client["empty_foreground_fraction"]
                ),
            })
    return json_path, csv_path


def load_frozen_partitions(
    partition_file,
    train_indices,
    num_clients,
    include_payload=False,
):
    partition_path = Path(partition_file)
    if not partition_path.is_file():
        raise FileNotFoundError(f"Frozen partition not found: {partition_path}")
    with partition_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    clients = payload.get("clients")
    if not isinstance(clients, list) or len(clients) != int(num_clients):
        raise ValueError(
            "Frozen partition client count does not match num_clients"
        )
    clients_by_id = {
        int(client["client_id"]): [
            int(sample_id) for sample_id in client.get("sample_indices", [])
        ]
        for client in clients
    }
    if set(clients_by_id) != set(range(int(num_clients))):
        raise ValueError("Frozen partition client IDs must be contiguous")

    partitions = [
        clients_by_id[client_id]
        for client_id in range(int(num_clients))
    ]
    flattened = [
        sample_id
        for partition in partitions
        for sample_id in partition
    ]
    expected = [int(sample_id) for sample_id in train_indices]
    if len(flattened) != len(set(flattened)):
        raise ValueError("Frozen partition contains duplicate sample indices")
    if set(flattened) != set(expected):
        missing = sorted(set(expected) - set(flattened))
        unexpected = sorted(set(flattened) - set(expected))
        raise ValueError(
            "Frozen partition does not exactly cover the training split: "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}"
        )

    metadata = dict(payload.get("partition", {}))
    metadata.update({
        "partition_source": "frozen_partition_evidence",
        "frozen_partition_file": str(partition_path.resolve()),
    })
    if include_payload:
        return partitions, metadata, payload
    return partitions, metadata


def load_frozen_holdout_partitions(
    payload,
    split_name,
    expected_indices,
    num_clients,
):
    evaluation = payload.get("evaluation_partitions", {})
    entries = evaluation.get(split_name)
    if entries is None:
        return None
    if not isinstance(entries, list) or len(entries) != int(num_clients):
        raise ValueError(
            f"Frozen {split_name} partition client count does not match"
        )
    entries_by_id = {
        int(entry["client_id"]): [
            int(sample_id)
            for sample_id in entry.get("sample_indices", [])
        ]
        for entry in entries
    }
    if set(entries_by_id) != set(range(int(num_clients))):
        raise ValueError(
            f"Frozen {split_name} client IDs must be contiguous"
        )
    partitions = [
        entries_by_id[client_id]
        for client_id in range(int(num_clients))
    ]
    flattened = [
        sample_id
        for partition in partitions
        for sample_id in partition
    ]
    expected = [int(sample_id) for sample_id in expected_indices]
    if len(flattened) != len(set(flattened)):
        raise ValueError(
            f"Frozen {split_name} partition contains duplicate samples"
        )
    if set(flattened) != set(expected):
        raise ValueError(
            f"Frozen {split_name} partition does not exactly cover its split"
        )
    return partitions


def _make_global_split(n, val_ratio, test_ratio, seed):
    rng = np.random.RandomState(seed)
    indices = np.arange(n, dtype=np.int64)
    rng.shuffle(indices)
    num_test = int(round(n * test_ratio))
    num_val = int(round(n * val_ratio))
    num_train = max(1, n - num_val - num_test)
    return (
        indices[:num_train].tolist(),
        indices[num_train:num_train + num_val].tolist(),
        indices[num_train + num_val:].tolist(),
    )


def get_federated_dataloaders(
    num_clients=10,
    batch_size=2,
    img_size=64,
    iid=False,
    alpha=0.5,
    data_root="./data/BraTS2021",
    val_ratio=0.1,
    test_ratio=0.1,
    split_seed=0,
    partition_seed=None,
    partition_file=None,
    force_split=False,
    loader_seed=0,
    allow_synthetic=False,
    entropy_scope=FULL_VOLUME_SCOPE,
    min_client_samples=1,
    balance_client_sizes=False,
    partition_basis="dominant_foreground_label",
    composition_bins=10,
):
    resolved_partition_seed = (
        int(split_seed)
        if partition_seed is None
        else int(partition_seed)
    )
    use_synthetic = False
    try:
        full_train = MedicalSegmentationDataset(
            data_root,
            split="train",
            img_size=img_size,
            class_count_scope=entropy_scope,
        )
        print(f"[Data] Using real BraTS data from: {data_root}")
    except Exception as exc:
        if not allow_synthetic:
            raise RuntimeError(
                f"Failed to load the requested dataset at {data_root}. "
                "Synthetic fallback is disabled."
            ) from exc
        use_synthetic = True

    if use_synthetic:
        print("[Synthetic] Using a BraTS-like synthetic dataset")
        full_train = SyntheticMedicalDataset(
            n=200,
            img_size=img_size,
            n_classes=4,
            seed=split_seed,
            alpha_dirichlet=alpha,
        )

    split_cache = Path(data_root) / f"_global_split_seed{split_seed}.json"
    if not use_synthetic and split_cache.exists() and not force_split:
        try:
            with split_cache.open("r", encoding="utf-8") as handle:
                split = json.load(handle)
            train_indices = split["train"]
            val_indices = split["val"]
            test_indices = split["test"]
        except Exception:
            train_indices, val_indices, test_indices = _make_global_split(
                len(full_train),
                val_ratio,
                test_ratio,
                split_seed,
            )
    else:
        train_indices, val_indices, test_indices = _make_global_split(
            len(full_train),
            val_ratio,
            test_ratio,
            split_seed,
        )
        if not use_synthetic:
            split_cache.parent.mkdir(parents=True, exist_ok=True)
            with split_cache.open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "train": train_indices,
                        "val": val_indices,
                        "test": test_indices,
                    },
                    handle,
                )

    print(
        "[Split] Global split sizes: "
        f"train={len(train_indices)}, "
        f"val={len(val_indices)}, "
        f"test={len(test_indices)}"
    )
    print(
        f"[Partition] IID={iid}, alpha={alpha}, "
        f"partition_seed={resolved_partition_seed}, "
        f"min_client_samples={min_client_samples}, "
        f"balance_client_sizes={balance_client_sizes}, "
        f"partition_basis={partition_basis}"
    )
    frozen_payload = None
    if partition_file:
        partitions, partition_metadata, frozen_payload = load_frozen_partitions(
            partition_file,
            train_indices,
            num_clients,
            include_payload=True,
        )
    else:
        partitions, partition_metadata = partition_data(
            full_train,
            train_indices,
            num_clients,
            iid=iid,
            alpha=alpha,
            seed=resolved_partition_seed,
            min_client_samples=min_client_samples,
            balance_client_sizes=balance_client_sizes,
            partition_basis=partition_basis,
            composition_bins=composition_bins,
            return_metadata=True,
        )
    effective_partition_seed = int(
        partition_metadata.get("partition_seed", resolved_partition_seed)
    )
    partition_metadata.update({
        "split_seed": int(split_seed),
        "partition_seed": effective_partition_seed,
        "entropy_scope": str(entropy_scope),
        "allocation_unit": "complete patient/volume",
    })

    num_workers = 0
    pin_memory = torch.cuda.is_available()
    client_loaders = []
    for client_id, sample_ids in enumerate(partitions):
        loader_generator = make_torch_generator(
            derive_seed(loader_seed, "data_loader", client_id),
            device="cpu",
        )
        client_loaders.append(
            DataLoader(
                Subset(full_train, sample_ids),
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=False,
                generator=loader_generator,
            )
        )

    val_loader = DataLoader(
        Subset(full_train, val_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        Subset(full_train, test_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    print("[Entropy] Building complete local foreground distributions...")
    client_stats = build_partition_evidence(
        full_train,
        partitions,
        partition_metadata,
    )
    client_val_partitions = None
    client_test_partitions = None
    if frozen_payload is not None:
        client_val_partitions = load_frozen_holdout_partitions(
            frozen_payload,
            "validation",
            val_indices,
            num_clients,
        )
        client_test_partitions = load_frozen_holdout_partitions(
            frozen_payload,
            "test",
            test_indices,
            num_clients,
        )
    if client_val_partitions is None:
        client_val_partitions = partition_holdout_by_training_profile(
            full_train,
            val_indices,
            client_stats,
            derive_seed(
                effective_partition_seed,
                "client_validation_partition",
            ),
        )
    if client_test_partitions is None:
        client_test_partitions = partition_holdout_by_training_profile(
            full_train,
            test_indices,
            client_stats,
            derive_seed(
                effective_partition_seed,
                "client_test_partition",
            ),
        )
    client_stats.client_val_loaders = [
        DataLoader(
            Subset(full_train, sample_ids),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        for sample_ids in client_val_partitions
    ]
    client_stats.client_test_loaders = [
        DataLoader(
            Subset(full_train, sample_ids),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        for sample_ids in client_test_partitions
    ]
    client_stats.metadata["evaluation_partitions"] = {
        "definition": (
            "disjoint central holdout allocation matched to each client's "
            "training dominant-label profile; used only for personalized "
            "FedBN evaluation"
        ),
        "validation": [
            {
                "client_id": int(client_id),
                "sample_indices": [int(sample_id) for sample_id in sample_ids],
            }
            for client_id, sample_ids in enumerate(client_val_partitions)
        ],
        "test": [
            {
                "client_id": int(client_id),
                "sample_indices": [int(sample_id) for sample_id in sample_ids],
            }
            for client_id, sample_ids in enumerate(client_test_partitions)
        ],
    }
    return client_loaders, val_loader, test_loader, client_stats
