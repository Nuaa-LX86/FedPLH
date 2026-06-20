from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch


def derive_seed(base_seed: int, stream: str, index: int = 0) -> int:
    payload = f"{int(base_seed)}:{stream}:{int(index)}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def seed_everything(seed: int, deterministic: bool = False) -> None:
    import random

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def make_torch_generator(seed: int, device: str = "cpu") -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def build_client_schedule(
    num_clients: int,
    rounds: int,
    client_fraction: float,
    seed: int,
) -> List[List[int]]:
    if num_clients <= 0:
        raise ValueError("num_clients must be positive")
    if rounds <= 0:
        raise ValueError("rounds must be positive")
    if not 0.0 < client_fraction <= 1.0:
        raise ValueError("client_fraction must be in (0, 1]")

    clients_per_round = min(
        num_clients,
        max(2, int(num_clients * client_fraction)),
    )
    rng = np.random.default_rng(derive_seed(seed, "client_selection"))
    return [
        [int(v) for v in rng.choice(num_clients, clients_per_round, replace=False)]
        for _ in range(rounds)
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint_files(root: Path, relative_paths: Iterable[str]) -> Dict[str, str]:
    fingerprints: Dict[str, str] = {}
    for relative_path in relative_paths:
        path = root / relative_path
        if path.is_file():
            fingerprints[relative_path.replace("\\", "/")] = sha256_file(path)
    return fingerprints


def dataset_inventory(data_root: Path) -> Dict[str, Any]:
    if not data_root.exists():
        return {"path": str(data_root.resolve()), "exists": False}

    entries = []
    total_bytes = 0
    for path in sorted(data_root.rglob("*.npy")):
        stat = path.stat()
        relative = path.relative_to(data_root).as_posix()
        entries.append(f"{relative}\t{stat.st_size}")
        total_bytes += stat.st_size

    inventory_hash = hashlib.sha256("\n".join(entries).encode("utf-8")).hexdigest()
    split_hashes = {
        path.name: sha256_file(path)
        for path in sorted(data_root.glob("_global_split_seed*.json"))
    }
    return {
        "path": str(data_root.resolve()),
        "exists": True,
        "npy_file_count": len(entries),
        "total_bytes": total_bytes,
        "inventory_sha256": inventory_hash,
        "split_file_sha256": split_hashes,
    }


def collect_environment() -> Dict[str, Any]:
    gpu_names = []
    if torch.cuda.is_available():
        gpu_names = [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_names": gpu_names,
        "pid": os.getpid(),
    }


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    temporary.replace(path)
