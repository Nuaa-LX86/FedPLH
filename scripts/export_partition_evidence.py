import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.dataset_loader import (
    FULL_VOLUME_SCOPE,
    get_federated_dataloaders,
    write_partition_evidence,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="./dataset/processed")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--clients", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--partition_seed", type=int, default=None)
    parser.add_argument("--min_client_samples", type=int, default=10)
    parser.add_argument("--balance_client_sizes", action="store_true")
    parser.add_argument(
        "--partition_basis",
        default="dominant_foreground_label",
    )
    parser.add_argument("--composition_bins", type=int, default=10)
    parser.add_argument("--entropy_scope", default=FULL_VOLUME_SCOPE)
    args = parser.parse_args()

    _, _, _, client_stats = get_federated_dataloaders(
        num_clients=args.clients,
        batch_size=args.batch_size,
        img_size=args.img_size,
        iid=False,
        alpha=args.alpha,
        data_root=args.data_root,
        val_ratio=0.1,
        test_ratio=0.1,
        split_seed=args.split_seed,
        partition_seed=args.partition_seed,
        loader_seed=args.split_seed,
        allow_synthetic=False,
        entropy_scope=args.entropy_scope,
        min_client_samples=args.min_client_samples,
        balance_client_sizes=args.balance_client_sizes,
        partition_basis=args.partition_basis,
        composition_bins=args.composition_bins,
    )
    write_partition_evidence(client_stats.metadata, args.output_dir)


if __name__ == "__main__":
    main()
