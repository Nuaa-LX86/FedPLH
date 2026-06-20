import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_characterization(evidence_path: Path, output_prefix: Path) -> None:
    with evidence_path.open("r", encoding="utf-8") as handle:
        evidence = json.load(handle)

    clients = evidence["clients"]
    client_ids = np.array([client["client_id"] for client in clients])
    sample_counts = np.array([client["num_samples"] for client in clients])
    entropies = np.array([client["entropy"] for client in clients])
    proportions = np.array([
        [
            client["foreground_proportions"]["1"],
            client["foreground_proportions"]["2"],
            client["foreground_proportions"]["3"],
        ]
        for client in clients
    ])

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(8.4, 3.25),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.05, 1.35]},
    )

    image = axes[0].imshow(
        proportions,
        aspect="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=max(0.5, float(proportions.max())),
    )
    axes[0].set_xlabel("Foreground label")
    axes[0].set_ylabel("Client")
    axes[0].set_xticks([0, 1, 2], ["1", "2", "3"])
    axes[0].set_yticks(client_ids)
    colorbar = figure.colorbar(image, ax=axes[0], fraction=0.05, pad=0.03)
    colorbar.set_label("Voxel proportion")

    axes[1].bar(
        client_ids,
        sample_counts,
        color="#4C78A8",
        width=0.78,
        label="Patient volumes",
    )
    axes[1].set_xlabel("Client")
    axes[1].set_ylabel("Patient volumes", color="#2F5D8A")
    axes[1].tick_params(axis="y", labelcolor="#2F5D8A")
    axes[1].set_xticks(client_ids)

    entropy_axis = axes[1].twinx()
    entropy_axis.plot(
        client_ids,
        entropies,
        color="#D1495B",
        marker="o",
        markersize=3,
        linewidth=1.2,
        label="Foreground entropy",
    )
    entropy_axis.set_ylabel("Entropy (bits)", color="#A62E3B")
    entropy_axis.tick_params(axis="y", labelcolor="#A62E3B")

    partition = evidence["partition"]
    heterogeneity = evidence["realized_heterogeneity"]
    figure.suptitle(
        "Realized client heterogeneity: "
        f"alpha={partition.get('alpha')}, split={partition.get('split_seed')}, "
        f"mean JSD={heterogeneity['mean_client_global_jsd_bits']:.4f} bits",
        fontsize=9,
    )

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot_characterization(args.evidence, args.output)


if __name__ == "__main__":
    main()
