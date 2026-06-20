from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


COLORS = {
    "signal": "#4E79A7",
    "budget": "#59A14F",
    "privacy": "#E15759",
    "control": "#F28E2B",
    "neutral": "#BAB0AC",
    "ink": "#222222",
}


def box(ax, x, y, width, height, text, color, fontsize=6.5, linewidth=1.2):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        facecolor="white",
        edgecolor=color,
        linewidth=linewidth,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=COLORS["ink"],
    )
    return patch


def arrow(ax, start, end, color, label=None, label_offset=(0, 0), style="-|>"):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=10,
        linewidth=1.15,
        color=color,
        connectionstyle="arc3",
    )
    ax.add_patch(patch)
    if label:
        ax.text(
            (start[0] + end[0]) / 2 + label_offset[0],
            (start[1] + end[1]) / 2 + label_offset[1],
            label,
            ha="center",
            va="center",
            fontsize=7,
            color=color,
        )


def plot(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.15, 4.15))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    ax.text(
        0.2,
        6.92,
        "BEU optimizer-update datapath and budget state",
        fontsize=9,
        fontweight="bold",
        color=COLORS["ink"],
    )

    box(
        ax,
        0.25,
        5.25,
        1.75,
        0.8,
        "HMPE timing slack\n$\\Delta C_k(p)$",
        COLORS["signal"],
    )
    box(
        ax,
        2.5,
        5.25,
        1.75,
        0.8,
        "Budget accrual\n$\\lambda_B\\Delta C_k(p)$",
        COLORS["budget"],
    )
    box(
        ax,
        4.75,
        5.25,
        1.9,
        0.8,
        "Pre-consumption budget\n$\\widetilde{B}_k=B_k+\\lambda_B\\Delta C_k$",
        COLORS["budget"],
        fontsize=6.1,
    )
    box(
        ax,
        7.15,
        5.25,
        1.55,
        0.8,
        "Coverage test\n$\\widetilde{B}_k\\geq c_{\\rm priv,k}$",
        COLORS["control"],
        fontsize=6.2,
    )
    box(
        ax,
        9.25,
        5.25,
        2.25,
        0.8,
        "Visible increment\n$\\max(0,c_{\\rm priv,k}-\\widetilde{B}_k)$",
        COLORS["control"],
        fontsize=6.1,
    )

    arrow(ax, (2.0, 5.65), (2.5, 5.65), COLORS["signal"])
    arrow(ax, (4.25, 5.65), (4.75, 5.65), COLORS["budget"])
    arrow(ax, (6.65, 5.65), (7.15, 5.65), COLORS["budget"])
    arrow(ax, (8.7, 5.65), (9.25, 5.65), COLORS["control"])

    box(
        ax,
        0.25,
        3.2,
        1.75,
        0.9,
        "Local gradient\n$g_k$ at writeback",
        COLORS["signal"],
    )
    box(
        ax,
        2.5,
        3.2,
        1.75,
        0.9,
        "RMS / norm\nstatistics",
        COLORS["privacy"],
    )
    box(
        ax,
        4.75,
        3.2,
        1.9,
        0.9,
        "Clipping + noise\ngeneration/injection",
        COLORS["privacy"],
        fontsize=6.2,
    )
    box(
        ax,
        7.15,
        3.2,
        1.55,
        0.9,
        "Privacy-cost meter\n$c_{\\rm priv,k}$",
        COLORS["privacy"],
        fontsize=6.2,
    )
    box(
        ax,
        9.25,
        3.2,
        2.25,
        0.9,
        "Processed\ngradient stream",
        COLORS["privacy"],
    )

    arrow(ax, (2.0, 3.65), (2.5, 3.65), COLORS["signal"])
    arrow(ax, (4.25, 3.65), (4.75, 3.65), COLORS["privacy"])
    arrow(ax, (6.65, 3.65), (7.15, 3.65), COLORS["privacy"])
    arrow(ax, (8.7, 3.65), (9.25, 3.65), COLORS["privacy"])
    arrow(
        ax,
        (7.925, 4.1),
        (7.925, 5.25),
        COLORS["privacy"],
        "$c_{\\rm priv,k}$",
        (0.45, 0),
    )

    box(
        ax,
        2.5,
        1.25,
        1.75,
        0.85,
        "Window trigger\nwriteback / existing wait",
        COLORS["signal"],
        fontsize=6.1,
    )
    box(
        ax,
        4.75,
        1.25,
        1.9,
        0.85,
        "Budget register\n$B_k\\geq0$",
        COLORS["budget"],
    )
    box(
        ax,
        7.15,
        1.25,
        1.55,
        0.85,
        "State update\n$B_{k+1}=\\max(0,\\widetilde{B}_k-c_{\\rm priv,k})$",
        COLORS["budget"],
        fontsize=5.7,
    )
    box(
        ax,
        9.25,
        1.25,
        2.25,
        0.85,
        "Client-round aggregates\n$\\Delta C=\\sum_k\\Delta C_k$, "
        "$C_{\\rm priv}=\\sum_k c_{\\rm priv,k}$",
        COLORS["neutral"],
        fontsize=5.8,
    )

    arrow(ax, (3.375, 2.1), (3.375, 3.2), COLORS["signal"], "window", (0.35, 0))
    arrow(ax, (8.7, 1.675), (9.25, 1.675), COLORS["neutral"])
    arrow(
        ax,
        (7.15, 1.675),
        (6.65, 1.675),
        COLORS["budget"],
    )
    arrow(
        ax,
        (7.925, 3.2),
        (7.925, 2.1),
        COLORS["privacy"],
        "$c_{\\rm priv,k}$",
        (0.45, 0),
    )

    ax.text(
        0.25,
        0.42,
        r"$c_{\rm priv,k}=c_{\rm RMS,k}+c_{\rm clip,k}+"
        r"c_{\rm noise\ gen/inj,k}$; scheduling changes the execution window, "
        r"not clipping/noise semantics.",
        fontsize=6.6,
        color=COLORS["ink"],
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(
        output.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.04,
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper_figures_unet/Fig3_BEU_tildeB.pdf"),
    )
    args = parser.parse_args()
    plot(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
