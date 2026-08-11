#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


INPUT_CSV = Path("/scratch/jbayasi/Cavmalproject1/results/locked24_final_metrics/locked24_dice_centroid_hd95_assd_case_metrics.csv")
OUT_DIR = Path("/scratch/jbayasi/Cavmalproject1/results/locked24_final_metrics/figures")


def setup_matplotlib():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 14,
        "axes.titlesize": 18,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "figure.dpi": 300,
        "savefig.dpi": 600,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def make_boxplot(values, title, ylabel, out_prefix, ylim=None):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    fig, ax = plt.subplots(figsize=(4.4, 5.4))

    ax.boxplot(
        [values],
        widths=0.45,
        patch_artist=True,
        showfliers=False,

        # Important:
        # Median is the line inside the box.
        # Mean is the diamond marker.
        showmeans=True,
        meanline=False,

        boxprops=dict(facecolor="#4A90E2", edgecolor="black", linewidth=1.6),
        whiskerprops=dict(color="black", linewidth=1.5),
        capprops=dict(color="black", linewidth=1.5),
        medianprops=dict(color="black", linewidth=2.0),
        meanprops=dict(
            marker="D",
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=1.0,
            markersize=8,
        ),
    )

    # Individual patient points
    rng = np.random.default_rng(42)
    x = rng.normal(loc=1.0, scale=0.035, size=len(values))

    ax.scatter(
        x,
        values,
        s=28,
        alpha=0.65,
        facecolors="#4A90E2",
        edgecolors="black",
        linewidths=0.4,
        zorder=3,
    )

    ax.set_title(title, pad=12)
    ax.set_ylabel(ylabel)
    ax.set_xticks([1])
    ax.set_xticklabels(["Locked test set\nn = 24"])
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)

    if ylim is not None:
        ax.set_ylim(ylim)

    fig.tight_layout()

    png_path = OUT_DIR / f"{out_prefix}.png"
    pdf_path = OUT_DIR / f"{out_prefix}.pdf"
    svg_path = OUT_DIR / f"{out_prefix}.svg"

    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", png_path)
    print("Saved:", pdf_path)
    print("Saved:", svg_path)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)
    valid = df[df["error"].fillna("") == ""].copy()

    if len(valid) != 24:
        raise ValueError(f"Expected 24 valid cases, found {len(valid)}")

    setup_matplotlib()

    make_boxplot(
        values=valid["dice"],
        title="Lesion Segmentation Dice",
        ylabel="Dice score",
        out_prefix="locked24_dice_boxplot",
        ylim=(0.4, 1.02),
    )

    make_boxplot(
        values=valid["centroid_distance_mm"],
        title="Centroid Error",
        ylabel="Centroid distance (mm)",
        out_prefix="locked24_centroid_distance_boxplot",
        ylim=None,
    )

    print()
    print("Done. Figure folder:")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
