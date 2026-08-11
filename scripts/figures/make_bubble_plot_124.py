#!/usr/bin/env python

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


INPUT_CSV = Path("/scratch/jbayasi/Cavmalproject1/results/lesion_segmentation_split_124/selection_master_124.csv")
OUT_DIR = Path("/scratch/jbayasi/Cavmalproject1/results/lesion_segmentation_split_124/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUT_DIR / "bubble_scatter_contrast_heterogeneity_volume_region.png"
OUT_PDF = OUT_DIR / "bubble_scatter_contrast_heterogeneity_volume_region.pdf"


def setup_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "figure.dpi": 400,
        "savefig.dpi": 400,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 1.5,
    })


def bubble_size(values, vmin, vmax, min_size=80, max_size=1100):
    values = np.asarray(values, dtype=float)
    if vmax <= vmin:
        return np.full_like(values, (min_size + max_size) / 2.0)
    return min_size + (values - vmin) * (max_size - min_size) / (vmax - vmin)


def main():
    setup_style()

    df = pd.read_csv(INPUT_CSV)

    required_cols = [
        "pat_id",
        "region",
        "lesion_volume_ml",
        "contrast_score",
        "heterogeneity_score",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[required_cols].dropna().copy()

    region_colors = {
        "Midbrain": "#0B1F4D",   # navy
        "Pons": "#F2C94C",       # yellow
        "Medulla": "#7A7A7A",    # gray
    }

    bad_regions = sorted(set(df["region"]) - set(region_colors))
    if bad_regions:
        raise ValueError(f"Unexpected region labels: {bad_regions}")

    vmin = float(df["lesion_volume_ml"].min())
    vmax = float(df["lesion_volume_ml"].max())
    df["bubble_size"] = bubble_size(df["lesion_volume_ml"], vmin, vmax)

    # Large canvas with a separate legend panel.
    fig = plt.figure(figsize=(17, 10.5), dpi=400)
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[4.6, 1.9],
        wspace=0.10,
    )

    ax = fig.add_subplot(gs[0, 0])
    legend_ax = fig.add_subplot(gs[0, 1])
    legend_ax.axis("off")

    # Scatter plot
    for region in ["Midbrain", "Pons", "Medulla"]:
        sub = df[df["region"] == region]
        ax.scatter(
            sub["contrast_score"],
            sub["heterogeneity_score"],
            s=sub["bubble_size"],
            c=region_colors[region],
            alpha=0.82,
            edgecolors="black",
            linewidths=0.8,
            label=region,
            zorder=3,
        )

    # Clean axes
    ax.set_title(
        "Lesion Appearance by Region and Volume",
        fontsize=34,
        pad=22,
    )

    ax.set_xlabel("")
    ax.set_ylabel("")

    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(length=0)

    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.25, zorder=0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Roomy plot limits
    x = df["contrast_score"].to_numpy(dtype=float)
    y = df["heterogeneity_score"].to_numpy(dtype=float)

    xpad = 0.10 * (x.max() - x.min() if x.max() > x.min() else 1.0)
    ypad = 0.10 * (y.max() - y.min() if y.max() > y.min() else 1.0)

    ax.set_xlim(x.min() - xpad, x.max() + xpad)
    ax.set_ylim(y.min() - ypad, y.max() + ypad)

    # X-axis direction arrow and labels
    ax.annotate(
        "",
        xy=(0.90, -0.13),
        xytext=(0.10, -0.13),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="<->", lw=2.2, color="black"),
        annotation_clip=False,
    )

    ax.text(
        0.50,
        -0.205,
        "Contrast",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=32,
    )

    ax.text(
        0.10,
        -0.29,
        "Darker",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=26,
    )

    ax.text(
        0.90,
        -0.29,
        "Brighter",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=26,
    )

    # Y-axis direction arrow and labels.
    # These are separated so they do not overlap the y-axis title.
    ax.annotate(
        "",
        xy=(-0.12, 0.90),
        xytext=(-0.12, 0.10),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="<->", lw=2.2, color="black"),
        annotation_clip=False,
    )

    ax.text(
        -0.245,
        0.50,
        "Heterogeneity",
        transform=ax.transAxes,
        ha="center",
        va="center",
        rotation=90,
        fontsize=32,
    )

    ax.text(
        -0.17,
        0.90,
        "More\nheterogeneous",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=24,
        linespacing=1.05,
    )

    ax.text(
        -0.17,
        0.10,
        "More\nuniform",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=24,
        linespacing=1.05,
    )

    # =========================
    # Custom legend panel
    # =========================

    legend_ax.text(
        0.00,
        0.94,
        "Region",
        fontsize=30,
        fontweight="semibold",
        ha="left",
        va="top",
        transform=legend_ax.transAxes,
    )

    region_y = [0.82, 0.70, 0.58]
    for y_pos, region in zip(region_y, ["Midbrain", "Pons", "Medulla"]):
        legend_ax.scatter(
            0.08,
            y_pos,
            s=330,
            c=region_colors[region],
            edgecolors="black",
            linewidths=0.9,
            transform=legend_ax.transAxes,
            clip_on=False,
        )
        legend_ax.text(
            0.20,
            y_pos,
            region,
            fontsize=27,
            ha="left",
            va="center",
            transform=legend_ax.transAxes,
        )

    legend_ax.text(
        0.00,
        0.43,
        "Lesion volume",
        fontsize=30,
        fontweight="semibold",
        ha="left",
        va="top",
        transform=legend_ax.transAxes,
    )

    ref_volumes = [0.5, 2.0, 5.0]
    ref_y = [0.31, 0.18, 0.045]
    ref_sizes = bubble_size(ref_volumes, vmin, vmax)

    for y_pos, vol, size in zip(ref_y, ref_volumes, ref_sizes):
        legend_ax.scatter(
            0.09,
            y_pos,
            s=size,
            facecolors="white",
            edgecolors="black",
            linewidths=1.1,
            transform=legend_ax.transAxes,
            clip_on=False,
        )

        label = f"{vol:g} mL"
        legend_ax.text(
            0.25,
            y_pos,
            label,
            fontsize=24,
            ha="left",
            va="center",
            transform=legend_ax.transAxes,
        )

    # Manual margins. Do not use tight_layout here because it can clip the legend/text.
    fig.subplots_adjust(
        left=0.20,
        right=0.96,
        bottom=0.30,
        top=0.88,
    )

    fig.savefig(OUT_PNG, dpi=400, pad_inches=0.35)
    fig.savefig(OUT_PDF, dpi=400, pad_inches=0.35)
    plt.close(fig)

    print("Saved PNG:", OUT_PNG)
    print("Saved PDF:", OUT_PDF)
    print("N patients plotted:", len(df))


if __name__ == "__main__":
    main()
