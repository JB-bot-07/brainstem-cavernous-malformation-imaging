import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Result folders
# ============================================================
RESULTS = {
    "Frozen BrainIAC MLP": Path(
        "/scratch/jbayasi/Cavmalproject1/results/frozen_brainiac_mlp_5fold_150ep_from_pt_clean"
    ),
    "Partial FT last 2 blocks": Path(
        "/scratch/jbayasi/Cavmalproject1/results/partial_finetune_clean_last2blocks_150ep_best_and_final"
    ),
    "Partial FT last 4 blocks": Path(
        "/scratch/jbayasi/Cavmalproject1/results/partial_finetune_clean_last4blocks_150ep_best_and_final"
    ),
}

OUT_DIR = Path("/scratch/jbayasi/Cavmalproject1/results/key_training_figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Helper functions
# ============================================================
def load_histories(result_dir):
    dfs = []

    for fold in range(5):
        possible_paths = [
            result_dir / f"fold{fold}" / "training_history.csv",
            result_dir / f"fold{fold}" / "partial_finetune_fold0_training_history.csv",
        ]

        path = None
        for p in possible_paths:
            if p.exists():
                path = p
                break

        if path is None:
            print(f"Missing training history for fold {fold} in {result_dir}")
            continue

        df = pd.read_csv(path)
        df["fold"] = fold
        dfs.append(df)

    if len(dfs) == 0:
        return None

    return pd.concat(dfs, ignore_index=True)


def mean_sd_by_epoch(df, metric):
    summary = df.groupby("epoch")[metric].agg(["mean", "std"]).reset_index()
    summary["std"] = summary["std"].fillna(0)
    return summary


def add_moving_average(summary, window=5):
    summary = summary.copy()
    summary["moving_average"] = summary["mean"].rolling(
        window=window,
        min_periods=1,
        center=False
    ).mean()
    return summary


def plot_loss_for_model(model_name, df, filename):
    train = mean_sd_by_epoch(df, "train_loss")
    val = mean_sd_by_epoch(df, "val_loss")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(train["epoch"], train["mean"], label="Train loss")
    ax.fill_between(
        train["epoch"],
        train["mean"] - train["std"],
        train["mean"] + train["std"],
        alpha=0.2,
    )

    ax.plot(val["epoch"], val["mean"], label="Validation loss")
    ax.fill_between(
        val["epoch"],
        val["mean"] - val["std"],
        val["mean"] + val["std"],
        alpha=0.2,
    )

    ax.set_title(f"{model_name}: train vs validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    png_path = OUT_DIR / f"{filename}.png"
    pdf_path = OUT_DIR / f"{filename}.pdf"

    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)

    print("Saved:", png_path)
    print("Saved:", pdf_path)


def plot_validation_metric_comparison(histories, metric, ylabel, title, filename):
    fig, ax = plt.subplots(figsize=(8, 5))

    for model_name, df in histories.items():
        summary = mean_sd_by_epoch(df, metric)
        summary = add_moving_average(summary, window=5)

        ax.plot(
            summary["epoch"],
            summary["mean"],
            alpha=0.45,
            label=f"{model_name} mean",
        )

        ax.plot(
            summary["epoch"],
            summary["moving_average"],
            linewidth=2,
            label=f"{model_name} moving avg",
        )

        ax.fill_between(
            summary["epoch"],
            summary["mean"] - summary["std"],
            summary["mean"] + summary["std"],
            alpha=0.12,
        )

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    png_path = OUT_DIR / f"{filename}.png"
    pdf_path = OUT_DIR / f"{filename}.pdf"

    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)

    print("Saved:", png_path)
    print("Saved:", pdf_path)


# ============================================================
# Main
# ============================================================
def main():
    histories = {}

    for model_name, result_dir in RESULTS.items():
        print("\n" + "=" * 80)
        print(model_name)
        print("=" * 80)

        df = load_histories(result_dir)

        if df is None:
            print(f"Skipping {model_name}: no histories found.")
            continue

        histories[model_name] = df
        print(f"Loaded {len(df)} rows from {result_dir}")

    # Figure 1
    if "Frozen BrainIAC MLP" in histories:
        plot_loss_for_model(
            model_name="Frozen BrainIAC MLP",
            df=histories["Frozen BrainIAC MLP"],
            filename="figure1_frozen_train_vs_val_loss",
        )

    # Figure 2
    if "Partial FT last 2 blocks" in histories:
        plot_loss_for_model(
            model_name="Partial FT last 2 blocks",
            df=histories["Partial FT last 2 blocks"],
            filename="figure2_partial_last2_train_vs_val_loss",
        )

    # Figure 3
    if "Partial FT last 4 blocks" in histories:
        plot_loss_for_model(
            model_name="Partial FT last 4 blocks",
            df=histories["Partial FT last 4 blocks"],
            filename="figure3_partial_last4_train_vs_val_loss",
        )

    # Figure 4
    if len(histories) >= 2:
        plot_validation_metric_comparison(
            histories=histories,
            metric="val_balanced_accuracy",
            ylabel="Validation balanced accuracy",
            title="Validation balanced accuracy across models",
            filename="figure4_validation_balanced_accuracy_across_models",
        )

    # Figure 5
    if len(histories) >= 2:
        plot_validation_metric_comparison(
            histories=histories,
            metric="val_macro_f1",
            ylabel="Validation macro-F1",
            title="Validation macro-F1 across models",
            filename="figure5_validation_macro_f1_across_models",
        )

    print("\nDone.")
    print("Figures saved to:")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
