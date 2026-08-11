import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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

OUT_DIR = Path("/scratch/jbayasi/Cavmalproject1/results/key_training_figures_clean_comparison")
OUT_DIR.mkdir(parents=True, exist_ok=True)


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


def mean_by_epoch(df, metric):
    return df.groupby("epoch")[metric].mean().reset_index()


def plot_clean_metric(histories, metric, ylabel, title, filename):
    fig, ax = plt.subplots(figsize=(8, 5))

    for model_name, df in histories.items():
        summary = mean_by_epoch(df, metric)

        ax.plot(
            summary["epoch"],
            summary[metric],
            linewidth=2,
            label=model_name,
        )

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
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


def main():
    histories = {}

    for model_name, result_dir in RESULTS.items():
        df = load_histories(result_dir)

        if df is None:
            print(f"Skipping {model_name}")
            continue

        histories[model_name] = df

    plot_clean_metric(
        histories=histories,
        metric="val_balanced_accuracy",
        ylabel="Validation balanced accuracy",
        title="Validation balanced accuracy across models",
        filename="figure4_clean_validation_balanced_accuracy_mean_only",
    )

    plot_clean_metric(
        histories=histories,
        metric="val_macro_f1",
        ylabel="Validation macro-F1",
        title="Validation macro-F1 across models",
        filename="figure5_clean_validation_macro_f1_mean_only",
    )

    print("\nDone. Clean comparison figures saved to:")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
