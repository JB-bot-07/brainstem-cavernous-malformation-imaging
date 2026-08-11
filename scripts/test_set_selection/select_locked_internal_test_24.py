#!/usr/bin/env python

from pathlib import Path
import json
import numpy as np
import pandas as pd

BASE = Path("/scratch/jbayasi/Cavmalproject1")
MASTER_CSV = BASE / "results/lesion_segmentation_split_124/selection_master_124.csv"
OUT_DIR = BASE / "results/lesion_segmentation_split_124"

TEST_CSV = OUT_DIR / "locked_internal_test_24.csv"
TRAINVAL_CSV = OUT_DIR / "trainval_100.csv"
REPORT_JSON = OUT_DIR / "locked_internal_test_24_selection_report.json"

RANDOM_SEED = 20260728
N_ITERATIONS = 500000

VOLUME_TARGET = {"Q1": 6, "Q2": 6, "Q3": 6, "Q4": 6}
REGION_TARGET = {"Midbrain": 6, "Pons": 14, "Medulla": 4}
CONTRAST_TARGET = {"Q1": 6, "Q2": 6, "Q3": 6, "Q4": 6}
HETEROGENEITY_TARGET = {"Q1": 6, "Q2": 6, "Q3": 6, "Q4": 6}

SOFT_MIN = 5
SOFT_MAX = 7


def count_dict(df, col, levels):
    counts = df[col].value_counts().to_dict()
    return {level: int(counts.get(level, 0)) for level in levels}


def abs_count_error(observed, target):
    return int(sum(abs(observed[k] - target[k]) for k in target))


def summarize(df):
    return {
        "n": int(len(df)),
        "region_counts": count_dict(df, "region", ["Midbrain", "Pons", "Medulla"]),
        "volume_quartile_counts": count_dict(df, "volume_quartile", ["Q1", "Q2", "Q3", "Q4"]),
        "contrast_quartile_counts": count_dict(df, "contrast_quartile", ["Q1", "Q2", "Q3", "Q4"]),
        "heterogeneity_quartile_counts": count_dict(df, "heterogeneity_quartile", ["Q1", "Q2", "Q3", "Q4"]),
        "lesion_volume_ml_mean": float(df["lesion_volume_ml"].mean()),
        "lesion_volume_ml_median": float(df["lesion_volume_ml"].median()),
        "contrast_score_mean": float(df["contrast_score"].mean()),
        "contrast_score_median": float(df["contrast_score"].median()),
        "heterogeneity_score_mean": float(df["heterogeneity_score"].mean()),
        "heterogeneity_score_median": float(df["heterogeneity_score"].median()),
    }


def main():
    df = pd.read_csv(MASTER_CSV)

    if len(df) != 124:
        raise ValueError(f"Expected 124 patients, got {len(df)}")

    rng = np.random.default_rng(RANDOM_SEED)

    volume_levels = ["Q1", "Q2", "Q3", "Q4"]
    region_levels = ["Midbrain", "Pons", "Medulla"]
    contrast_levels = ["Q1", "Q2", "Q3", "Q4"]
    heterogeneity_levels = ["Q1", "Q2", "Q3", "Q4"]

    grouped_by_volume = {
        q: df.loc[df["volume_quartile"] == q, "pat_id"].to_numpy()
        for q in volume_levels
    }

    best_score = None
    best_ids = None
    best_details = None

    candidates_checked = 0
    candidates_rejected_soft_quartiles = 0
    candidates_accepted = 0

    for i in range(N_ITERATIONS):
        selected = []

        # Hard constraint: exactly 6 from each volume quartile.
        for q in volume_levels:
            selected.extend(
                rng.choice(grouped_by_volume[q], size=6, replace=False).tolist()
            )

        cand = df[df["pat_id"].isin(selected)].copy()

        if len(cand) != 24:
            continue

        candidates_checked += 1

        volume_counts = count_dict(cand, "volume_quartile", volume_levels)
        region_counts = count_dict(cand, "region", region_levels)
        contrast_counts = count_dict(cand, "contrast_quartile", contrast_levels)
        heterogeneity_counts = count_dict(cand, "heterogeneity_quartile", heterogeneity_levels)

        if volume_counts != VOLUME_TARGET:
            continue

        # Soft filters: contrast and heterogeneity should be 5–7 each.
        if any(v < SOFT_MIN or v > SOFT_MAX for v in contrast_counts.values()):
            candidates_rejected_soft_quartiles += 1
            continue

        if any(v < SOFT_MIN or v > SOFT_MAX for v in heterogeneity_counts.values()):
            candidates_rejected_soft_quartiles += 1
            continue

        candidates_accepted += 1

        region_error = abs_count_error(region_counts, REGION_TARGET)
        contrast_error = abs_count_error(contrast_counts, CONTRAST_TARGET)
        heterogeneity_error = abs_count_error(heterogeneity_counts, HETEROGENEITY_TARGET)

        imbalance_score = region_error + contrast_error + heterogeneity_error

        score_tuple = (
            imbalance_score,
            region_error,
            contrast_error,
            heterogeneity_error,
            ",".join(sorted(selected)),
        )

        if best_score is None or score_tuple < best_score:
            best_score = score_tuple
            best_ids = selected
            best_details = {
                "iteration": i,
                "imbalance_score": int(imbalance_score),
                "region_error": int(region_error),
                "contrast_error": int(contrast_error),
                "heterogeneity_error": int(heterogeneity_error),
                "volume_counts": volume_counts,
                "region_counts": region_counts,
                "contrast_counts": contrast_counts,
                "heterogeneity_counts": heterogeneity_counts,
            }

    if best_ids is None:
        raise RuntimeError(
            "No candidate found. Increase N_ITERATIONS or relax soft quartile range."
        )

    test_df = df[df["pat_id"].isin(best_ids)].copy()
    trainval_df = df[~df["pat_id"].isin(best_ids)].copy()

    test_df = test_df.sort_values(["volume_quartile", "region", "pat_id"])
    trainval_df = trainval_df.sort_values("pat_id")

    if len(test_df) != 24:
        raise ValueError(f"Expected 24 test patients, got {len(test_df)}")

    if len(trainval_df) != 100:
        raise ValueError(f"Expected 100 train/val patients, got {len(trainval_df)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    test_df.to_csv(TEST_CSV, index=False)
    trainval_df.to_csv(TRAINVAL_CSV, index=False)

    report = {
        "random_seed": RANDOM_SEED,
        "n_iterations": N_ITERATIONS,
        "candidates_checked": candidates_checked,
        "candidates_rejected_soft_quartiles": candidates_rejected_soft_quartiles,
        "candidates_accepted_after_soft_filters": candidates_accepted,
        "excluded_patient_numbers": [58, 91, 123],
        "hard_constraint": {
            "volume_quartile_counts": VOLUME_TARGET,
        },
        "region_target": REGION_TARGET,
        "contrast_soft_target": CONTRAST_TARGET,
        "heterogeneity_soft_target": HETEROGENEITY_TARGET,
        "soft_quartile_acceptable_range": [SOFT_MIN, SOFT_MAX],
        "best_candidate_details": best_details,
        "full_cohort_summary": summarize(df),
        "locked_test_summary": summarize(test_df),
        "trainval_summary": summarize(trainval_df),
        "locked_test_patients": test_df["pat_id"].tolist(),
    }

    with open(REPORT_JSON, "w") as f:
        json.dump(report, f, indent=2)

    print("Saved:", TEST_CSV)
    print("Saved:", TRAINVAL_CSV)
    print("Saved:", REPORT_JSON)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
