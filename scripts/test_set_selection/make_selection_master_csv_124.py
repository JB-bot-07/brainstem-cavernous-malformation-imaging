#!/usr/bin/env python

from pathlib import Path
import json
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage

BASE = Path("/scratch/jbayasi/Cavmalproject1")
IMG_DIR = BASE / "unprocessednifti"
MASK_DIR = BASE / "semiautomasks"
REGION_CSV = BASE / "results/lesion_segmentation_split_124/patient_region_labels_124.csv"

OUT_CSV = BASE / "results/lesion_segmentation_split_124/selection_master_124.csv"
OUT_REPORT = BASE / "results/lesion_segmentation_split_124/selection_master_124_report.json"

EXCLUDED = {58, 91, 123}


def robust_nonzero_mask(img):
    finite_nonzero = np.isfinite(img) & (img != 0)
    vals = img[finite_nonzero]

    if vals.size == 0:
        return np.isfinite(img)

    low, high = np.percentile(vals, [0.5, 99.9])
    return finite_nonzero & (img >= low) & (img <= high)


def robust_mad(x):
    x = np.asarray(x)
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def assign_exact_quartiles(df, metric_col, out_col):
    df = df.sort_values([metric_col, "pat_id"]).copy()

    if len(df) != 124:
        raise ValueError(f"Expected 124 patients for exact quartiles, got {len(df)}")

    labels = []
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        labels.extend([q] * 31)

    df[out_col] = labels
    return df.sort_values("pat_id").reset_index(drop=True)


def compute_patient_features(row):
    pat_id = row["pat_id"]
    region = row["region"]
    patient_number = int(row["patient_number"])

    img_path = IMG_DIR / f"{pat_id}.nii.gz"
    mask_path = MASK_DIR / f"{pat_id}_semiautomask.nii.gz"

    if not img_path.exists():
        raise FileNotFoundError(f"Missing image: {img_path}")

    if not mask_path.exists():
        matches = sorted(MASK_DIR.glob(f"{pat_id}*.nii*"))
        if len(matches) == 1:
            mask_path = matches[0]
        else:
            raise FileNotFoundError(f"Missing or ambiguous mask for {pat_id}: {matches}")

    img_nii = nib.load(str(img_path))
    mask_nii = nib.load(str(mask_path))

    img = img_nii.get_fdata(dtype=np.float32)
    mask = mask_nii.get_fdata(dtype=np.float32)

    if img.shape != mask.shape:
        raise ValueError(f"{pat_id}: shape mismatch image {img.shape}, mask {mask.shape}")

    if not np.allclose(img_nii.affine, mask_nii.affine, atol=1e-3):
        raise ValueError(f"{pat_id}: affine mismatch between image and mask")

    lesion = np.isfinite(mask) & (mask > 0)

    lesion_voxels = int(lesion.sum())
    if lesion_voxels == 0:
        raise ValueError(f"{pat_id}: empty lesion mask")

    spacing = np.asarray(nib.affines.voxel_sizes(img_nii.affine), dtype=float)
    voxel_volume_mm3 = float(np.prod(spacing))

    lesion_volume_mm3 = float(lesion_voxels * voxel_volume_mm3)
    lesion_volume_ml = lesion_volume_mm3 / 1000.0

    brain_like = robust_nonzero_mask(img)

    dist_from_lesion = ndimage.distance_transform_edt(~lesion, sampling=spacing)

    shell = (
        (dist_from_lesion >= 3.0)
        & (dist_from_lesion <= 15.0)
        & brain_like
        & (~lesion)
    )

    shell_vals = img[shell]
    lesion_vals = img[lesion]

    if shell_vals.size < 50:
        raise ValueError(f"{pat_id}: too few shell voxels before trimming: {shell_vals.size}")

    shell_low, shell_high = np.percentile(shell_vals, [5, 95])
    shell_trim = shell & (img >= shell_low) & (img <= shell_high)
    shell_vals = img[shell_trim]

    if shell_vals.size < 50:
        raise ValueError(f"{pat_id}: too few shell voxels after trimming: {shell_vals.size}")

    lesion_median = float(np.median(lesion_vals))
    shell_median = float(np.median(shell_vals))

    shell_mad = robust_mad(shell_vals)
    shell_iqr = float(np.percentile(shell_vals, 75) - np.percentile(shell_vals, 25))
    shell_std = float(np.std(shell_vals))

    lesion_iqr = float(np.percentile(lesion_vals, 75) - np.percentile(lesion_vals, 25))
    lesion_std = float(np.std(lesion_vals))

    eps = 1e-6

    contrast_score = float((lesion_median - shell_median) / max(1.4826 * shell_mad, eps))
    heterogeneity_score = float(lesion_iqr / max(shell_iqr, eps))

    return {
        "pat_id": pat_id,
        "patient_number": patient_number,
        "region": region,
        "image_path": str(img_path),
        "mask_path": str(mask_path),
        "lesion_voxels": lesion_voxels,
        "voxel_volume_mm3": voxel_volume_mm3,
        "lesion_volume_mm3": lesion_volume_mm3,
        "lesion_volume_ml": lesion_volume_ml,
        "shell_voxels": int(shell_trim.sum()),
        "lesion_median_intensity": lesion_median,
        "shell_median_intensity": shell_median,
        "contrast_score": contrast_score,
        "lesion_iqr": lesion_iqr,
        "shell_iqr": shell_iqr,
        "heterogeneity_score": heterogeneity_score,
        "lesion_std": lesion_std,
        "shell_std": shell_std,
    }


def main():
    region_df = pd.read_csv(REGION_CSV)

    allowed = {"Midbrain", "Pons", "Medulla"}
    bad = sorted(set(region_df["region"]) - allowed)
    if bad:
        raise ValueError(f"Invalid region labels: {bad}")

    if len(region_df) != 124:
        raise ValueError(f"Expected 124 rows in region CSV, got {len(region_df)}")

    rows = []
    for _, row in region_df.iterrows():
        print("Processing", row["pat_id"], flush=True)
        rows.append(compute_patient_features(row))

    df = pd.DataFrame(rows)

    if len(df) != 124:
        raise ValueError(f"Expected 124 patients after feature extraction, got {len(df)}")

    df = assign_exact_quartiles(df, "lesion_volume_ml", "volume_quartile")
    df = assign_exact_quartiles(df, "contrast_score", "contrast_quartile")
    df = assign_exact_quartiles(df, "heterogeneity_score", "heterogeneity_quartile")

    front_cols = [
        "pat_id",
        "patient_number",
        "volume_quartile",
        "region",
        "contrast_quartile",
        "heterogeneity_quartile",
        "lesion_volume_ml",
        "contrast_score",
        "heterogeneity_score",
    ]

    other_cols = [c for c in df.columns if c not in front_cols]
    df = df[front_cols + other_cols].sort_values("patient_number")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    report = {
        "n_patients": int(len(df)),
        "excluded_patients": sorted(EXCLUDED),
        "region_counts": df["region"].value_counts().to_dict(),
        "volume_quartile_counts": df["volume_quartile"].value_counts().sort_index().to_dict(),
        "contrast_quartile_counts": df["contrast_quartile"].value_counts().sort_index().to_dict(),
        "heterogeneity_quartile_counts": df["heterogeneity_quartile"].value_counts().sort_index().to_dict(),
        "region_targets_for_24": {
            "Midbrain": 6,
            "Pons": 14,
            "Medulla": 4,
        },
        "volume_target_for_24": {
            "Q1": 6,
            "Q2": 6,
            "Q3": 6,
            "Q4": 6,
        },
        "contrast_soft_target_for_24": {
            "Q1": 6,
            "Q2": 6,
            "Q3": 6,
            "Q4": 6,
            "acceptable_range": "5-7 per quartile",
        },
        "heterogeneity_soft_target_for_24": {
            "Q1": 6,
            "Q2": 6,
            "Q3": 6,
            "Q4": 6,
            "acceptable_range": "5-7 per quartile",
        },
    }

    with open(OUT_REPORT, "w") as f:
        json.dump(report, f, indent=2)

    print("Saved:", OUT_CSV)
    print("Saved:", OUT_REPORT)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
