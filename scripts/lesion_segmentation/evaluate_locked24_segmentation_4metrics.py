#!/usr/bin/env python

from pathlib import Path
import argparse
import json
import re

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage


def patient_number_from_pat_id(pat_id):
    m = re.search(r"(\d+)$", str(pat_id))
    if not m:
        raise ValueError(f"Could not parse patient number from {pat_id}")
    return int(m.group(1))


def find_prediction(pred_dir, patient_number):
    candidates = [
        pred_dir / f"Bni_Cm_{patient_number:03d}.nii.gz",
        pred_dir / f"Bni_Cm_Ret_{patient_number:03d}.nii.gz",
        pred_dir / f"Bni_Cm_{patient_number:03d}_0000.nii.gz",
        pred_dir / f"Bni_Cm_Ret_{patient_number:03d}_0000.nii.gz",
    ]

    for c in candidates:
        if c.exists():
            return c

    matches = sorted(pred_dir.glob(f"*{patient_number:03d}*.nii.gz"))

    if len(matches) == 1:
        return matches[0]

    raise FileNotFoundError(
        f"Missing or ambiguous prediction for patient {patient_number:03d}: "
        f"{[m.name for m in matches]}"
    )


def find_gt_mask(gt_dir, patient_number):
    pat_id = f"Bni_Cm_Ret_{patient_number:03d}"

    candidates = [
        gt_dir / f"{pat_id}_semiautomask.nii.gz",
        gt_dir / f"{pat_id}.nii.gz",
    ]

    for c in candidates:
        if c.exists():
            return c

    matches = sorted(gt_dir.glob(f"{pat_id}*.nii*"))

    if len(matches) == 1:
        return matches[0]

    raise FileNotFoundError(
        f"Missing or ambiguous GT mask for {pat_id}: {[m.name for m in matches]}"
    )


def dice_score(pred, gt):
    pred_sum = int(pred.sum())
    gt_sum = int(gt.sum())

    if pred_sum == 0 and gt_sum == 0:
        return 1.0

    if pred_sum == 0 or gt_sum == 0:
        return 0.0

    intersection = int(np.logical_and(pred, gt).sum())
    return float((2.0 * intersection) / (pred_sum + gt_sum))


def centroid_world(mask, affine):
    coords = np.argwhere(mask)

    if coords.shape[0] == 0:
        return None, None

    centroid_voxel = coords.mean(axis=0)
    centroid_world = nib.affines.apply_affine(affine, centroid_voxel)

    return centroid_voxel, centroid_world


def surface_voxels(mask):
    if int(mask.sum()) == 0:
        return np.zeros(mask.shape, dtype=bool)

    structure = ndimage.generate_binary_structure(rank=3, connectivity=1)
    eroded = ndimage.binary_erosion(mask, structure=structure, border_value=0)
    return mask & (~eroded)


def surface_distance_metrics(pred, gt, spacing):
    """
    Returns HD95 and ASSD in mm using symmetric surface distances.

    HD95 = 95th percentile of bidirectional surface distances.
    ASSD = average symmetric surface distance.
    """

    pred_empty = int(pred.sum()) == 0
    gt_empty = int(gt.sum()) == 0

    if pred_empty and gt_empty:
        return 0.0, 0.0

    if pred_empty or gt_empty:
        return np.nan, np.nan

    pred_surface = surface_voxels(pred)
    gt_surface = surface_voxels(gt)

    if int(pred_surface.sum()) == 0 or int(gt_surface.sum()) == 0:
        return np.nan, np.nan

    # Distance from every voxel to nearest GT surface.
    dist_to_gt_surface = ndimage.distance_transform_edt(
        ~gt_surface,
        sampling=spacing,
    )

    # Distance from every voxel to nearest predicted surface.
    dist_to_pred_surface = ndimage.distance_transform_edt(
        ~pred_surface,
        sampling=spacing,
    )

    pred_to_gt = dist_to_gt_surface[pred_surface]
    gt_to_pred = dist_to_pred_surface[gt_surface]

    all_surface_distances = np.concatenate([pred_to_gt, gt_to_pred])

    hd95 = float(np.percentile(all_surface_distances, 95))
    assd = float(all_surface_distances.mean())

    return hd95, assd


def evaluate_case(pred_path, gt_path):
    pred_nii = nib.load(str(pred_path))
    gt_nii = nib.load(str(gt_path))

    pred_data = pred_nii.get_fdata(dtype=np.float32)
    gt_data = gt_nii.get_fdata(dtype=np.float32)

    shape_equal = pred_nii.shape == gt_nii.shape
    affine_close = np.allclose(pred_nii.affine, gt_nii.affine, atol=1e-3)

    if not shape_equal:
        raise ValueError(f"Shape mismatch: pred {pred_nii.shape}, gt {gt_nii.shape}")

    if not affine_close:
        raise ValueError(f"Affine mismatch: pred {pred_path}, gt {gt_path}")

    pred = np.isfinite(pred_data) & (pred_data > 0)
    gt = np.isfinite(gt_data) & (gt_data > 0)

    pred_voxels = int(pred.sum())
    gt_voxels = int(gt.sum())
    intersection_voxels = int(np.logical_and(pred, gt).sum())

    dice = dice_score(pred, gt)

    pred_centroid_voxel, pred_centroid_world = centroid_world(pred, pred_nii.affine)
    gt_centroid_voxel, gt_centroid_world = centroid_world(gt, gt_nii.affine)

    if pred_centroid_world is None or gt_centroid_world is None:
        centroid_distance_mm = np.nan
    else:
        centroid_distance_mm = float(np.linalg.norm(pred_centroid_world - gt_centroid_world))

    spacing = np.asarray(nib.affines.voxel_sizes(gt_nii.affine), dtype=float)
    voxel_volume_mm3 = float(np.prod(spacing))

    hd95_mm, assd_mm = surface_distance_metrics(pred, gt, spacing)

    if pred_centroid_voxel is None:
        pred_centroid_voxel = [np.nan, np.nan, np.nan]
        pred_centroid_world = [np.nan, np.nan, np.nan]

    if gt_centroid_voxel is None:
        gt_centroid_voxel = [np.nan, np.nan, np.nan]
        gt_centroid_world = [np.nan, np.nan, np.nan]

    return {
        "error": "",
        "shape_equal": bool(shape_equal),
        "affine_close": bool(affine_close),
        "pred_voxels": pred_voxels,
        "gt_voxels": gt_voxels,
        "intersection_voxels": intersection_voxels,
        "pred_volume_ml": float(pred_voxels * voxel_volume_mm3 / 1000.0),
        "gt_volume_ml": float(gt_voxels * voxel_volume_mm3 / 1000.0),
        "dice": dice,
        "centroid_distance_mm": centroid_distance_mm,
        "hd95_mm": hd95_mm,
        "assd_mm": assd_mm,
        "pred_centroid_voxel_x": float(pred_centroid_voxel[0]),
        "pred_centroid_voxel_y": float(pred_centroid_voxel[1]),
        "pred_centroid_voxel_z": float(pred_centroid_voxel[2]),
        "gt_centroid_voxel_x": float(gt_centroid_voxel[0]),
        "gt_centroid_voxel_y": float(gt_centroid_voxel[1]),
        "gt_centroid_voxel_z": float(gt_centroid_voxel[2]),
        "pred_centroid_world_x_mm": float(pred_centroid_world[0]),
        "pred_centroid_world_y_mm": float(pred_centroid_world[1]),
        "pred_centroid_world_z_mm": float(pred_centroid_world[2]),
        "gt_centroid_world_x_mm": float(gt_centroid_world[0]),
        "gt_centroid_world_y_mm": float(gt_centroid_world[1]),
        "gt_centroid_world_z_mm": float(gt_centroid_world[2]),
    }


def safe_float_summary(series):
    x = pd.to_numeric(series, errors="coerce").dropna()

    if len(x) == 0:
        return {
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
        }

    return {
        "mean": float(x.mean()),
        "median": float(x.median()),
        "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
        "min": float(x.min()),
        "max": float(x.max()),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate locked 24 CM segmentation predictions using Dice, centroid distance, HD95, and ASSD."
    )

    parser.add_argument("--locked_csv", required=True, type=Path)
    parser.add_argument("--pred_dir", required=True, type=Path)
    parser.add_argument("--gt_mask_dir", required=True, type=Path)
    parser.add_argument("--out_csv", required=True, type=Path)
    parser.add_argument("--out_json", required=True, type=Path)

    args = parser.parse_args()

    locked = pd.read_csv(args.locked_csv)

    if "pat_id" not in locked.columns:
        raise ValueError("locked_csv must contain column: pat_id")

    if len(locked) != 24:
        raise ValueError(f"Expected 24 locked patients, got {len(locked)}")

    rows = []

    for _, row in locked.iterrows():
        pat_id = row["pat_id"]
        patient_number = patient_number_from_pat_id(pat_id)

        try:
            pred_path = find_prediction(args.pred_dir, patient_number)
            gt_path = find_gt_mask(args.gt_mask_dir, patient_number)

            metrics = evaluate_case(pred_path, gt_path)

            result = {
                "pat_id": pat_id,
                "patient_number": patient_number,
                "prediction_file": str(pred_path),
                "gt_file": str(gt_path),
            }

            for extra_col in [
                "volume_quartile",
                "region",
                "contrast_quartile",
                "heterogeneity_quartile",
            ]:
                if extra_col in locked.columns:
                    result[extra_col] = row[extra_col]

            result.update(metrics)

        except Exception as e:
            result = {
                "pat_id": pat_id,
                "patient_number": patient_number,
                "prediction_file": "",
                "gt_file": "",
                "error": str(e),
            }

        rows.append(result)

        print(
            pat_id,
            "dice=", result.get("dice"),
            "centroid_mm=", result.get("centroid_distance_mm"),
            "hd95_mm=", result.get("hd95_mm"),
            "assd_mm=", result.get("assd_mm"),
            "error=", result.get("error"),
            flush=True,
        )

    df = pd.DataFrame(rows).sort_values("patient_number")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    valid = df[df["error"].fillna("") == ""].copy()
    errors = df[df["error"].fillna("") != ""].copy()

    summary = {
        "locked_csv": str(args.locked_csv),
        "pred_dir": str(args.pred_dir),
        "gt_mask_dir": str(args.gt_mask_dir),
        "n_locked_patients": int(len(df)),
        "n_valid_cases": int(len(valid)),
        "n_error_cases": int(len(errors)),
        "error_cases": errors[["pat_id", "error"]].to_dict(orient="records"),
        "dice": safe_float_summary(valid["dice"]) if len(valid) else {},
        "centroid_distance_mm": safe_float_summary(valid["centroid_distance_mm"]) if len(valid) else {},
        "hd95_mm": safe_float_summary(valid["hd95_mm"]) if len(valid) else {},
        "assd_mm": safe_float_summary(valid["assd_mm"]) if len(valid) else {},
    }

    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2, allow_nan=False)

    print()
    print("Saved case-level CSV:", args.out_csv)
    print("Saved summary JSON:", args.out_json)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
