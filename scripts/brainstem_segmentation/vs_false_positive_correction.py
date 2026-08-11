#!/usr/bin/env python3
"""
Post-process a *folder* of auto-segmented vestibular-schwannoma masks by
removing false-positive components, using each patient’s brain-stem
labels (midbrain=1, pons=2, medulla=3) as anatomical reference.

Folder layout (example)
-----------------------
tumor_output_folder/
    VS_001.nii.gz
    VS_002.nii.gz
    ...
brainstem_output_folder/
    VS_001.nii.gz
    VS_002.nii.gz
    ...
postprocess_folder/
    VS_001.nii.gz
    VS_002.nii.gz
    ...
"""

import argparse
import concurrent.futures as cf
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import scipy.ndimage as ndi


# ─────────────────────────── Utility Loaders ────────────────────────────
def load_bool(path: Path):
    img = nib.load(str(path))
    return img.get_fdata().astype(bool), img


def load_int(path: Path):
    img = nib.load(str(path))
    return img.get_fdata().astype(np.int16)


# ───────────────────────── Brainstem Helpers ───────────────────────────
def z_medulla_max(bs: np.ndarray) -> int:
    zs = np.where(bs == 3)[2]
    if zs.size == 0:
        raise RuntimeError("No medulla voxels (label 3) in brain-stem mask!")
    return zs.max()


def z_medulla_pons_overlap(bs: np.ndarray) -> int:
    z_med = np.unique(np.where(bs == 3)[2])
    z_pon = np.unique(np.where(bs == 2)[2])
    overlap = np.intersect1d(z_med, z_pon)
    if overlap.size:
        return int(np.median(overlap))
    # fallback: midpoint between pons & medulla extents
    return int(0.5 * (z_med.max() + z_pon.min()))


# ───────────────────── Tumor Component Helpers ─────────────────────────
def connected_components(mask: np.ndarray):
    labeled, n = ndi.label(mask)
    slices = ndi.find_objects(labeled)
    return labeled, n, slices


def centroid_x(labeled, label_id):
    # weight=1 everywhere ⇒ geometric centroid
    cx, _, _ = ndi.center_of_mass(np.ones_like(labeled, dtype=np.uint8),
                                  labeled, label_id)
    return cx


def choose_component(slices, labeled, iac_z_low, iac_z_high, mid_x):
    # Components intersecting z range endpoints
    candidates = []
    for idx, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        z0, z1 = sl[2].start, sl[2].stop - 1
        if (z0 <= iac_z_low <= z1) or (z0 <= iac_z_high - 1 <= z1):
            candidates.append(idx)

    if candidates:
        dists = [abs(centroid_x(labeled, lb) - mid_x) for lb in candidates]
        return candidates[int(np.argmin(dists))]

    # fallback: component whose lower z is nearest iac_z_high
    best, best_dist = None, np.inf
    for idx, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        z0 = sl[2].start
        dist = abs(z0 - iac_z_high)
        if dist < best_dist:
            best, best_dist = idx, dist
    return best


# ───────────────────────── Single-Case Processing ──────────────────────────
def process_case(tumor_path: Path, brain_path: Path, out_path: Path):
    try:
        tumor_mask, tumor_img = load_bool(tumor_path)
        brainstem = load_int(brain_path)

        if tumor_mask.shape != brainstem.shape:
            raise ValueError("Shape mismatch between tumor and brain-stem masks")

        z_high = z_medulla_max(brainstem)
        z_low = z_medulla_pons_overlap(brainstem)

        labeled, n_comp, slcs = connected_components(tumor_mask)
        if n_comp == 0:
            raise RuntimeError("Empty tumor mask")

        keep = choose_component(slcs, labeled, z_low, z_high,
                                mid_x=tumor_mask.shape[0] / 2)
        cleaned = (labeled == keep).astype(np.uint8)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(cleaned, tumor_img.affine, tumor_img.header),
                 out_path)
        return f"{tumor_path.name}  done"
    except Exception as e:
        return f"{tumor_path.name}  failed  ({e})"


# ──────────────────────────── Main ──────────────────────────────
def main(args):
    tumor_dir  = Path(args.tumor)
    brain_dir  = Path(args.brainstem)
    out_dir    = Path(args.out)

    cases = []
    for tum_f in sorted(tumor_dir.glob("*.nii.gz")):
        pid     = tum_f.name.replace(".nii.gz", "")
        brain_f = brain_dir / f"{pid}.nii.gz"
        if not brain_f.exists():
            print(f"No brain-stem mask for {pid}, skipping")
            continue
        out_f   = out_dir / f"{pid}.nii.gz"
        cases.append((tum_f, brain_f, out_f))

    if not cases:
        print("No matching cases found."); return

    print(f"Processing {len(cases)} case(s)…")
    out_dir.mkdir(parents=True, exist_ok=True)
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for msg in ex.map(lambda t: process_case(*t), cases):
            print(msg)



if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Batch false-positive "
                                             "correction for VS masks")
    ap.add_argument("--tumor",      required=True,
                    help="Folder with raw VS+FP masks (*.nii.gz)")
    ap.add_argument("--brainstem",  required=True,
                    help="Folder with labelled brain-stem masks (*.nii.gz)")
    ap.add_argument("--out",        required=True,
                    help="Destination folder for cleaned masks")
    ap.add_argument("--workers", type=int, default=1,
                    help="Parallel workers (default 1 = serial)")
    main(ap.parse_args())
