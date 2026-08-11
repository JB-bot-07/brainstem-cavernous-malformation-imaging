#!/usr/bin/env python

from pathlib import Path
import re
import shutil

import nibabel as nib
import numpy as np
import pandas as pd


BASE = Path("/scratch/jbayasi/Cavmalproject1")

SOURCE_IMG_DIR = BASE / "unprocessednifti"
SOURCE_MASK_DIR = BASE / "semiautomasks"

SPLIT_DIR = BASE / "results/lesion_segmentation_split_124"

TRAINVAL_CSV = SPLIT_DIR / "trainval_100_clean_for_bao.csv"
LOCKED_TEST_CSV = SPLIT_DIR / "locked_internal_test_24_clean_for_bao.csv"

OUT_BASE = SPLIT_DIR / "nnunet_folders_for_bao"

IMAGES_TR = OUT_BASE / "imagesTr_100"
LABELS_TR = OUT_BASE / "labelsTr_100"
IMAGES_TS = OUT_BASE / "imagesTs_locked24"
MANIFEST_DIR = OUT_BASE / "manifests"


def pat_to_bni_cm_name(pat_id):
    """
    Converts:
      Bni_Cm_Ret_001 -> Bni_Cm_001
    """
    m = re.search(r"(\d+)$", str(pat_id))
    if not m:
        raise ValueError(f"Could not parse patient number from {pat_id}")
    return f"Bni_Cm_{int(m.group(1)):03d}"


def find_source_image(pat_id):
    p = SOURCE_IMG_DIR / f"{pat_id}.nii.gz"
    if not p.exists():
        raise FileNotFoundError(f"Missing source image: {p}")
    return p


def find_source_mask(pat_id):
    p = SOURCE_MASK_DIR / f"{pat_id}_semiautomask.nii.gz"
    if p.exists():
        return p

    matches = sorted(SOURCE_MASK_DIR.glob(f"{pat_id}*.nii*"))
    if len(matches) == 1:
        return matches[0]

    raise FileNotFoundError(f"Missing or ambiguous mask for {pat_id}: {matches}")


def save_binary_mask_like_image(mask_path, image_path, output_path):
    img_nii = nib.load(str(image_path))
    mask_nii = nib.load(str(mask_path))

    if img_nii.shape != mask_nii.shape:
        raise ValueError(
            f"Shape mismatch for {image_path.name}: "
            f"image {img_nii.shape}, mask {mask_nii.shape}"
        )

    # We already fixed the source masks, but keep this check strict.
    if not np.allclose(img_nii.affine, mask_nii.affine, atol=1e-3):
        raise ValueError(
            f"Affine mismatch for {image_path.name}: "
            f"{image_path} vs {mask_path}"
        )

    mask_data = mask_nii.get_fdata(dtype=np.float32)
    binary = (np.isfinite(mask_data) & (mask_data > 0)).astype(np.uint8)

    if int(binary.sum()) == 0:
        raise ValueError(f"Empty mask after binarization: {mask_path}")

    label_img = nib.Nifti1Image(
        binary,
        affine=img_nii.affine,
        header=img_nii.header.copy(),
    )

    label_img.set_data_dtype(np.uint8)

    qform, qcode = img_nii.get_qform(coded=True)
    sform, scode = img_nii.get_sform(coded=True)

    if qcode is None:
        qcode = 1
    if scode is None:
        scode = 1

    label_img.set_qform(qform, int(qcode))
    label_img.set_sform(sform, int(scode))
    label_img.header.set_slope_inter(1.0, 0.0)

    nib.save(label_img, str(output_path))


def prepare_trainval():
    df = pd.read_csv(TRAINVAL_CSV)

    if len(df) != 100:
        raise ValueError(f"Expected 100 trainval patients, got {len(df)}")

    rows = []

    for pat_id in df["pat_id"]:
        short_name = pat_to_bni_cm_name(pat_id)

        src_img = find_source_image(pat_id)
        src_mask = find_source_mask(pat_id)

        dst_img = IMAGES_TR / f"{short_name}_0000.nii.gz"
        dst_mask = LABELS_TR / f"{short_name}.nii.gz"

        shutil.copy2(src_img, dst_img)
        save_binary_mask_like_image(src_mask, src_img, dst_mask)

        rows.append({
            "pat_id": pat_id,
            "nnunet_case_id": short_name,
            "source_image": str(src_img),
            "source_mask": str(src_mask),
            "nnunet_image": str(dst_img),
            "nnunet_label": str(dst_mask),
        })

    out_manifest = MANIFEST_DIR / "trainval_100_nnunet_manifest.csv"
    pd.DataFrame(rows).to_csv(out_manifest, index=False)

    print("Prepared train/val images:", len(rows))
    print("Saved manifest:", out_manifest)


def prepare_locked_test_images():
    df = pd.read_csv(LOCKED_TEST_CSV)

    if len(df) != 24:
        raise ValueError(f"Expected 24 locked test patients, got {len(df)}")

    rows = []

    for pat_id in df["pat_id"]:
        short_name = pat_to_bni_cm_name(pat_id)

        src_img = find_source_image(pat_id)
        dst_img = IMAGES_TS / f"{short_name}_0000.nii.gz"

        shutil.copy2(src_img, dst_img)

        rows.append({
            "pat_id": pat_id,
            "nnunet_case_id": short_name,
            "source_image": str(src_img),
            "nnunet_image": str(dst_img),
        })

    out_manifest = MANIFEST_DIR / "locked_test_24_imagesTs_manifest.csv"
    pd.DataFrame(rows).to_csv(out_manifest, index=False)

    print("Prepared locked test images:", len(rows))
    print("Saved manifest:", out_manifest)


def final_checks():
    n_img_tr = len(list(IMAGES_TR.glob("*.nii.gz")))
    n_lab_tr = len(list(LABELS_TR.glob("*.nii.gz")))
    n_img_ts = len(list(IMAGES_TS.glob("*.nii.gz")))

    print()
    print("Final counts:")
    print("imagesTr_100:", n_img_tr)
    print("labelsTr_100:", n_lab_tr)
    print("imagesTs_locked24:", n_img_ts)

    if n_img_tr != 100:
        raise ValueError(f"Expected 100 training images, found {n_img_tr}")
    if n_lab_tr != 100:
        raise ValueError(f"Expected 100 training labels, found {n_lab_tr}")
    if n_img_ts != 24:
        raise ValueError(f"Expected 24 locked test images, found {n_img_ts}")

    image_cases = {p.name.replace("_0000.nii.gz", "") for p in IMAGES_TR.glob("*.nii.gz")}
    label_cases = {p.name.replace(".nii.gz", "") for p in LABELS_TR.glob("*.nii.gz")}

    if image_cases != label_cases:
        raise ValueError(
            f"Training image/label case mismatch. "
            f"Image-only: {sorted(image_cases - label_cases)}; "
            f"Label-only: {sorted(label_cases - image_cases)}"
        )

    test_cases = {p.name.replace("_0000.nii.gz", "") for p in IMAGES_TS.glob("*.nii.gz")}
    overlap = image_cases & test_cases

    if overlap:
        raise ValueError(f"Locked test cases overlap with training cases: {sorted(overlap)}")

    print("Image/label names match.")
    print("No train/test overlap.")
    print("Folder prep complete.")


def main():
    for d in [IMAGES_TR, LABELS_TR, IMAGES_TS, MANIFEST_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # Clear previous contents so reruns are clean.
    for d in [IMAGES_TR, LABELS_TR, IMAGES_TS]:
        for f in d.glob("*.nii.gz"):
            f.unlink()

    prepare_trainval()
    prepare_locked_test_images()
    final_checks()


if __name__ == "__main__":
    main()
