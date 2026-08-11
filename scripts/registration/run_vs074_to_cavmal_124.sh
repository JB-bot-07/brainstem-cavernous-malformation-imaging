#!/bin/bash
#SBATCH --job-name=VS074_CM
#SBATCH --partition=legion1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/jbayasi/Cavmalproject1/logs/vs074_to_cavmal_124/%A_%a.out
#SBATCH --error=/scratch/jbayasi/Cavmalproject1/logs/vs074_to_cavmal_124/%A_%a.err

set -euo pipefail


# ============================================================
# SOFTWARE
# ============================================================

ANTS_BIN=/scratch/jbayasi/conda_envs/ants_cli/bin
PYTHON_BIN=/scratch/jbayasi/brainiac_py39/bin/python

export PATH="${ANTS_BIN}:$PATH"
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="${SLURM_CPUS_PER_TASK}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1


# ============================================================
# VS074 TEMPLATE
# ============================================================

MOVING=/scratch/jbayasi/Cavmalproject1/VS074_template/nifti/T1C_series.nii.gz

MOVING_SEG=/scratch/jbayasi/Cavmalproject1/VS074_template/brainstem/final/Sbl_Vs_Ret_074_brainstem.nii.gz


# ============================================================
# 124-PATIENT COHORT
# ============================================================

CASE_FILE=/scratch/jbayasi/Cavmalproject1/results/vs074_to_cavmal_124/registration_cases_124.tsv

BASE_OUT=/scratch/jbayasi/Cavmalproject1/results/vs074_to_cavmal_124


# ============================================================
# READ PATIENT CORRESPONDING TO ARRAY INDEX
#
# TSV:
# pat_id    region    image_path    mask_path
#
# Array 1 = first patient row after header.
# ============================================================

LINE_NUMBER=$((SLURM_ARRAY_TASK_ID + 1))
LINE=$(sed -n "${LINE_NUMBER}p" "$CASE_FILE")

if [[ -z "$LINE" ]]; then
    echo "ERROR: No patient found for array index ${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

IFS=$'\t' read -r PAT_ID REGION FIXED LESION_MASK <<< "$LINE"


# ============================================================
# REGION -> BRAINSTEM LABEL
#
# Confirmed mapping:
# 1 = Midbrain
# 2 = Pons
# 3 = Medulla
# ============================================================

case "$REGION" in

    Midbrain)
        REGION_LABEL=1
        ;;

    Pons)
        REGION_LABEL=2
        ;;

    Medulla)
        REGION_LABEL=3
        ;;

    *)
        echo "ERROR: Unknown region '${REGION}' for ${PAT_ID}"
        exit 1
        ;;
esac


# ============================================================
# PATIENT OUTPUT DIRECTORIES
# ============================================================

PAT_OUT="${BASE_OUT}/${PAT_ID}"

REG_OUT="${PAT_OUT}/registration"
MASK_OUT="${PAT_OUT}/masks"
BS_OUT="${PAT_OUT}/brainstem"

mkdir -p "$REG_OUT" "$MASK_OUT" "$BS_OUT"

STATUS="${PAT_OUT}/status.txt"
POST_REPORT="${PAT_OUT}/postprocessing_report.txt"

PREFIX="${REG_OUT}/VS074_to_${PAT_ID}_"

FIXED_MASK="${MASK_OUT}/lesion_exclusion_mask.nii.gz"

BS_RAW="${BS_OUT}/${PAT_ID}_brainstem_from_VS074.nii.gz"

BS_LESION="${BS_OUT}/${PAT_ID}_brainstem_lesion_added.nii.gz"

BS_FINAL="${BS_OUT}/${PAT_ID}_brainstem_FINAL.nii.gz"


# ============================================================
# PIPELINE SUMMARY
# ============================================================

{
echo "============================================================"
echo "VS074 -> ${PAT_ID}"
echo "============================================================"
echo "Array task:              ${SLURM_ARRAY_TASK_ID}"
echo "Patient:                 ${PAT_ID}"
echo "Region:                  ${REGION}"
echo "Regional label:          ${REGION_LABEL}"
echo "Fixed MRI:               ${FIXED}"
echo "Lesion mask:             ${LESION_MASK}"
echo "Moving VS template:      ${MOVING}"
echo "Moving brainstem mask:   ${MOVING_SEG}"
echo
echo "Registration:"
echo "  Rigid -> Affine -> SyN"
echo "  Rigid mask:            none"
echo "  Affine mask:           none"
echo "  SyN mask:              lesion exclusion"
echo "  SyN gradient:          0.2"
echo "  Histogram matching:    OFF"
echo "  Precision:             double"
echo
echo "Postprocessing:"
echo "  1. Preserve raw transformed brainstem"
echo "  2. Add lesion voxels only where current label = 0"
echo "  3. Preserve existing nonzero regional labels"
echo "  4. Apply 2-voxel lesion-local regional expansion"
echo "============================================================"
} | tee "${PAT_OUT}/pipeline_summary.txt"


# ============================================================
# VERIFY ALL REQUIRED FILES
# ============================================================

for REQUIRED in \
    "${ANTS_BIN}/antsRegistration" \
    "${ANTS_BIN}/antsRegistrationSyN.sh" \
    "${ANTS_BIN}/antsApplyTransforms" \
    "$PYTHON_BIN" \
    "$MOVING" \
    "$MOVING_SEG" \
    "$FIXED" \
    "$LESION_MASK"
do
    if [[ ! -e "$REQUIRED" ]]; then
        echo "FAILED: missing required path: $REQUIRED" | tee "$STATUS"
        exit 1
    fi
done


# ============================================================
# CHECK FIXED MRI / LESION GEOMETRY
# CREATE LESION-EXCLUSION MASK
#
# Original lesion:
#   1 = lesion
#   0 = everything else
#
# Registration exclusion:
#   0 = lesion ignored
#   1 = everything else used
# ============================================================

"$PYTHON_BIN" - "$FIXED" "$LESION_MASK" "$FIXED_MASK" <<'PY'
import sys
import nibabel as nib
import numpy as np

fixed_path, lesion_path, out_path = sys.argv[1:4]

fixed = nib.load(fixed_path)
lesion = nib.load(lesion_path)

print("Fixed shape:", fixed.shape)
print("Lesion shape:", lesion.shape)

if fixed.shape != lesion.shape:
    raise RuntimeError(
        f"Fixed/lesion shape mismatch: "
        f"{fixed.shape} vs {lesion.shape}"
    )

if not np.allclose(
    fixed.affine,
    lesion.affine,
    atol=1e-4
):
    raise RuntimeError(
        "Fixed MRI and lesion mask affine mismatch"
    )

lesion_data = np.asanyarray(lesion.dataobj)

lesion_bool = lesion_data > 0

if lesion_bool.sum() == 0:
    raise RuntimeError(
        "Lesion mask contains zero lesion voxels"
    )

exclusion = (~lesion_bool).astype(np.uint8)

out = nib.Nifti1Image(
    exclusion,
    fixed.affine,
    fixed.header
)

out.set_data_dtype(np.uint8)
nib.save(out, out_path)

print("Lesion voxels:", int(lesion_bool.sum()))
print("Saved exclusion mask:", out_path)
PY


# ============================================================
# FULL VS074 -> PATIENT REGISTRATION
#
# Moving = VS074
# Fixed = CavMal patient
#
# Rigid: no mask
# Affine: no mask
# SyN: lesion exclusion
# ============================================================

echo "REGISTRATION_STARTED" > "$STATUS"

antsRegistrationSyN.sh \
    -d 3 \
    -f "$FIXED" \
    -m "$MOVING" \
    -o "$PREFIX" \
    -t s \
    -x NULL \
    -x NULL \
    -x "$FIXED_MASK" \
    -g 0.2 \
    -n "${SLURM_CPUS_PER_TASK}" \
    -p d \
    -j 0 \
    2>&1 | tee "${PAT_OUT}/registration_log.txt"


# ============================================================
# VERIFY REGISTRATION OUTPUTS
# ============================================================

AFFINE="${PREFIX}0GenericAffine.mat"
WARP="${PREFIX}1Warp.nii.gz"
INVWARP="${PREFIX}1InverseWarp.nii.gz"
WARPED="${PREFIX}Warped.nii.gz"
INVWARPED="${PREFIX}InverseWarped.nii.gz"

for OUTFILE in \
    "$AFFINE" \
    "$WARP" \
    "$INVWARP" \
    "$WARPED"
do
    if [[ ! -f "$OUTFILE" ]]; then
        echo "FAILED: registration output missing: $OUTFILE" | tee "$STATUS"
        exit 1
    fi
done


# ============================================================
# TRANSFORM VS074 BRAINSTEM INTO PATIENT SPACE
#
# Raw output - never modified after this
# ============================================================

echo "TRANSFORMING_BRAINSTEM" > "$STATUS"

antsApplyTransforms \
    -d 3 \
    -i "$MOVING_SEG" \
    -r "$FIXED" \
    -o "$BS_RAW" \
    -n GenericLabel \
    -t "$WARP" \
    -t "$AFFINE" \
    --float 0 \
    -v 1


if [[ ! -f "$BS_RAW" ]]; then
    echo "FAILED: raw transformed brainstem was not created" | tee "$STATUS"
    exit 1
fi


# ============================================================
# BRAINSTEM POSTPROCESSING
#
# Output 1:
#   *_brainstem_from_VS074.nii.gz
#   untouched registration result
#
# Output 2:
#   *_brainstem_lesion_added.nii.gz
#   only missing lesion voxels added
#
# Output 3:
#   *_brainstem_FINAL.nii.gz
#   lesion-added output +
#   conservative 2-voxel lesion-local expansion
# ============================================================

echo "POSTPROCESSING_BRAINSTEM" > "$STATUS"

"$PYTHON_BIN" - \
    "$BS_RAW" \
    "$LESION_MASK" \
    "$BS_LESION" \
    "$BS_FINAL" \
    "$REGION_LABEL" \
    "$PAT_ID" \
    "$REGION" \
    "$POST_REPORT" <<'PY'

import sys
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure

(
    bs_raw_path,
    lesion_path,
    bs_lesion_path,
    bs_final_path,
    region_label,
    pat_id,
    region,
    report_path
) = sys.argv[1:9]

region_label = int(region_label)

bs_img = nib.load(bs_raw_path)
lesion_img = nib.load(lesion_path)

bs = np.asanyarray(
    bs_img.dataobj
).astype(np.uint8)

lesion = (
    np.asanyarray(
        lesion_img.dataobj
    ) > 0
)

# ------------------------------------------------------------
# Geometry validation
# ------------------------------------------------------------

if bs.shape != lesion.shape:
    raise RuntimeError(
        f"Brainstem / lesion shape mismatch: "
        f"{bs.shape} vs {lesion.shape}"
    )

if not np.allclose(
    bs_img.affine,
    lesion_img.affine,
    atol=1e-4
):
    raise RuntimeError(
        "Brainstem / lesion affine mismatch"
    )


# ------------------------------------------------------------
# Statistics before correction
# ------------------------------------------------------------

total_lesion = int(lesion.sum())

inside_any_brainstem = lesion & (bs > 0)

already_inside = int(
    inside_any_brainstem.sum()
)

missing_lesion = (
    lesion & (bs == 0)
)

newly_added = int(
    missing_lesion.sum()
)


# ------------------------------------------------------------
# Count lesion voxels already assigned to each label
# ------------------------------------------------------------

inside_label_1 = int(
    np.sum(lesion & (bs == 1))
)

inside_label_2 = int(
    np.sum(lesion & (bs == 2))
)

inside_label_3 = int(
    np.sum(lesion & (bs == 3))
)

conflicting_existing = int(
    np.sum(
        lesion
        & (bs > 0)
        & (bs != region_label)
    )
)


# ============================================================
# OUTPUT 2:
# ADD ONLY MISSING LESION VOXELS
# ============================================================

lesion_added = bs.copy()

lesion_added[
    missing_lesion
] = region_label


out_lesion = nib.Nifti1Image(
    lesion_added,
    bs_img.affine,
    bs_img.header
)

out_lesion.set_data_dtype(
    np.uint8
)

nib.save(
    out_lesion,
    bs_lesion_path
)


# ============================================================
# OUTPUT 3:
# 2-VOXEL LOCAL EXPANSION
#
# Only expand:
#   - background voxels
#   - near lesion
#   - near target regional label
#
# Does NOT globally dilate brainstem.
# ============================================================

structure = generate_binary_structure(
    3,
    1
)

target_region = (
    lesion_added == region_label
)

background = (
    lesion_added == 0
)

lesion_neighborhood = binary_dilation(
    lesion,
    structure=structure,
    iterations=2
)

region_expanded = binary_dilation(
    target_region,
    structure=structure,
    iterations=2
)

local_add = (
    background
    & lesion_neighborhood
    & region_expanded
)

final = lesion_added.copy()

final[
    local_add
] = region_label

local_expansion_voxels = int(
    local_add.sum()
)


out_final = nib.Nifti1Image(
    final,
    bs_img.affine,
    bs_img.header
)

out_final.set_data_dtype(
    np.uint8
)

nib.save(
    out_final,
    bs_final_path
)


# ------------------------------------------------------------
# Final lesion coverage
# ------------------------------------------------------------

final_coverage = int(
    np.sum(
        lesion & (final > 0)
    )
)


# ------------------------------------------------------------
# Report
# ------------------------------------------------------------

report = f"""
Patient: {pat_id}
Known region: {region}
Regional label: {region_label}

Total lesion voxels: {total_lesion}

Lesion already inside transformed brainstem:
{already_inside}

Already labeled Midbrain (1):
{inside_label_1}

Already labeled Pons (2):
{inside_label_2}

Already labeled Medulla (3):
{inside_label_3}

Existing lesion voxels with a different regional label:
{conflicting_existing}

Missing lesion voxels added:
{newly_added}

Additional voxels added by 2-voxel local expansion:
{local_expansion_voxels}

Final lesion coverage:
{final_coverage} / {total_lesion}

Raw transformed brainstem:
{bs_raw_path}

Lesion-added brainstem:
{bs_lesion_path}

Final locally expanded brainstem:
{bs_final_path}
"""

print(report)

with open(
    report_path,
    "w"
) as f:
    f.write(report)

PY


# ============================================================
# FINAL GEOMETRY / LABEL QC
# ============================================================

"$PYTHON_BIN" - \
    "$FIXED" \
    "$WARPED" \
    "$BS_RAW" \
    "$BS_LESION" \
    "$BS_FINAL" <<'PY'

import sys
import nibabel as nib
import numpy as np

(
    fixed_path,
    warped_path,
    raw_path,
    lesion_path,
    final_path
) = sys.argv[1:6]

fixed = nib.load(fixed_path)
warped = nib.load(warped_path)
raw = nib.load(raw_path)
lesion_added = nib.load(lesion_path)
final = nib.load(final_path)

for name, img in [
    ("Warped VS074", warped),
    ("Raw brainstem", raw),
    ("Lesion-added brainstem", lesion_added),
    ("Final brainstem", final),
]:

    if fixed.shape != img.shape:
        raise RuntimeError(
            f"{name} shape does not match fixed MRI"
        )

    if not np.allclose(
        fixed.affine,
        img.affine,
        atol=1e-4
    ):
        raise RuntimeError(
            f"{name} affine does not match fixed MRI"
        )


final_data = np.asanyarray(
    final.dataobj
)

labels, counts = np.unique(
    final_data,
    return_counts=True
)

print("Final brainstem labels:")

for label, count in zip(
    labels,
    counts
):
    print(label, count)


allowed = {0, 1, 2, 3}

present = set(
    int(x)
    for x in labels
)

unexpected = (
    present - allowed
)

if unexpected:
    raise RuntimeError(
        f"Unexpected labels: {unexpected}"
    )

if present == {0}:
    raise RuntimeError(
        "Final brainstem mask is empty"
    )

print(
    "FINAL_GEOMETRY_AND_LABEL_CHECK_PASSED"
)

PY


# ============================================================
# SUCCESS
# ============================================================

{
echo "COMPLETED"
echo "Patient: ${PAT_ID}"
echo "Region: ${REGION}"
echo
echo "Fixed MRI:"
echo "${FIXED}"
echo
echo "Warped VS074:"
echo "${WARPED}"
echo
echo "Raw brainstem:"
echo "${BS_RAW}"
echo
echo "Lesion-added brainstem:"
echo "${BS_LESION}"
echo
echo "Final brainstem:"
echo "${BS_FINAL}"
} > "$STATUS"


echo
echo "============================================================"
echo "SUCCESS: ${PAT_ID}"
echo "============================================================"

echo
echo "Warped VS074:"
echo "$WARPED"

echo
echo "Raw transformed brainstem:"
echo "$BS_RAW"

echo
echo "Lesion-added brainstem:"
echo "$BS_LESION"

echo
echo "Final 2-voxel locally expanded brainstem:"
echo "$BS_FINAL"

echo
echo "Postprocessing report:"
echo "$POST_REPORT"

echo
echo "============================================================"
