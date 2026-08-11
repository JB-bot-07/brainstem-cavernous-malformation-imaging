# Cavernous Malformation Lesion Segmentation

## Purpose

This workflow performs automatic segmentation of brainstem cavernous malformations from preoperative MRI.

The goal is to generate patient-specific lesion masks that can support quantitative anatomical analysis and future surgical-approach modeling.

## Imaging

Primary native T1+C NIfTI images:

/scratch/jbayasi/Cavmalproject1/unprocessednifti

## Ground-Truth Lesion Masks

Semi-automatic lesion masks:

/scratch/jbayasi/Cavmalproject1/semiautomasks

These masks were used as lesion-segmentation ground truth.

## Analysis Cohort

The primary analysis cohort contains 124 cases.

The split consists of:

- 100 training/validation cases
- 24 locked held-out inference cases

See:

docs/TEST_SET_SELECTION.md

## Model

Lesion segmentation was performed using nnUNet v2.

The nnUNet workspace is stored under:

/scratch/jbayasi/Cavmalproject1/results/lesion_segmentation_split_124

The associated nnUNet directories include:

- nnunet_raw
- nnunet_preprocessed
- nnunet_results

## Locked Test Evaluation

Final evaluation was performed on the locked 24-case inference set.

Metrics include:

### Dice coefficient

Measures spatial overlap between the predicted lesion mask and reference lesion mask.

### Centroid distance

Measures the Euclidean distance between predicted and reference lesion centroids.

Reported in millimeters.

### HD95

95th-percentile Hausdorff distance.

Reported in millimeters.

### ASSD

Average symmetric surface distance.

Reported in millimeters.

## Final Evaluation Results

Primary output directory:

/scratch/jbayasi/Cavmalproject1/results/locked24_final_metrics

Case-level metrics:

/scratch/jbayasi/Cavmalproject1/results/locked24_final_metrics/locked24_dice_centroid_hd95_assd_case_metrics.csv

Summary metrics:

/scratch/jbayasi/Cavmalproject1/results/locked24_final_metrics/locked24_dice_centroid_hd95_assd_summary.json

Predicted masks:

/scratch/jbayasi/Cavmalproject1/results/locked24_final_metrics/Predictedmasks_24locked_semiauto

## Primary Scripts

### prepare_nnunet_folders_for_bao.py

Prepares the nnUNet dataset/workspace.

Repository location:

scripts/lesion_segmentation/prepare_nnunet_folders_for_bao.py

### evaluate_locked24_segmentation_4metrics.py

Computes Dice, centroid distance, HD95, and ASSD for the locked 24-case test set.

Repository location:

scripts/lesion_segmentation/evaluate_locked24_segmentation_4metrics.py

## Downstream Use

Automatic lesion masks can be used to derive features such as:

- Lesion centroid
- Lesion volume
- Brainstem region
- Spatial relationship to brainstem anatomy

These features form a foundation for future surgical-approach classification.
