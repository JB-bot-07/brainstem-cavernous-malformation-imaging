# CavMal HPC Directory Map

## Project Root

/scratch/jbayasi/Cavmalproject1

Large imaging data, patient-derived masks, model checkpoints, and registration outputs remain on the institutional HPC and are not stored in this GitHub repository.

## Native Imaging

Original DICOM:

/scratch/jbayasi/Cavmalproject1/dicom

Original native T1+C NIfTI:

/scratch/jbayasi/Cavmalproject1/unprocessednifti

BrainIAC-preprocessed NIfTI:

/scratch/jbayasi/Cavmalproject1/processednifti

## Cavernous Malformation Masks

Semi-automatic lesion masks:

/scratch/jbayasi/Cavmalproject1/semiautomasks

## Dataset Selection

/scratch/jbayasi/Cavmalproject1/results/lesion_segmentation_split_124

Important files include:

- selection_master_124.csv
- patient_region_labels_124.csv
- locked_internal_test_24.csv
- trainval_100.csv

The master 124-case table is the source of truth for the primary analysis cohort.

## Locked 24-Case Lesion Segmentation Evaluation

/scratch/jbayasi/Cavmalproject1/results/locked24_final_metrics

Contains case-level and summary lesion segmentation metrics.

## Direct Brainstem Segmentation

Primary results directory:

/scratch/jbayasi/Cavmalproject1/results/brainstem_segmentation_unprocessednifti_final

Final mask set:

/scratch/jbayasi/Cavmalproject1/results/brainstem_segmentation_unprocessednifti_final/brainstem_masks_127_final

## VS074 Template

T1+C MRI:

/scratch/jbayasi/Cavmalproject1/VS074_template/nifti/T1C_series.nii.gz

Verified brainstem segmentation:

/scratch/jbayasi/Cavmalproject1/VS074_template/brainstem/final/Sbl_Vs_Ret_074_brainstem.nii.gz

## 124-Case VS074-to-CavMal Registration

Primary output root:

/scratch/jbayasi/Cavmalproject1/results/vs074_to_cavmal_124

Cohort definition:

/scratch/jbayasi/Cavmalproject1/results/vs074_to_cavmal_124/registration_cases_124.tsv

Primary pipeline:

/scratch/jbayasi/Cavmalproject1/scripts/ants_cli_registration/run_vs074_to_cavmal_124.sh

Each patient contains:

- registration transforms
- warped VS074 MRI
- lesion-exclusion mask
- raw transferred brainstem mask
- lesion-added brainstem mask
- final locally expanded brainstem mask
- registration log
- pipeline summary
- postprocessing report
- status file

## BrainIAC Classification Results

Frozen BrainIAC:

/scratch/jbayasi/Cavmalproject1/results/frozen_brainiac_mlp_5fold_150ep_from_pt_clean

Partial fine-tuning, last two blocks:

/scratch/jbayasi/Cavmalproject1/results/partial_finetune_clean_last2blocks_150ep_best_and_final

Partial fine-tuning, last four blocks:

/scratch/jbayasi/Cavmalproject1/results/partial_finetune_clean_last4blocks_150ep_best_and_final

## Development Scripts

Full development script directory:

/scratch/jbayasi/Cavmalproject1/scripts

The GitHub repository contains only a curated subset of the primary reproducible workflows.
