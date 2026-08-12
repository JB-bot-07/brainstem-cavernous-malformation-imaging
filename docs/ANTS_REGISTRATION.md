# VS074-to-CavMal ANTs Registration

## Purpose

This workflow registers a minimally brainstem-distorted T1+C template MRI, VS074, into the native space of each cavernous malformation patient.

The goal is to propagate a verified multi-label brainstem segmentation into each patient's native MRI space.

Brainstem labels:

- 1 = Midbrain
- 2 = Pons
- 3 = Medulla

## Registration Direction

Moving image:

VS074 T1+C MRI

Fixed image:

Native CavMal patient T1+C MRI

Therefore, the VS074 anatomy is transformed into each CavMal patient's native imaging space.

## VS074 Template

Moving MRI:

/scratch/jbayasi/Cavmalproject1/VS074_template/nifti/T1C_series.nii.gz

Moving brainstem segmentation:

/scratch/jbayasi/Cavmalproject1/VS074_template/brainstem/final/Sbl_Vs_Ret_074_brainstem.nii.gz

## 124-Case Cohort

The registration cohort is defined by:

/scratch/jbayasi/Cavmalproject1/results/vs074_to_cavmal_124/registration_cases_124.tsv

The TSV contains:

- pat_id
- region
- image_path
- mask_path

## Registration Pipeline

The main workflow is:

VS074
-> Rigid
-> Affine
-> SyN deformable registration
-> VS074 warped into patient native space
-> VS074 brainstem segmentation propagated into patient space

Primary registration settings:

- Rigid mask: none
- Affine mask: none
- SyN mask: lesion exclusion
- SyN gradient step: 0.2
- Histogram matching: off
- Precision: double

The cavernous malformation is excluded from the SyN similarity calculation using an inverted lesion mask.

## Main Script

Primary pipeline:

/scratch/jbayasi/Cavmalproject1/scripts/ants_cli_registration/run_vs074_to_cavmal_124.sh

Curated repository copy:

scripts/registration/run_vs074_to_cavmal_124.sh

The script is run as a Slurm array across the 124-patient cohort.

## Patient-Level Output Structure

Primary output root:

/scratch/jbayasi/Cavmalproject1/results/vs074_to_cavmal_124

Each patient has:

Bni_Cm_Ret_XXX/
- registration/
- masks/
- brainstem/
- pipeline_summary.txt
- registration_log.txt
- postprocessing_report.txt
- status.txt

## Registration Outputs

Within registration/:

- VS074_to_Bni_Cm_Ret_XXX_0GenericAffine.mat
- VS074_to_Bni_Cm_Ret_XXX_1Warp.nii.gz
- VS074_to_Bni_Cm_Ret_XXX_1InverseWarp.nii.gz
- VS074_to_Bni_Cm_Ret_XXX_Warped.nii.gz

The warped VS074 image is used for visual registration QC.

## Brainstem Label Propagation

The VS074 brainstem segmentation is transformed using the same forward transforms.

antsApplyTransforms is used with GenericLabel interpolation so integer segmentation labels are preserved.

## Preserved Brainstem Outputs

Three versions are retained for each patient.

### Raw transferred brainstem

<PATIENT>_brainstem_from_VS074.nii.gz

This is the untouched ANTs-transferred segmentation.

### Lesion-added brainstem

<PATIENT>_brainstem_lesion_added.nii.gz

Missing lesion voxels are assigned to the known brainstem region while existing nonzero regional labels are preserved.

### Final brainstem

<PATIENT>_brainstem_FINAL.nii.gz

This contains the lesion-added segmentation followed by a conservative 2-voxel lesion-local expansion.

The expansion is local and does not globally dilate the full brainstem mask.

## QC

All 124 registration outputs underwent visual QC by comparing the native CavMal MRI with the warped VS074 image and transferred brainstem segmentation.

The raw transferred brainstem output is preserved for every patient so registration quality can be assessed independently of later lesion-based postprocessing.

## Known Postprocessing Exceptions

Patients Bni_Cm_Ret_087 and Bni_Cm_Ret_089 contain fourth-ventricle/rhomboid lesions.

For these cases, the registration and raw transferred brainstem segmentation are acceptable, but the lesion should not be incorporated into the brainstem parenchymal segmentation.

The raw ANTs-transferred brainstem masks are therefore used instead of the lesion-added final masks.

## Developmental Registration Experiments

Earlier development experiments included:

- MNI-to-patient 026 affine registration
- MNI-to-patient 026 SyN registration
- lesion exclusion testing
- gradient-step testing
- histogram-matching testing
- T1+C patient 104 to patient 026 registration

These remain on the HPC for provenance but are not the primary 124-case workflow.

---

## Case-Specific QC and Exceptions

Patient-specific registration and brainstem-mask QC decisions are documented separately in:

docs/BRAINSTEM_REGISTRATION_QC.md

This document includes the final handling decisions for Bni_Cm_Ret_066, Bni_Cm_Ret_087, Bni_Cm_Ret_089, and Bni_Cm_Ret_116.
