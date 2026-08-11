# Brainstem Segmentation

## Purpose

This workflow generates multi-label brainstem segmentations from MRI using the existing nnUNet-based brainstem pipeline.

Brainstem label mapping:

- 1 = Midbrain
- 2 = Pons
- 3 = Medulla

## Main Scripts

### vs_pipeline.sh

Primary HPC brainstem-segmentation pipeline.

Original HPC location:

/scratch/jbayasi/vs_pipeline.sh

Curated repository copy:

scripts/brainstem_segmentation/vs_pipeline.sh

### vs_false_positive_correction.py

Postprocessing utility used by the brainstem segmentation workflow to reduce false-positive segmentation output.

Original HPC location:

/scratch/jbayasi/vs_false_positive_correction.py

Curated repository copy:

scripts/brainstem_segmentation/vs_false_positive_correction.py

## nnUNet Task

The brainstem segmentation workflow uses:

Task600_Brainstem

The pipeline expects nnUNet inference images to use the required modality suffix:

_0000.nii.gz

## Primary Brainstem Output

Final patient-level brainstem segmentation outputs are stored on the HPC rather than in GitHub.

Primary results directory:

/scratch/jbayasi/Cavmalproject1/results/brainstem_segmentation_unprocessednifti_final

Final mask set:

/scratch/jbayasi/Cavmalproject1/results/brainstem_segmentation_unprocessednifti_final/brainstem_masks_127_final

## Relationship to the ANTs Registration Workflow

The direct nnUNet brainstem segmentation workflow and the VS074-to-CavMal ANTs registration workflow are separate approaches.

The ANTs workflow uses a verified VS074 brainstem segmentation and propagates it into each CavMal patient's native MRI space.

See:

docs/ANTS_REGISTRATION.md

for documentation of the 124-case registration workflow.
