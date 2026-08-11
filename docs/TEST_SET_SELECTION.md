# Locked Test Set Selection

## Purpose

A locked 24-case internal inference set was selected from the 124-case cavernous malformation cohort before final lesion-segmentation evaluation.

The goal was to create a held-out test set that represented the variability of the full cohort rather than relying on a simple random split.

## Source Cohort

The master 124-case table is stored on the HPC at:

/scratch/jbayasi/Cavmalproject1/results/lesion_segmentation_split_124/selection_master_124.csv

## Selection Variables

The cohort was characterized using:

- Lesion volume
- Brainstem region
- Lesion-to-background contrast
- Lesion heterogeneity

## Lesion Volume

Lesion volume was treated as the hard balancing constraint.

The 124 cases were sorted by lesion volume and divided into four quartiles:

- Q1
- Q2
- Q3
- Q4

The locked 24-case test set contains exactly six patients from each lesion-volume quartile.

This gives:

6 cases x 4 quartiles = 24 held-out cases

## Brainstem Region

Brainstem region was treated as a soft balancing constraint.

Regions include:

- Midbrain
- Pons
- Medulla

The test-set selection attempted to approximate the regional distribution of the complete 124-case cohort.

## Lesion Appearance

Two automated lesion-appearance measurements were included:

### Contrast

Characterizes lesion intensity relative to the surrounding neighborhood.

### Heterogeneity

Characterizes within-lesion intensity variability.

Contrast and heterogeneity were each divided into quartiles, and candidate test sets were scored according to how closely they represented the overall cohort distribution.

## Selection Procedure

Many candidate 24-case sets were generated while enforcing the exact six-per-volume-quartile requirement.

Candidate sets were then scored according to deviation from:

- Brainstem-region distribution
- Contrast-quartile distribution
- Heterogeneity-quartile distribution

The candidate with the best overall balance was selected and locked before final nnUNet evaluation.

## Final Split

Locked test set:

24 cases

Training/validation set:

100 cases

Important HPC files:

/scratch/jbayasi/Cavmalproject1/results/lesion_segmentation_split_124/locked_internal_test_24.csv

/scratch/jbayasi/Cavmalproject1/results/lesion_segmentation_split_124/trainval_100.csv

## Primary Scripts

scripts/test_set_selection/make_selection_master_csv_124.py

scripts/test_set_selection/create_patient_region_labels_124.py

scripts/test_set_selection/select_locked_internal_test_24.py

scripts/test_set_selection/run_make_selection_master_124.sh
