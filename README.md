# Brainstem Cavernous Malformation Imaging Project

This repository contains the curated scripts and documentation for an imaging and machine-learning project focused on brainstem cavernous malformations (BSCMs).

The project evaluates automatic lesion segmentation, brainstem localization, MRI registration, and imaging-based modeling as foundations for future surgical-approach classification.

## Current Project Components

### 1. Dataset Selection

A 124-case analysis cohort was characterized using lesion volume, brainstem region, lesion contrast, and lesion heterogeneity.

A locked 24-case internal test set was selected using constrained sampling, with the remaining 100 cases used for training and validation.

Scripts:

scripts/test_set_selection/

### 2. Cavernous Malformation Segmentation

nnUNet is used for automatic cavernous malformation lesion segmentation.

Held-out predictions are evaluated using Dice coefficient, centroid distance, HD95, and ASSD.

Scripts:

scripts/lesion_segmentation/

### 3. Brainstem Segmentation

An nnUNet-based workflow produces multi-label brainstem segmentations.

Label mapping:

- 1 = Midbrain
- 2 = Pons
- 3 = Medulla

Scripts:

scripts/brainstem_segmentation/

### 4. VS074-to-CavMal ANTs Registration

A minimally brainstem-distorted T1+C template MRI (VS074) is registered to each of the 124 CavMal patients using rigid, affine, and SyN registration.

The verified VS074 brainstem segmentation is then propagated into each patient's native MRI space.

The primary 124-case pipeline is:

scripts/registration/run_vs074_to_cavmal_124.sh

### 5. BrainIAC Classification

Earlier experiments evaluated frozen and partially fine-tuned BrainIAC representations for direct surgical-approach classification.

Scripts:

scripts/brainiac/

### 6. Figures

Curated figure-generation scripts are stored under:

scripts/figures/

## Repository Structure

Cavmalproject1_github/

- README.md
- config/
- docs/
- scripts/
  - brainiac/
  - brainstem_segmentation/
  - figures/
  - lesion_segmentation/
  - registration/
  - test_set_selection/

## Documentation

Start with:

- docs/PROJECT_WORKFLOW.md — overview of how the project components connect
- docs/HPC_DIRECTORY_MAP.md — locations of datasets and results on the HPC
- docs/ANTS_REGISTRATION.md — current 124-case VS074 registration workflow
- docs/BRAINSTEM_SEGMENTATION.md — direct brainstem segmentation workflow

Additional workflow-specific documentation is maintained in the docs/ directory.

## Data Availability

Patient MRI, patient-derived masks, large registration outputs, and model checkpoints are not stored in this repository.

These data remain on the institutional HPC.

See:

docs/HPC_DIRECTORY_MAP.md

for the corresponding HPC locations.

## Primary HPC Project Root

/scratch/jbayasi/Cavmalproject1
