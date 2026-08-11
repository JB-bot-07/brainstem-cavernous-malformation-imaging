# CavMal Project Workflow

## Project Goal

The long-term goal of this project is to develop imaging-based methods that can support anatomical characterization and eventual surgical-approach classification for brainstem cavernous malformations.

The project has progressed through several connected stages:

1. Direct MRI classification using BrainIAC
2. Automatic cavernous malformation lesion segmentation
3. Brainstem segmentation and anatomical localization
4. Template-to-patient ANTs registration
5. Future extraction of lesion and anatomical features for downstream surgical-approach modeling

---

## 1. BrainIAC Classification

Initial experiments tested whether pretrained BrainIAC representations could directly classify surgical approach from preoperative T1+C MRI.

Experiments included:

- Frozen BrainIAC feature extraction with an MLP classifier
- Partial fine-tuning of the final two BrainIAC blocks
- Partial fine-tuning of the final four BrainIAC blocks
- Five-fold cross-validation

These experiments established a baseline but motivated a shift toward more anatomically informed modeling.

See:

docs/BRAINIAC_CLASSIFICATION.md

---

## 2. Cavernous Malformation Lesion Segmentation

The next stage focused on automatic cavernous malformation segmentation using nnUNet.

A 124-case analysis cohort was characterized using:

- Lesion volume
- Brainstem region
- Lesion-to-background contrast
- Lesion heterogeneity

A locked 24-case internal test set was selected using constrained sampling.

The remaining 100 cases were used for nnUNet training and validation.

Held-out segmentation performance was evaluated using:

- Dice coefficient
- Centroid distance
- HD95
- ASSD

See:

docs/LESION_SEGMENTATION.md

and:

docs/TEST_SET_SELECTION.md

---

## 3. Brainstem Segmentation

A multi-label brainstem segmentation workflow was evaluated to identify:

- Midbrain
- Pons
- Medulla

The direct brainstem segmentation workflow uses an existing nnUNet-based model.

See:

docs/BRAINSTEM_SEGMENTATION.md

---

## 4. VS074-to-CavMal Registration

A minimally distorted T1+C template MRI, VS074, was selected as a common anatomical reference.

For each of the 124 CavMal patients:

VS074
-> rigid registration
-> affine registration
-> SyN deformable registration
-> warped VS074 in patient native space
-> propagated multi-label brainstem segmentation

The cavernous malformation is excluded from the SyN similarity calculation.

Three propagated brainstem outputs are retained:

1. Raw ANTs-transferred brainstem
2. Lesion-added brainstem
3. Final lesion-added + 2-voxel local-expansion brainstem

See:

docs/ANTS_REGISTRATION.md

---

## 5. Downstream Anatomical Features

The lesion and brainstem masks provide a framework for extracting anatomically meaningful variables including:

- Lesion centroid
- Lesion volume
- Brainstem region
- Lesion relationship to brainstem boundaries
- Spatial localization relative to midbrain, pons, and medulla

These features can be evaluated as inputs to classical machine-learning or other downstream surgical-approach classification models.

---

## Data Storage

Patient imaging, masks, model checkpoints, and large derived outputs remain on the institutional HPC.

The GitHub repository contains:

- Curated analysis scripts
- Reproducible pipeline scripts
- Documentation
- Small configuration files

See:

docs/HPC_DIRECTORY_MAP.md
