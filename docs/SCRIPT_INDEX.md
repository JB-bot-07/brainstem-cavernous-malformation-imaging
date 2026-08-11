# Script Index

## Test Set Selection

### create_patient_region_labels_124.py
Creates brainstem-region labels for the 124-case analysis cohort.

### make_selection_master_csv_124.py
Creates the master 124-case table containing variables used for cohort characterization and test-set selection.

### run_make_selection_master_124.sh
HPC launcher for generating the master 124-case table.

### select_locked_internal_test_24.py
Performs constrained selection of the locked 24-case internal test set.

---

## Lesion Segmentation

### prepare_nnunet_folders_for_bao.py
Prepares the nnUNet lesion-segmentation dataset/workspace.

### evaluate_locked24_segmentation_4metrics.py
Evaluates locked 24-case lesion predictions using Dice, centroid distance, HD95, and ASSD.

---

## Brainstem Segmentation

### vs_pipeline.sh
Primary nnUNet-based brainstem-segmentation HPC pipeline.

### vs_false_positive_correction.py
Postprocessing utility for reducing false-positive brainstem predictions.

---

## ANTs Registration

### run_vs074_to_cavmal_124.sh
Primary 124-case VS074-to-CavMal registration and brainstem-label propagation pipeline.

The script performs:

- lesion-exclusion-mask creation
- rigid registration
- affine registration
- SyN deformable registration
- warped VS074 generation
- brainstem-label propagation
- raw brainstem preservation
- lesion addition
- 2-voxel lesion-local expansion
- output geometry checks
- label checks
- per-patient reporting

---

## BrainIAC

### train_frozen_brainiac_mlp_5fold_150ep_from_pt_clean.py
Five-fold frozen BrainIAC + MLP classification implementation.

### run_frozen_brainiac_mlp_5fold_150ep_from_pt_clean_cpu.sh
HPC launcher for the frozen BrainIAC experiment.

### train_partial_finetune_clean_best_and_final.py
Clean partial fine-tuning implementation.

### run_partial_last2_clean_best_and_final.sh
Runs partial fine-tuning with the final two BrainIAC blocks unfrozen.

### run_partial_last4_clean_best_and_final.sh
Runs partial fine-tuning with the final four BrainIAC blocks unfrozen.

---

## Figures

### make_locked24_metric_boxplots.py
Generates locked 24-case lesion-segmentation metric boxplots.

### make_bubble_plot_124.py
Generates a 124-case dataset-variability visualization.

### make_clean_comparison_figures.py
Generates cleaned model-comparison figures.

### make_5_key_training_figures.py
Generates selected BrainIAC training figures.
