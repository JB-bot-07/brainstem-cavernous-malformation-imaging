# BrainIAC Surgical-Approach Classification Experiments

## Purpose

Initial experiments evaluated whether pretrained BrainIAC MRI representations could be used to classify surgical approach for brainstem cavernous malformations directly from preoperative T1+C MRI.

These experiments served as an early baseline before the project shifted toward lesion segmentation and anatomical localization.

## Evaluation

Experiments used five-fold stratified cross-validation.

Primary evaluation metrics included:

- Balanced accuracy
- Macro F1
- Accuracy
- Weighted F1

Balanced accuracy was emphasized because of class imbalance.

## Frozen BrainIAC

The first clean workflow kept the BrainIAC backbone frozen and trained an MLP classification head using BrainIAC-derived representations.

Primary result directory:

/scratch/jbayasi/Cavmalproject1/results/frozen_brainiac_mlp_5fold_150ep_from_pt_clean

Repository scripts:

scripts/brainiac/train_frozen_brainiac_mlp_5fold_150ep_from_pt_clean.py

scripts/brainiac/run_frozen_brainiac_mlp_5fold_150ep_from_pt_clean_cpu.sh

## Partial Fine-Tuning

Additional experiments partially fine-tuned the BrainIAC backbone.

### Final two blocks unfrozen

Results:

/scratch/jbayasi/Cavmalproject1/results/partial_finetune_clean_last2blocks_150ep_best_and_final

Launcher:

scripts/brainiac/run_partial_last2_clean_best_and_final.sh

### Final four blocks unfrozen

Results:

/scratch/jbayasi/Cavmalproject1/results/partial_finetune_clean_last4blocks_150ep_best_and_final

Launcher:

scripts/brainiac/run_partial_last4_clean_best_and_final.sh

Shared training implementation:

scripts/brainiac/train_partial_finetune_clean_best_and_final.py

## Interpretation

Direct surgical-approach classification performance remained limited.

These experiments motivated the project pivot toward explicitly modeling lesion location and brainstem anatomy rather than relying solely on global MRI representations.

The subsequent workflow therefore focused on:

- Automatic cavernous malformation segmentation
- Brainstem localization
- Template-to-patient registration
- Anatomical feature extraction

See:

docs/PROJECT_WORKFLOW.md
