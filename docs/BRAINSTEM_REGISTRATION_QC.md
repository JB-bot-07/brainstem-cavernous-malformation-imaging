# Brainstem Registration QC and Case-Specific Exceptions

## Overview

All 124 cases in the VS074-to-CavMal registration cohort underwent visual quality control.

QC included review of:

- Native patient T1+C MRI
- VS074 MRI warped into patient space
- Raw ANTs-transferred brainstem segmentation
- Lesion-added brainstem segmentation
- Final lesion-added + 2-voxel local-expansion brainstem segmentation

Most cases produced acceptable registration and brainstem segmentation results.

Four cases required case-specific handling after visual QC:

- Bni_Cm_Ret_066
- Bni_Cm_Ret_087
- Bni_Cm_Ret_089
- Bni_Cm_Ret_116

The issues and final handling decisions are documented below.

---

## Bni_Cm_Ret_087

### Issue

The cavernous malformation is located in the fourth-ventricle/rhomboid region rather than within the brainstem parenchyma.

The VS074-to-patient registration and raw transferred brainstem segmentation were acceptable.

However, the automated lesion-addition postprocessing step incorporated the lesion into the brainstem mask even though the lesion should not be included as brainstem parenchyma.

### Solution

Use the raw ANTs-transferred brainstem segmentation without lesion addition.

Use:

Bni_Cm_Ret_087_brainstem_from_VS074.nii.gz

Do not use the lesion-added or locally expanded version for the final brainstem mask.

---

## Bni_Cm_Ret_089

### Issue

This case had the same issue as Bni_Cm_Ret_087.

The lesion is located in the fourth-ventricle/rhomboid region and should not be incorporated into the brainstem parenchymal segmentation.

The underlying registration and raw transferred brainstem segmentation were acceptable.

### Solution

Use the raw ANTs-transferred brainstem segmentation without lesion addition.

Use:

Bni_Cm_Ret_089_brainstem_from_VS074.nii.gz

Do not use the lesion-added or locally expanded version for the final brainstem mask.

---

## Bni_Cm_Ret_066

### Issue

The VS074-to-patient registration failed.

Visual QC showed poor anatomical correspondence between the warped VS074 MRI and the native patient MRI. Because the registration itself was incorrect, the propagated brainstem segmentation was not considered reliable.

### Solution

Use the previously generated brainstem segmentation produced directly from the patient MRI without image registration.

The direct brainstem segmentation was visually reviewed and deemed sufficient for this case.

No registration-derived brainstem mask is used as the final mask for Bni_Cm_Ret_066.

---

## Bni_Cm_Ret_116

### Issue

The brainstem mask initially appeared inverted when overlaid on the native patient MRI.

Further review showed that the patient MRI had a different image orientation from the brainstem mask / majority orientation used in the workflow.

The transformed brainstem mask aligned correctly when reviewed in its corresponding registered image space, indicating that the apparent inversion was related to orientation consistency rather than incorrect brainstem labeling.

### Solution

Standardize the patient MRI and brainstem segmentation to the same orientation before downstream use.

A separate orientation-normalization step will be used to ensure that the unprocessed NIfTI images and brainstem masks share a consistent orientation convention.

The mask should not be manually flipped.

---

## Final Case Handling Summary

| Patient | QC issue | Final handling |
| --- | --- | --- |
| Bni_Cm_Ret_087 | Fourth-ventricle/rhomboid lesion incorrectly added to brainstem | Use raw ANTs-transferred mask without lesion addition |
| Bni_Cm_Ret_089 | Fourth-ventricle/rhomboid lesion incorrectly added to brainstem | Use raw ANTs-transferred mask without lesion addition |
| Bni_Cm_Ret_066 | Registration failure | Use previous direct brainstem segmentation generated without registration |
| Bni_Cm_Ret_116 | Patient/mask orientation mismatch | Reorient patient image and brainstem mask into a consistent orientation |

## General QC Principle

The raw ANTs-transferred brainstem segmentation is preserved for every patient so that registration quality can be evaluated independently of lesion-based postprocessing.

Case-specific corrections are documented rather than silently replacing outputs.
