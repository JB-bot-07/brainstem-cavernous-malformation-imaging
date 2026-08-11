#!/usr/bin/env python

from pathlib import Path
import pandas as pd

OUT = Path("/scratch/jbayasi/Cavmalproject1/results/lesion_segmentation_split_124/patient_region_labels_124.csv")

midbrain = [
    3, 5, 6, 8, 9, 10, 11, 45, 54, 55, 63, 68, 70, 73,
    83, 84, 85, 86, 95, 96, 97, 98, 99, 100, 113, 114,
    115, 124, 125
]

pons = [
    7, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
    24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
    37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50,
    51, 52, 56, 57, 59, 60, 61, 62, 69, 71, 72, 74, 75,
    76, 77, 78, 79, 80, 81, 82, 101, 102, 103, 104, 105,
    106, 107, 108, 109, 111, 117, 118, 119, 120, 121, 122
]

medulla = [
    1, 2, 4, 53, 64, 65, 66, 67, 87, 88, 89, 90, 92, 93,
    94, 110, 112, 116, 126, 127
]

excluded = [58, 91, 123]

rows = []

for n in midbrain:
    rows.append({
        "pat_id": f"Bni_Cm_Ret_{n:03d}",
        "patient_number": n,
        "region": "Midbrain"
    })

for n in pons:
    rows.append({
        "pat_id": f"Bni_Cm_Ret_{n:03d}",
        "patient_number": n,
        "region": "Pons"
    })

for n in medulla:
    rows.append({
        "pat_id": f"Bni_Cm_Ret_{n:03d}",
        "patient_number": n,
        "region": "Medulla"
    })

df = pd.DataFrame(rows).sort_values("patient_number").reset_index(drop=True)

# Checks
if len(df) != 124:
    raise ValueError(f"Expected 124 included patients, got {len(df)}")

if df["patient_number"].duplicated().any():
    dupes = df.loc[df["patient_number"].duplicated(), "patient_number"].tolist()
    raise ValueError(f"Duplicate patient numbers found: {dupes}")

present_excluded = sorted(set(df["patient_number"]) & set(excluded))
if present_excluded:
    raise ValueError(f"Excluded patients accidentally included: {present_excluded}")

expected_all = set(range(1, 128)) - set(excluded)
actual_all = set(df["patient_number"])

missing = sorted(expected_all - actual_all)
extra = sorted(actual_all - expected_all)

if missing:
    raise ValueError(f"Missing eligible patient numbers: {missing}")

if extra:
    raise ValueError(f"Unexpected patient numbers: {extra}")

OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)

print("Saved:", OUT)
print("Total included:", len(df))
print("\nRegion counts:")
print(df["region"].value_counts())
print("\nFirst rows:")
print(df.head(10).to_string(index=False))
