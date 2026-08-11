#!/bin/bash
#SBATCH --job-name=nnunet_postprocess
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --chdir=/scratch/jbayasi
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/jbayasi/logs/vs_pipeline_%j.out
#SBATCH --error=/scratch/jbayasi/logs/vs_pipeline_%j.err

set -euo pipefail

# Usage: pass the scratch/project base directory, e.g. /scratch/tbao
if [ "$#" -lt 1 ]; then
    echo "Usage: sbatch $0 /scratch/jbayasi"
    exit 1
fi

BASE="$1"


# Main paths
POSTPROCESS_SCRIPT="${BASE}/vs_false_positive_correction.py"
IMAGES_TS="${BASE}/nnUNet_raw_data_base/nnUNet_raw_data/Task600_Brainstem/imagesTs"
RAW_OUTPUT="${IMAGES_TS}/raw_output"
BRAINSTEM_OUTPUT="${IMAGES_TS}/brainstem"
POST_OUTPUT="${IMAGES_TS}/post_processed"


# Model and environment paths
V1_RESULTS="/scratch/tbao/ml_models/nnUNet1_brainstem_model"
V2_RESULTS="/scratch/tbao/ml_models/MC-RC+SC-GK-models"
CONDA_ENV="/scratch/tbao/conda_envs/nnunet_conda"


# Use Bao's existing nnU-Net conda environment directly.
# This avoids needing module load or conda activate inside Slurm.
export PATH="/scratch/jbayasi/local_bin:${CONDA_ENV}/bin:$PATH"
export PYTHONNOUSERSITE=1

echo "Using CONDA_ENV: $CONDA_ENV"
echo "Python: $(which python)"
echo "nnUNetv2_predict: $(which nnUNetv2_predict)"
echo "nnUNet_predict: $(which nnUNet_predict)"
python --version


# Keep cache/temp files on scratch instead of the default home directory
export HOME="$BASE"
export TMPDIR="${BASE}/tmp"
export XDG_CACHE_HOME="${BASE}/.cache"
export MPLCONFIGDIR="${XDG_CACHE_HOME}/matplotlib"
export TORCHINDUCTOR_CACHE_DIR="${XDG_CACHE_HOME}/torchinductor"
export HF_HOME="${XDG_CACHE_HOME}/huggingface"

mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TMPDIR" "$HF_HOME"


# nnU-Net v2 prediction
export nnUNet_raw="${BASE}/nnUNet_raw_data_base"
export nnUNet_preprocessed="${BASE}/nnUNet_raw_data_base"
export nnUNet_results="$V2_RESULTS"

nnUNetv2_predict \
    -d 920 \
    -f 0 1 2 3 4 \
    -tr nnUNetTrainer \
    -c 3d_fullres \
    -p nnUNetPlans \
    -i "$IMAGES_TS" \
    -o "$RAW_OUTPUT"


# nnU-Net v1 brainstem prediction
export nnUNet_raw_data_base="${BASE}/nnUNet_raw_data_base"
export nnUNet_preprocessed="${BASE}/nnUNet_raw_data_base"
export RESULTS_FOLDER="$V1_RESULTS"

nnUNet_predict \
    -i "$IMAGES_TS" \
    -o "$BRAINSTEM_OUTPUT" \
    -tr nnUNetTrainerV2 \
    -m 3d_fullres \
    -p nnUNetPlansv2.1_24GB \
    -t Task600_Brainstem \
    -f 0 1 2 3 4


# Remove likely false positives using tumor + brainstem predictions
python "$POSTPROCESS_SCRIPT" \
    --tumor "$RAW_OUTPUT" \
    --brainstem "$BRAINSTEM_OUTPUT" \
    --out "$POST_OUTPUT"


# Cleanup of scratch cache/temp contents created during this job
rm -rf "$XDG_CACHE_HOME"/*
rm -rf "$TMPDIR"/*
