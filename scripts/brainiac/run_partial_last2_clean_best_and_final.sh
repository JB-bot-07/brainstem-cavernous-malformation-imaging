#!/bin/bash
#SBATCH --job-name=ptft2clean
#SBATCH --output=/scratch/jbayasi/Cavmalproject1/logs/ptft2clean_%j.out
#SBATCH --error=/scratch/jbayasi/Cavmalproject1/logs/ptft2clean_%j.err
#SBATCH --time=60:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -e

module load anaconda3 || true
source $(conda info --base)/etc/profile.d/conda.sh

export CONDARC=/scratch/jbayasi/.condarc
export CONDA_PKGS_DIRS=/scratch/jbayasi/conda_pkgs
export CONDA_ENVS_PATH=/scratch/jbayasi/conda_envs
export CONDA_BLD_PATH=/scratch/jbayasi/conda_bld
export MPLCONFIGDIR=/scratch/jbayasi/matplotlib_cache
mkdir -p /scratch/jbayasi/matplotlib_cache

conda activate /scratch/jbayasi/brainiac_py39

cd /scratch/jbayasi/Cavmalproject1

echo "===== START CLEAN PARTIAL FT LAST 2 BLOCKS ====="
date
hostname
nvidia-smi

OUTPUT_BASE="/scratch/jbayasi/Cavmalproject1/results/partial_finetune_clean_last2blocks_150ep_best_and_final"

for FOLD in 0 1 2 3 4
do
    echo ""
    echo "============================================================"
    echo "RUNNING LAST 2 BLOCKS | FOLD ${FOLD}"
    echo "============================================================"
    date

    python /scratch/jbayasi/Cavmalproject1/scripts/train_partial_finetune_clean_best_and_final.py \
        --fold ${FOLD} \
        --unfreeze_last_n_blocks 2 \
        --backbone_lr 1e-5 \
        --output_base ${OUTPUT_BASE}

    echo "DONE FOLD ${FOLD}"
    date
done

echo "===== DONE CLEAN PARTIAL FT LAST 2 BLOCKS ====="
date
