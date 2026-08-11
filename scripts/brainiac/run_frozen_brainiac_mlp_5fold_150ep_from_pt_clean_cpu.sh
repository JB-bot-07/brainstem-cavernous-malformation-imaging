#!/bin/bash
#SBATCH --job-name=froz_pt_mlp
#SBATCH --output=/scratch/jbayasi/Cavmalproject1/logs/froz_pt_mlp_%j.out
#SBATCH --error=/scratch/jbayasi/Cavmalproject1/logs/froz_pt_mlp_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=legion1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

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

echo "===== START CLEAN FROZEN BRAINIAC MLP FROM PT ====="
date
hostname

echo "Checking PT feature file:"
ls -lh /scratch/jbayasi/Cavmalproject1/features/brainiac_features_127.pt

echo "Checking fold CSV:"
ls -lh /scratch/jbayasi/Cavmalproject1/csvs/cavmal_master_127_folds.csv

echo "Checking script:"
ls -lh /scratch/jbayasi/Cavmalproject1/scripts/train_frozen_brainiac_mlp_5fold_150ep_from_pt_clean.py

python /scratch/jbayasi/Cavmalproject1/scripts/train_frozen_brainiac_mlp_5fold_150ep_from_pt_clean.py

echo "Checking output:"
find /scratch/jbayasi/Cavmalproject1/results/frozen_brainiac_mlp_5fold_150ep_from_pt_clean -maxdepth 3 -type f | sort

echo "===== DONE CLEAN FROZEN BRAINIAC MLP FROM PT ====="
date
