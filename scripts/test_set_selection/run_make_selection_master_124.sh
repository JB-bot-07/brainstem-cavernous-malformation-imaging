#!/bin/bash
#SBATCH --job-name=make_split_master
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/jbayasi/Cavmalproject1/logs/make_split_master_%j.out
#SBATCH --error=/scratch/jbayasi/Cavmalproject1/logs/make_split_master_%j.err

set -euo pipefail

mkdir -p /scratch/jbayasi/Cavmalproject1/logs

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/jbayasi/brainiac_py39

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16

cd /scratch/jbayasi/Cavmalproject1

echo "Starting selection master feature extraction at $(date)"
echo "Python: $(which python)"
echo "Node: $(hostname)"

python /scratch/jbayasi/Cavmalproject1/scripts/make_selection_master_csv_124.py

echo "Finished selection master feature extraction at $(date)"
