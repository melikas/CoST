#!/bin/bash
#SBATCH --job-name=dssl-smoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Execution check on the GPU (drivers, deterministic CUDA ops, CosinorPy); no scientific results.
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
source slurm/env.sh
set -euo pipefail
for dataset in hrd globem; do
    for fold in 0 1; do
        srun python scripts/run_experiment.py --dataset "$dataset" --smoke --fold "$fold" --device cuda
    done
    srun python scripts/run_experiment.py --dataset "$dataset" --smoke --summarize --device cuda
done
