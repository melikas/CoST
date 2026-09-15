#!/bin/bash
#SBATCH --job-name=dssl-smoke
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Execution check on the study's GPU type (drivers, deterministic CUDA ops, CosinorPy, the
# reference stage and the whole output chain); no scientific results. Then the measured time
# per update of the full-size models, to check against the study's --time limit.
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
source slurm/env.sh
set -euo pipefail
for dataset in hrd globem; do
    for fold in 0 1; do
        srun python scripts/run_experiment.py --dataset "$dataset" --smoke --fold "$fold" --device cuda
    done
    srun python scripts/run_experiment.py --dataset "$dataset" --smoke --summarize --device cuda
done
srun python scripts/time_steps.py --device cuda
