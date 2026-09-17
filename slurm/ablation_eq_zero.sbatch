#!/bin/bash
#SBATCH --job-name=dssl-ablation-eqzero
#SBATCH --array=0-4
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# The endpoint of the w_eq dose-response: w_eq=0 (equivariance off entirely; head_eq is then not
# built at all). HRD seed 1, all 5 folds, to match a2_amplitude (w_eq=1.0) and b2_eq_small
# (w_eq=0.05) -- 5 folds is the minimum this project trusts after a 2-fold result overstated
# both the RQ1 gain (-0.0014 vs -0.0136 at 5 folds) and the Steps/screen phase story.
# Score all three together:
#   python scripts/ablation_report.py a2_amplitude b2_eq_small b3_eq_zero
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
source slurm/env.sh
set -euo pipefail
srun python scripts/run_experiment.py --dataset hrd \
    --config configs/ablation/hrd_b3_eq_zero.json \
    --seed 1 --fold "${SLURM_ARRAY_TASK_ID}" --device cuda --run-name b3_eq_zero --skip-reference
