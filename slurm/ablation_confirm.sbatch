#!/bin/bash
#SBATCH --job-name=dssl-ablation-confirm
#SBATCH --array=0-9
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# Replication of the ONE candidate from batch B, against its baseline, on all 5 HRD folds of
# seed 1 (2 arms x 5 folds = 10 tasks). Batch B ran only folds 0-1, where this project has
# already been burned once: a0_current's Steps acrophase own-R2 read 0.765 on folds 0-1 but
# 0.38 over the full 15-fold matrix. Nothing here is a new mechanism -- it is the same two
# configs at 5 folds instead of 2.
#   a2_amplitude   w_eq=1.0   baseline (its folds 0-1 were deleted by an rm -rf; regenerated here)
#   b2_eq_small    w_eq=0.05  candidate (folds 0-1 already exist and will be skipped as complete)
# Score with: python scripts/ablation_report.py a2_amplitude b2_eq_small
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
source slurm/env.sh
set -euo pipefail
ARMS=(a2_amplitude b2_eq_small)
task=${SLURM_ARRAY_TASK_ID:?}
arm=${ARMS[$((task / 5))]}
fold=$((task % 5))
srun python scripts/run_experiment.py --dataset hrd --config "configs/ablation/hrd_${arm}.json" \
    --seed 1 --fold "$fold" --device cuda --run-name "$arm" --skip-reference
