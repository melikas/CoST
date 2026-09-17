#!/bin/bash
#SBATCH --job-name=dssl-ablation-trend
#SBATCH --array=0-4
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# Batch C, Failure A: trend_views=disjoint_days on top of the working default (w_eq=0,
# readout_norm=none). HRD seed 1, all 5 folds, against b3_eq_zero (identical otherwise).
# --time is 4 h, not 3: the extra query pass measured +23% per step.
# Score:  python scripts/ablation_report.py b3_eq_zero c1_trend_days
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
source slurm/env.sh
set -euo pipefail
srun python scripts/run_experiment.py --dataset hrd \
    --config configs/ablation/hrd_c1_trend_days.json \
    --seed 1 --fold "${SLURM_ARRAY_TASK_ID}" --device cuda --run-name c1_trend_days --skip-reference
