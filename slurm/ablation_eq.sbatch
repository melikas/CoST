#!/bin/bash
#SBATCH --job-name=dssl-ablation-eq
#SBATCH --array=0-3
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# Batch B: loss/gradient-weight balance, on top of the adopted readout_norm=none. HRD seed 1,
# folds 0-1 (2 arms x 2 folds = 4 tasks). See configs/ablation/README.md for the diagnosis and
# the w_eq values chosen. The baseline (w_eq=1.0, batch A's a2_amplitude) already ran under
# --run-name a2_amplitude; reuse it rather than rerunning:
#   python scripts/ablation_report.py a2_amplitude b1_eq_peer b2_eq_small
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
source slurm/env.sh
set -euo pipefail
ARMS=(b1_eq_peer b2_eq_small)
task=${SLURM_ARRAY_TASK_ID:?}
arm=${ARMS[$((task / 2))]}
fold=$((task % 2))
srun python scripts/run_experiment.py --dataset hrd --config "configs/ablation/hrd_${arm}.json" \
    --seed 1 --fold "$fold" --device cuda --run-name "$arm" --skip-reference
