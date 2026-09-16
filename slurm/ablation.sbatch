#!/bin/bash
#SBATCH --job-name=dssl-ablation
#SBATCH --array=0-9
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# The minimal-repair ablation ladder on HRD, seed 1, folds 0-1 (5 arms x 2 folds = 10 tasks).
# Each arm differs from configs/hrd.json by ONE mechanism (configs/ablation/hrd_*.json):
#   a0_current    the narval_v1 model, unchanged (control)
#   a1_trend      mask_mode=binomial, moco_k=256   (upstream CoST defaults; Failure A)
#   a2_amplitude  readout_norm=none                (Failure B)
#   a3_phase      phase_mode=circular              (Failure C)
#   a4_combined   all three
# --skip-reference: the ladder compares DSSL against its own untrained control, so a second
# full training per task (the CoST reference) would only spend GPU hours. The scientific
# matrix keeps that rung; these runs are for the mechanistic criteria only.
# Results: results/hrd/<arm>/tcn_none/seed_1/fold_<f>/ ; compare with scripts/ablation_report.py
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
source slurm/env.sh
set -euo pipefail
ARMS=(a0_current a1_trend a2_amplitude a3_phase a4_combined)
task=${SLURM_ARRAY_TASK_ID:?}
arm=${ARMS[$((task / 2))]}
fold=$((task % 2))
srun python scripts/run_experiment.py --dataset hrd --config "configs/ablation/hrd_${arm}.json" \
    --seed 1 --fold "$fold" --device cuda --run-name "$arm" --skip-reference
