#!/bin/bash
#SBATCH --job-name=dssl-export
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# CPU only: time-resolved V^T / V^S summaries of one completed seed x fold (tasks/dynamics.py),
# written next to that fold's results as dynamics.npz. One task = one seed x fold.
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
source slurm/env.sh
set -euo pipefail
: "${DATASET:?Export DATASET=hrd or globem when submitting}"
folds=${FOLDS:-5}
task=${SLURM_ARRAY_TASK_ID:?}
if (( task < 0 || task >= 3 * folds )); then exit 2; fi
extra=()
[ -n "${CONFIG:-}" ] && extra+=(--config "$CONFIG")
[ -n "${OUTPUT_DIMS:-}" ] && extra+=(--output-dims "$OUTPUT_DIMS")
srun python scripts/run_experiment.py --dataset "$DATASET" --seed $((task / folds + 1)) \
    --fold $((task % folds)) --device cpu --run-name "${RUN_NAME:?}" --export-dynamics "${extra[@]}"
