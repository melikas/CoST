#!/bin/bash
#SBATCH --job-name=dssl-summary
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# CPU only: aggregates all 15 completed tasks of one variant into tables, figures and SUMMARY.md.
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
source slurm/env.sh
set -euo pipefail
: "${DATASET:?Export DATASET=hrd or globem when submitting}"
extra=()
[ -n "${BACKBONE:-}" ] && extra+=(--backbone "$BACKBONE")
[ -n "${ENCODING:-}" ] && extra+=(--temporal-encoding "$ENCODING")
[ -n "${CONFIG:-}" ] && extra+=(--config "$CONFIG")
[ -n "${OUTPUT_DIMS:-}" ] && extra+=(--output-dims "$OUTPUT_DIMS")
# SELECT=1: first choose the width per seed x fold from the completed sweep (scripts/select_dims.py).
if [ -n "${SELECT:-}" ]; then
    srun python scripts/select_dims.py --dataset "$DATASET" --run-name "${RUN_NAME:?}"
    exit 0
fi
srun python scripts/run_experiment.py --dataset "$DATASET" --summarize \
    --run-name "${RUN_NAME:-narval_v2}" "${extra[@]}"
