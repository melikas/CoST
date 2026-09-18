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
srun python scripts/run_experiment.py --dataset "$DATASET" --summarize \
    --run-name "${RUN_NAME:-narval_v1}" "${extra[@]}"
