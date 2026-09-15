#!/bin/bash
#SBATCH --job-name=dssl-rq123
#SBATCH --array=0-14
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# One task = one seed x fold.
#   STAGE=reference  trains the CoST reference adapter only (shared by every variant)
#   STAGE=variant    trains the variant (BACKBONE/ENCODING, default: the config's) and
#                    evaluates RQ1-RQ3 against the cached reference
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
source slurm/env.sh
set -euo pipefail
: "${DATASET:?Export DATASET=hrd or globem when submitting}"
RUN_NAME="${RUN_NAME:-narval_v1}"
case "$DATASET" in hrd|globem) ;; *) echo 'Invalid DATASET' >&2; exit 2;; esac
task=${SLURM_ARRAY_TASK_ID:?}
if (( task < 0 || task >= 15 )); then exit 2; fi
seed=$((task / 5 + 1))
fold=$((task % 5))
extra=()
[ -n "${BACKBONE:-}" ] && extra+=(--backbone "$BACKBONE")
[ -n "${ENCODING:-}" ] && extra+=(--temporal-encoding "$ENCODING")
[ "${STAGE:-variant}" = reference ] && extra+=(--reference)
srun python scripts/run_experiment.py --dataset "$DATASET" --seed "$seed" --fold "$fold" \
    --device cuda --run-name "$RUN_NAME" "${extra[@]}"
