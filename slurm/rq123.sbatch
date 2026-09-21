#!/bin/bash
#SBATCH --job-name=dssl-rq123
#SBATCH --array=0-14
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# One task = one seed x fold.
#   STAGE=reference  trains the CoST reference adapter only (shared by every variant)
#   STAGE=variant    trains the variant (BACKBONE/ENCODING, default: the config's) and
#                    evaluates RQ1-RQ3 against the cached reference
# Up to 3 h a job also fits Narval's 3 h partition, which queues far faster than 12 h.
# slurm/submit.sh overrides GPU and time (GPU=..., TIME=...); a task that times out resumes
# from its last checkpoint (every 100 updates) when resubmitted.
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
source slurm/env.sh
set -euo pipefail
: "${DATASET:?Export DATASET=hrd or globem when submitting}"
RUN_NAME="${RUN_NAME:-narval_v2}"
case "$DATASET" in hrd|globem) ;; *) echo 'Invalid DATASET' >&2; exit 2;; esac
# FOLDS: folds per seed, as in the config (5; 4 for GLOBEM leave-one-year-out). Tasks = 3 x FOLDS.
folds=${FOLDS:-5}
task=${SLURM_ARRAY_TASK_ID:?}
if (( task < 0 || task >= 3 * folds )); then exit 2; fi
seed=$((task / folds + 1))
fold=$((task % folds))
extra=()
[ -n "${BACKBONE:-}" ] && extra+=(--backbone "$BACKBONE")
[ -n "${ENCODING:-}" ] && extra+=(--temporal-encoding "$ENCODING")
[ "${STAGE:-variant}" = reference ] && extra+=(--reference)
[ -n "${CONFIG:-}" ] && extra+=(--config "$CONFIG")
# SKIP_REFERENCE: reuse the cached reference instead of retraining it. The rung stays in the
# ladder; the task just refuses to train a second copy, and errors if the cache is absent. Set it
# for every variant array submitted after the reference stage has already run for this run name.
[ -n "${SKIP_REFERENCE:-}" ] && [ "${STAGE:-variant}" != reference ] && extra+=(--skip-reference)
# DEV_COHORT: architecture development inside locked fold 0's training participants.
[ -n "${DEV_COHORT:-}" ] && extra+=(--dev-cohort)
srun python scripts/run_experiment.py --dataset "$DATASET" --seed "$seed" --fold "$fold" \
    --device cuda --run-name "$RUN_NAME" "${extra[@]}"
