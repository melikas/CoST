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
[ -n "${CONFIG:-}" ] && extra+=(--config "$CONFIG")
# SKIP_REFERENCE with a NEW run name: without this the 15 variant tasks would each try to train
# the reference and collide on its lock. With an existing run name the reference is cached, and
# submit.sh's SKIP_REFERENCE just avoids resubmitting that stage.
[ -n "${SKIP_REFERENCE:-}" ] && [ "${STAGE:-variant}" != reference ] && extra+=(--skip-reference)
# DEV_COHORT: architecture development inside locked fold 0's training participants.
[ -n "${DEV_COHORT:-}" ] && extra+=(--dev-cohort)
# REUSE: read an existing run's encoders out under this config's readout instead of training.
[ -n "${REUSE:-}" ] && extra+=(--reuse-encoders "$REUSE")
srun python scripts/run_experiment.py --dataset "$DATASET" --seed "$seed" --fold "$fold" \
    --device cuda --run-name "$RUN_NAME" "${extra[@]}"
