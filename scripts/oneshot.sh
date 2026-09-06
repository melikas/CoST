#!/bin/bash
# =====================================================================================
# The pre-registered 4-arm one shot, sharded across a SLURM array.
#
#   2 weightings x (10 folds x 3 repeats) = 60 encoder fits -> 120 (arm, fold) results.
#
# It is 60 and not 120 because `phase_readout` never enters the training path -- it only
# chooses what CoST._spectral emits at encode time -- so ONE encoder is read out both ways.
# That also makes the readout contrast exactly paired: same weights, two readouts, and no
# encoder-training variance between them at all.
#
# One array task = one (weights, fold) pair, ~30 min on an A100 at 6000 iters.
#
#   sbatch --array=0-59%12 scripts/oneshot.sh                # the full run
#   sbatch --array=0,30%2  scripts/oneshot.sh                # stage 0: one fold per weighting
#   WEIGHTS=paper sbatch --array=0-29%10 scripts/oneshot.sh  # one weighting only
#
# Run stage 0 FIRST and read the elapsed time off sacct before committing 60 tasks:
#   sacct -j <jobid> --format=JobID%20,State,Elapsed,MaxRSS
#
# Results: $OUT/<readout>_<weights>/<r#f#>/{repr.npz,fold.json}
# Monitor: squeue -u $USER ; tail -f logs/oneshot-<arrayjobid>_<taskid>.out
# =====================================================================================
#SBATCH --account=def-plago
#SBATCH --job-name=oneshot
#SBATCH --gres=gpu:a100_3g.20gb:1      # NARVAL. Rorqual: gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/oneshot-%A_%a.out

set -euo pipefail

PROJECT="${PROJECT:-$HOME/projects/def-plago/melikas/projects/rhythmssl_project}"
NPZ="${NPZ:-hrd_2224103.npz}"
OUT="${OUT:-results_oneshot}"
DATASET="${DATASET:-hrd}"
FOLDS="${FOLDS:-10}"
REPEATS="${REPEATS:-3}"
ITERS="${ITERS:-6000}"
MASTER_SEED="${MASTER_SEED:-20260906}"
WEIGHTS_LIST="${WEIGHTS:-paper,contracted}"
READOUTS="${READOUTS:-angle,circular}"

cd "$PROJECT"
mkdir -p logs "$OUT"

# ---- preflight -----------------------------------------------------------------------
# A missing module used to fail INSIDE a task, which then reported COMPLETED with an empty
# result. Three sweeps were lost that way (see CLUSTER.md). Fail loudly, before the GPU.
MISSING=""
for f in train.py cost.py model.py objective.py probe.py cv.py data_loader.py "$NPZ"; do
  [ -e "$f" ] || MISSING="$MISSING $f"
done
if [ -n "$MISSING" ]; then echo "ERROR: missing from $PROJECT:$MISSING"; exit 1; fi

module purge
module load StdEnv/2023 python/3.11
VENV="${VENV:-$HOME/venvs/cost}"
if [ -d "$VENV" ]; then source "$VENV/bin/activate"; fi
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# ---- resolve this task's (weights, fold) ----------------------------------------------
IFS=',' read -r -a W_LIST <<< "$WEIGHTS_LIST"
IFS=',' read -r -a R_LIST <<< "$READOUTS"
N_FOLDS=$(( FOLDS * REPEATS ))

TASK="${SLURM_ARRAY_TASK_ID:-0}"
W="${W_LIST[$(( TASK / N_FOLDS ))]}"
FOLD_IDX=$(( TASK % N_FOLDS ))
FOLD_TAG="r$(( FOLD_IDX / FOLDS ))f$(( FOLD_IDX % FOLDS ))"

# Every readout for this weighting, as train.py's --arms spec.
ARM_SPEC=""
for r in "${R_LIST[@]}"; do ARM_SPEC="${ARM_SPEC:+$ARM_SPEC,}$r:$W"; done

echo "=== task $TASK -> weights=$W fold=$FOLD_TAG arms=$ARM_SPEC ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ---- run ------------------------------------------------------------------------------
# --master-seed is IDENTICAL across every task on purpose: cv.make_folds is deterministic,
# so every task reconstructs the same 30 folds and --only-fold picks this task's one. Change
# it between tasks and they stop sharing a partition, which silently voids every paired
# comparison the analysis depends on.
srun python train.py \
  --dataset "$DATASET" \
  --npz "$NPZ" \
  --arms "$ARM_SPEC" \
  --folds "$FOLDS" \
  --repeats "$REPEATS" \
  --master-seed "$MASTER_SEED" \
  --only-fold "$FOLD_TAG" \
  --iters "$ITERS" \
  --device cuda \
  --quiet \
  --save-encoder \
  --out "$OUT" \
  "$@"

echo "=== done: $OUT/*_$W/$FOLD_TAG ==="
