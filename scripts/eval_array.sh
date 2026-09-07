#!/bin/bash
# =====================================================================================
# eval.py's per-fold pass, sharded across a SLURM array.
#
# One array task = one fold, ALL FOUR ARMS. Unlike training, eval.py --only-fold does
# not split by weighting: eval_fold() loads every arm present for that fold tag and scores
# RQ1/RQ2/RQ3 for all of them in one process, since they share the split and the cohort.
# So this is 30 tasks (10 folds x 3 repeats), not 60.
#
#   sbatch --array=0 scripts/eval_array.sh                 # stage 0 -- one fold, time it
#   sbatch --array=0-29%10 scripts/eval_array.sh            # the full pass
#   python eval.py --run results_oneshot --aggregate        # THEN run this by hand
#
# RQ2 re-encodes the whole cohort once per phase level (5) plus once for the baseline, per
# arm -- 6 forward-only passes x 4 arms = 24, all on an already-trained encoder. That is
# cheap next to training itself, but has not been timed on the real cohort; run stage 0 and
# read the elapsed time before committing the array.
#
# Results: $OUT/<arm>/<r#f#>/eval.json (one written per arm, all from the one task)
# Monitor: squeue -u $USER ; tail -f logs/eval-<arrayjobid>_<taskid>.out
# =====================================================================================
#SBATCH --account=def-plago
#SBATCH --job-name=eval
#SBATCH --gres=gpu:a100_3g.20gb:1      # NARVAL. Rorqual: gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=logs/eval-%A_%a.out

set -euo pipefail

PROJECT="${PROJECT:-$HOME/projects/def-plago/melikas/projects/rhythmssl_project}"
NPZ="${NPZ:-hrd_2224103.npz}"
OUT="${OUT:-results_oneshot}"
FOLDS="${FOLDS:-10}"
REPEATS="${REPEATS:-3}"

cd "$PROJECT"
mkdir -p logs

# ---- preflight -----------------------------------------------------------------------
MISSING=""
for f in eval.py cost.py model.py objective.py probe.py cv.py data_loader.py "$NPZ" \
         "$OUT/plan.json"; do
  [ -e "$f" ] || MISSING="$MISSING $f"
done
if [ -n "$MISSING" ]; then echo "ERROR: missing from $PROJECT:$MISSING"; exit 1; fi

module purge
module load StdEnv/2023 python/3.11
VENV="${VENV:-$HOME/venvs/cost}"
if [ -d "$VENV" ]; then source "$VENV/bin/activate"; fi
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# ---- resolve this task's fold ----------------------------------------------------------
TASK="${SLURM_ARRAY_TASK_ID:-0}"
FOLD_TAG="r$(( TASK / FOLDS ))f$(( TASK % FOLDS ))"
echo "=== task $TASK -> fold=$FOLD_TAG ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ---- run ------------------------------------------------------------------------------
srun python eval.py \
  --run "$OUT" \
  --npz "$NPZ" \
  --only-fold "$FOLD_TAG" \
  --device cuda \
  "$@"

echo "=== done: eval.json written under $OUT/*/$FOLD_TAG ==="
