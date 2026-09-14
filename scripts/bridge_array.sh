#!/bin/bash
# =====================================================================================
# scripts/rq3_bridge.py -- T1 (RQ3-state) and T2 (circular readout) -- one fold per task.
#
#   sbatch --array=0 scripts/bridge_array.sh            # stage 0: one fold, read the log
#   sbatch --array=1-29%10 scripts/bridge_array.sh      # the other 29
#   python scripts/rq3_bridge.py --run results_eq_h --out results_eq_h_bridge --report
#
# Reads the run's stored representations; nothing is trained. Each task rebuilds its fold's
# untrained control and writes $OUT/<r#f#>.json, so tasks never share a file.
# =====================================================================================
#SBATCH --account=def-plago
#SBATCH --job-name=bridge
#SBATCH --gres=gpu:a100_3g.20gb:1      # NARVAL. Rorqual: gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=logs/bridge-%A_%a.out

set -euo pipefail

PROJECT="${PROJECT:-$HOME/projects/def-plago/melikas/projects/rhythmssl_project}"
RUN="${RUN:-results_eq_h}"
NPZ="${NPZ:-hrd_2224103.npz}"
ENERGY="${ENERGY:-ee_windows.npz}"
OUT="${OUT:-${RUN}_bridge}"
FOLDS="${FOLDS:-10}"

cd "$PROJECT"
mkdir -p logs

# ---- preflight -----------------------------------------------------------------------
MISSING=""
for f in scripts/rq3_bridge.py eval.py cost.py model.py objective.py probe.py cv.py \
         data_loader.py "$NPZ" "$ENERGY" "$RUN/plan.json"; do
  [ -e "$f" ] || MISSING="$MISSING $f"
done
if [ -n "$MISSING" ]; then echo "ERROR: missing from $PROJECT:$MISSING"; exit 1; fi

module purge
module load StdEnv/2023 python/3.11
VENV="${VENV:-$HOME/venvs/cost}"
if [ -d "$VENV" ]; then source "$VENV/bin/activate"; fi

# ---- this task's fold ------------------------------------------------------------------
TASK="${SLURM_ARRAY_TASK_ID:-0}"
FOLD_TAG="r$(( TASK / FOLDS ))f$(( TASK % FOLDS ))"
echo "=== task $TASK -> fold=$FOLD_TAG ==="

srun python scripts/rq3_bridge.py --run "$RUN" --npz "$NPZ" --energy "$ENERGY" \
  --out "$OUT" --device cuda --only "$FOLD_TAG"

echo "=== done: $OUT/$FOLD_TAG.json ==="
