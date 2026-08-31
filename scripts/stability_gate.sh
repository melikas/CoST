#!/bin/bash
# =====================================================================
# The day-resolved phase-concentration gate, over the 24 saved encoders of one run.
#
#   sbatch --array=0-23%24 scripts/stability_gate.sh results_hrd/2002135
#   SCRIPT=rhythm_dynamics.py sbatch --array=0 scripts/stability_gate.sh results_hrd/2002135
#
# SCRIPT selects which CPU gate to run (default rhythm_stability.py). A gate that needs
# no encoder needs only --array=0; one that reads every seed needs the full range.
#
# NO GPU and NO TRAINING. Every encoder already exists; this only reads them. That is the
# point of running it first: it decides whether the R_k block is worth an architecture
# change BEFORE any GPU time is spent on training one.
#
# Cost, measured: the encoder forward pass dominates at ~0.5 s/window on 12 threads, and
# the run has ~3.9k windows scored twice (DSSL and its random-init control) -- about 1 h per
# task. --cpus-per-task=12 is therefore load-bearing, not a guess.
# =====================================================================
#SBATCH --account=def-plago
#SBATCH --job-name=stab_gate
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/stab_gate-%A_%a.out
set -u
PROJECT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT" || exit 1
RUN_DIR="${1:?usage: sbatch scripts/stability_gate.sh <results_hrd/RUNID>}"

# The variant directories of this run, sorted, one per array task. Read from disk rather
# than from a hard-coded seed list: a task that picks a directory nothing wrote is exactly
# the failure mode run 2074341 hit, and there is no reason to reproduce it here.
mapfile -t DIRS < <(find "$RUN_DIR" -mindepth 1 -maxdepth 1 -type d -name '*_seed*' \
                    -exec test -f '{}/encoder.pt' \; -print | sort)
ID="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$ID" -ge "${#DIRS[@]}" ]; then
  echo "[gate] task $ID >= ${#DIRS[@]} encoders in $RUN_DIR -- nothing to do"; exit 0
fi
VARIANT_DIR="${DIRS[$ID]}"
echo "[gate] task $ID/${#DIRS[@]}  ->  $VARIANT_DIR"

module load StdEnv/2023 python/3.11
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch numpy pandas scikit-learn einops matplotlib

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
CACHE_DIR="${SLURM_TMPDIR:-/tmp}/hrd_cache"; mkdir -p "$CACHE_DIR"

PY_SCRIPT="${SCRIPT:-rhythm_stability.py}"
[ -f "$PY_SCRIPT" ] || { echo "[gate] no such script: $PY_SCRIPT"; exit 1; }
echo "[gate] running $PY_SCRIPT"
python "$PY_SCRIPT" --variant-dir "$VARIANT_DIR" --cache-dir "$CACHE_DIR"
