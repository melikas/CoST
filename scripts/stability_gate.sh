#!/bin/bash
# =====================================================================
# The day-resolved phase-concentration gate, over the 24 saved encoders of one run.
#
#   sbatch --array=0-23%24 scripts/stability_gate.sh results_hrd/2002135
#   SCRIPT=analysis/readout_interaction.py sbatch --array=0 scripts/stability_gate.sh results_hrd/2002135
#   SCRIPT=experiment_q3.py EXTRA='--no-supervised --no-plain-ssl' \
#     sbatch --gres=gpu:a100_3g.20gb:1 --time=3:00:00 --array=0-23%12 \
#     scripts/stability_gate.sh results_hrd/2002135
#
# SCRIPT selects which CPU gate to run (default analysis/rhythm_stability.py). A gate that needs
# no encoder needs only --array=0; one that reads every seed needs the full range.
# EXTRA passes flags through to it. A gate needing a GPU asks for one on the sbatch
# command line -- SLURM reads #SBATCH before this script runs, so it cannot be set here.
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
# NEED_ENC=0 for a gate that reads no trained weights. run.sh keeps only one encoder per
# sweep unless KEEP_ENC_ALL=1, so filtering on encoder.pt silently reduced a 24-task array
# to one -- which is how job 2188351 came back with a single seed.
if [ "${NEED_ENC:-1}" = "1" ]; then
  mapfile -t DIRS < <(find "$RUN_DIR" -mindepth 1 -maxdepth 1 -type d -name '*_seed*' \
                      -exec test -f '{}/encoder.pt' \; -print | sort)
else
  mapfile -t DIRS < <(find "$RUN_DIR" -mindepth 1 -maxdepth 1 -type d -name '*_seed*' \
                      -exec test -f '{}/metrics.json' \; -print | sort)
fi
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

# CosinorPy only when the gate actually needs it -- the RQ2/RQ3 scripts carry the cosinor
# baseline, which is the arm this whole design is being measured against, and losing it
# silently would make the comparison meaningless. It is not in the Alliance wheelhouse and
# compute nodes have no internet, hence the local wheelhouse first. Mirrors scripts/run.sh.
case "${SCRIPT:-}" in
  experiment_q*.py)
    pip install --no-index seaborn matplotlib pandas \
      || echo "  [WARN] no seaborn -> CosinorPy import will fail"
    pip install --no-index --find-links "$PROJECT/wheels" CosinorPy 2>/dev/null \
      || pip install CosinorPy \
      || echo "  [WARN] CosinorPy install failed"
    python -c "from CosinorPy import cosinor" 2>/dev/null \
      && echo "[gate] CosinorPy OK" \
      || echo "[gate] [WARN] CosinorPy MISSING -- the cosinor rung will be skipped, which is "\
              "the comparison this run exists to make"
    ;;
esac

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
CACHE_DIR="${SLURM_TMPDIR:-/tmp}/hrd_cache"; mkdir -p "$CACHE_DIR"

PY_SCRIPT="${SCRIPT:-analysis/rhythm_stability.py}"
[ -f "$PY_SCRIPT" ] || { echo "[gate] no such script: $PY_SCRIPT"; exit 1; }
echo "[gate] running $PY_SCRIPT"
# shellcheck disable=SC2086  -- EXTRA is a caller-supplied argument list, word splitting wanted
python "$PY_SCRIPT" --variant-dir "$VARIANT_DIR" --cache-dir "$CACHE_DIR" ${EXTRA:-}
