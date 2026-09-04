#!/bin/bash
# Run one python script on a CPU node, in the SAME environment the gate builds.
#
# There is no persistent virtualenv on this cluster: every job builds one in $SLURM_TMPDIR
# from the local wheelhouse, because compute nodes have no internet. A one-off
#   sbatch --wrap="source ~/venv/bin/activate && python ..."
# therefore fails in three seconds with no output anyone reads, which is what happened to
# job 2394399. This exists so a script that is not the stability gate can still get that
# environment without a second copy of it drifting away from the first.
#
#   sbatch scripts/cpu_job.sh kfold_eval.py --npz hrd_2224103.npz
#   sbatch --mem=96G --time=4:00:00 scripts/cpu_job.sh some_other.py --flag
# =====================================================================
#SBATCH --account=def-plago
#SBATCH --job-name=cpu_job
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=logs/cpu_job-%j.out
set -u
PROJECT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT" || exit 1
mkdir -p logs
PY_SCRIPT="${1:?usage: sbatch scripts/cpu_job.sh <script.py> [args...]}"
shift
[ -f "$PY_SCRIPT" ] || { echo "[cpu_job] no such script: $PY_SCRIPT"; exit 1; }

module load StdEnv/2023 python/3.11
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch numpy pandas scikit-learn einops matplotlib

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "[cpu_job] $PY_SCRIPT $*"
python "$PY_SCRIPT" "$@"
