#!/bin/bash
# =====================================================================
# Dump a run's windows and splits to one npz, so evaluation can leave the cluster.
#
#   sbatch scripts/dump.sh results_hrd/2166049 hrd_2166049.npz
#
# CPU ONLY. The 53.5M-row CSV is parsed once, the windows and every seed's masks are written,
# and the result is around 40 MB for HRD -- small enough to download and work on locally
# forever. Everything that does not train a network or read trained weights runs from it:
# the random-init audit, marker recovery, probe and readout comparisons, permutation controls.
#
# The heavy part is the CSV parse, so mem matters and cpus do not much.
# =====================================================================
#SBATCH --account=def-plago
#SBATCH --job-name=dump_ctx
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=logs/dump_ctx-%j.out
set -u
PROJECT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT" || exit 1
RUN_DIR="${1:?usage: sbatch scripts/dump.sh <results_x/RUNID> [out.npz]}"
OUT="${2:-$(basename "$RUN_DIR").npz}"

module load StdEnv/2023 python/3.11
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch numpy pandas scikit-learn einops

export PYTHONUNBUFFERED=1
CACHE_DIR="${SLURM_TMPDIR:-/tmp}/dump_cache"; mkdir -p "$CACHE_DIR"

python analysis/dump_context.py --run-dir "$RUN_DIR" --out "$OUT" --cache-dir "$CACHE_DIR"
ls -lh "$OUT"
