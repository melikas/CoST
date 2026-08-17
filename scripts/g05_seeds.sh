#!/bin/bash
# =====================================================================
# GATE G0.5 at n=6 -- E1.3: CoST vs RANDOM-INIT across ALL seeds.
#
# The seed-42 gate (job 19981667) returned FAIL on one random draw. This pairs the
# two arms over every seed. No retraining, no encoder.pt required: the random-init
# arm needs only (config, data, seed), and the trained values are already stored in
# each rq1/rq1.json. Nothing is overwritten -- output goes to g05_seeds.json.
#
#   sbatch scripts/g05_seeds.sh
# =====================================================================
#SBATCH --account=def-plago
#SBATCH --job-name=g05_seeds
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-user=melikaseyedi@gmail.com
#SBATCH --mail-type=END,FAIL

set -e

PROJECT=~/projects/def-plago/melikas/projects/rhythmssl_project
cd "$PROJECT" || { echo "Cannot cd to $PROJECT"; exit 1; }
mkdir -p logs

RUN=results_hrd/19937323

[ -f scripts/g05_seeds.py ] || { echo "FATAL: scripts/g05_seeds.py not uploaded"; exit 1; }
N=$(ls -d $RUN/*/ 2>/dev/null | wc -l)
[ "$N" -ge 12 ] || { echo "FATAL: expected 12 variant dirs under $RUN, found $N"; exit 1; }
echo "[preflight] script present, $N variant dirs"

module purge
module load StdEnv/2023 python/3.11
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch numpy pandas scipy scikit-learn einops matplotlib umap-learn
pip install --no-index seaborn || echo "  [WARN] no seaborn"
pip install CosinorPy || echo "  [WARN] CosinorPy install failed"
python -c "from CosinorPy import cosinor" 2>/dev/null \
  && echo "[env] CosinorPy OK" \
  || { echo "FATAL: CosinorPy unimportable -- E1.3 cannot be evaluated."; exit 1; }

export MPLBACKEND=Agg
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

CACHE="$SLURM_TMPDIR/hrd_cache"; mkdir -p "$CACHE"

python scripts/g05_seeds.py --run-dir "$RUN" --cache-dir "$CACHE" --gpu 0

echo "=== G0.5 n=6 done $(date) ==="
