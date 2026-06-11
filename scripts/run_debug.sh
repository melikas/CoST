#!/bin/bash
# =====================================================================
# DEBUG runner for CoST — SHORT job to verify everything works before
# committing to the full 48h sweep.
#
# Runs OPTION D (20 pretrain iterations, TCN only). Should finish in
# well under an hour. Short --time => schedules almost immediately.
#
# Submit from the project root with:  sbatch scripts/run_debug.sh
# Watch with: squeue -u $USER  and  tail -f logs/cost_hrd_debug-<jobid>.out
# =====================================================================
#SBATCH --account=def-plago
#SBATCH --job-name=cost_hrd_debug
#SBATCH --gres=gpu:1                 # 1x A100 GPU on Narval
#SBATCH --cpus-per-task=6            # 6 CPU cores for data loading
#SBATCH --mem=64G                    # Raw CSV ~4.5 GB + model + batch
#SBATCH --time=01:00:00              # 1 hour is plenty for 20 iters; short => fast queue
#SBATCH --output=logs/%x-%j.out      # stdout+stderr: logs/cost_hrd_debug-<jobid>.out
#SBATCH --mail-user=melikaseyedi@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

set -e  # Exit on any error

# ============================================================================
# 0. PROJECT SETUP & VALIDATION
# ============================================================================
PROJECT=~/projects/def-plago/melikas/projects/rhythmssl_project
cd "$PROJECT" || { echo "Cannot cd to $PROJECT"; exit 1; }
mkdir -p logs results_hrd_debug

echo "=========================================="
echo "CoST DEBUG run (short)"
echo "=========================================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Host:         $(hostname)"
echo "Project Dir:  $PROJECT"
echo "Start Time:   $(date)"
echo "=========================================="

# ============================================================================
# 1. LOAD ALLIANCE ENVIRONMENT & CREATE VIRTUAL ENV
# ============================================================================
echo ""
echo "[env] Loading StdEnv/2023 and Python 3.11..."
module purge
module load StdEnv/2023 python/3.11

echo "[env] Creating virtualenv in $SLURM_TMPDIR/env..."
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"

echo "[env] Upgrading pip..."
pip install --no-index --upgrade pip

echo "[env] Installing packages from Alliance wheelhouse (no internet needed)..."
pip install --no-index torch numpy pandas scikit-learn einops

# ---- GPU Configuration ----
export MPLBACKEND=Agg
export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUBLAS_WORKSPACE_CONFIG=:16:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

# ============================================================================
# 2. VALIDATION & SYSTEM INFO
# ============================================================================
echo ""
echo "[validation] GPU Status (nvidia-smi):"
nvidia-smi --query-gpu=index,name,memory.total,memory.free,compute_cap --format=csv,noheader || echo "  (no NVIDIA GPU detected)"
echo ""
python --version
python -c "import torch; print('  torch:', torch.__version__); print('  CUDA available:', torch.cuda.is_available())"

echo ""
echo "[validation] Checking data file..."
if [ ! -f "datasets/HRD_RAW_MinuteLevel.csv" ]; then
  echo "ERROR: datasets/HRD_RAW_MinuteLevel.csv not found!"
  exit 1
fi
echo "  [OK] Data file found."

# ============================================================================
# 3. QUICK DEBUG TRAINING (20 iterations, TCN only)
# ============================================================================
echo ""
echo "=========================================="
echo "RUNNING DEBUG TRAINING (20 iters, TCN)"
echo "=========================================="
srun python train_hrd.py \
    --sensor-csv datasets/HRD_RAW_MinuteLevel.csv \
    --backbone tcn \
    --iters 20 \
    --batch-size 256 \
    --output-dir results_hrd_debug \
    --seed 42 \
    --gpu 0 \
    --max-threads "$SLURM_CPUS_PER_TASK"

echo ""
echo "=========================================="
echo "DEBUG RUN COMPLETE"
echo "End Time:     $(date)"
echo "=========================================="
ls -lhR "results_hrd_debug/" 2>/dev/null || echo "  (no results)"
