#!/bin/bash
# =====================================================================
# SLURM runner for CoST — full positional-encoding sweep (11 variants)
#
# Trains CoST on HRD wearable data for depression-endpoint classification.
# Runs: Transformer x 8 PEs + Transformer/Time2Vec + TCN baseline + TCN/Time2Vec
#
# Submit from the project root with:  sbatch scripts/run.sh
# Monitor with: squeue -u $USER  and  tail -f logs/cost_hrd-<jobid>.out
# =====================================================================
#SBATCH --account=def-plago
#SBATCH --job-name=cost_hrd
#SBATCH --gres=gpu:1                 # 1x A100 GPU on Narval
#SBATCH --cpus-per-task=6            # 6 CPU cores for data loading
#SBATCH --mem=64G                    # Raw CSV ~4.5 GB + model + batch
#SBATCH --time=48:00:00              # Wall-clock limit; full 11-variant sweep
#SBATCH --output=logs/%x-%j.out      # stdout+stderr: logs/cost_hrd-<jobid>.out
#SBATCH --mail-user=melikaseyedi@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

set -e  # Exit on any error

# ============================================================================
# 0. PROJECT SETUP & VALIDATION
# ============================================================================
PROJECT=~/projects/def-plago/melikas/projects/rhythmssl_project
cd "$PROJECT" || { echo "Cannot cd to $PROJECT"; exit 1; }
mkdir -p logs results_hrd

echo "=========================================="
echo "CoST Time Series Forecasting Training"
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

echo "[env] Installing packages from Alliance wheelhouse..."
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
echo "[validation] Python & PyTorch:"
python --version
python << 'EOF'
import torch
print('  torch:', torch.__version__)
print('  CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('  GPU 0:', torch.cuda.get_device_name(0))
    print('  GPU memory:', torch.cuda.get_device_properties(0).total_memory // int(1e9), 'GB')
EOF

echo ""
echo "[validation] Checking data file..."
if [ ! -f "datasets/HRD_RAW_MinuteLevel.csv" ]; then
  echo "ERROR: datasets/HRD_RAW_MinuteLevel.csv not found!"
  exit 1
fi
echo "  [OK] Data file found: $(ls -lh datasets/HRD_RAW_MinuteLevel.csv | awk '{print $5, $9}')"

# ============================================================================
# 3. GPU COMPUTE CHECK
# ============================================================================
echo ""
echo "[gpu-check] Verifying GPU allocation..."
python << 'EOF'
import torch
assert torch.cuda.is_available(), "CUDA not available!"
total_mem = torch.cuda.get_device_properties(0).total_memory // int(1e9)
print(f'  [OK] {torch.cuda.get_device_name(0)} with {total_mem} GB VRAM')
x = torch.randn(1000, 1000, device='cuda')
z = torch.matmul(x, x)
print(f'  [OK] GPU computation successful')
EOF
echo ""

# ============================================================================
# 4. TRAINING — FULL POSITIONAL-ENCODING SWEEP (11 variants)
# ============================================================================
# batch-size=64 (down from 128) to fit the attention matrix (B, heads, T, T)
# within the A100's 40 GB VRAM at T=672 (168h window, 15-min bins).

echo "=========================================="
echo "RUNNING FULL PE SWEEP (11 variants)"
echo "=========================================="

SENSOR_CSV="datasets/HRD_RAW_MinuteLevel.csv"
OUTPUT_DIR="results_hrd"
SEED=42
RUN_ID="$SLURM_JOB_ID"

echo "=== Transformer backbone: 8 PEs + Time2Vec ==="
for PE in sinusoidal learnable tape rpe erpe tupe convspe tpe time2vec; do
  echo "--- transformer / $PE ---"
  python train_hrd.py \
    --sensor-csv "$SENSOR_CSV" \
    --backbone transformer \
    --pe "$PE" \
    --hidden-dims 48 --repr-dims 240 --depth 6 \
    --iters 600 --batch-size 64 \
    --output-dir "$OUTPUT_DIR" --run-id "$RUN_ID" --seed "$SEED" --gpu 0
done

echo ""
echo "=== TCN backbone: baseline + Time2Vec ==="
for PE in none time2vec; do
  echo "--- tcn / $PE ---"
  python train_hrd.py \
    --sensor-csv "$SENSOR_CSV" \
    --backbone tcn \
    --pe "$PE" \
    --hidden-dims 64 --repr-dims 320 --depth 10 \
    --iters 600 --batch-size 64 \
    --output-dir "$OUTPUT_DIR" --run-id "$RUN_ID" --seed "$SEED" --gpu 0
done

# ============================================================================
# 5. CLEANUP & SUMMARY
# ============================================================================
echo ""
echo "=========================================="
echo "TRAINING COMPLETE"
echo "=========================================="
echo "End Time:     $(date)"
echo "Results Dir:  $PROJECT/$OUTPUT_DIR/$RUN_ID/"
echo "=========================================="
echo ""
echo "Output files:"
ls -lhR "$OUTPUT_DIR/$RUN_ID/" 2>/dev/null || echo "  (no results yet)"
echo ""
echo "Compare results with:"
echo "  python scripts/collect_results.py --results-dir $OUTPUT_DIR"
