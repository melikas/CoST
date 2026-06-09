#!/bin/bash
# =====================================================================
# SLURM runner for CoST: Contrastive Learning of Disentangled 
# Seasonal-Trend Representations for Time Series Forecasting
# 
# Paper: https://arxiv.org/abs/2202.01575 (ICLR 2022)
# GitHub: https://github.com/salesforce/CoST
#
# Trains CoST on HRD wearable data for depression-endpoint classification
# (via train_hrd.py with time-domain and frequency-domain contrastive losses).
#
# Submit with:  sbatch run.sh
# Monitor with: squeue -u $USER  and  tail -f logs/cost_hrd-<jobid>.out
# =====================================================================
#SBATCH --account=def-plago
#SBATCH --job-name=cost_hrd
#SBATCH --gres=gpu:1                 # 1x A100 GPU on Narval
#SBATCH --cpus-per-task=6            # 6 CPU cores for data loading
#SBATCH --mem=64G                    # Raw CSV ~4.5 GB + model + batch
#SBATCH --time=24:00:00              # Wall-clock limit (job stops when done if earlier)
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

echo "[env] Installing packages from Alliance wheelhouse (no internet needed)..."
# Core dependencies: PyTorch, NumPy, Pandas, scikit-learn, einops (for tensor operations)
# These match the requirements.txt / train_hrd.py imports
pip install --no-index torch numpy pandas scikit-learn einops

# ---- GPU Configuration ----
export MPLBACKEND=Agg
export CUDA_VISIBLE_DEVICES=0           # Use GPU 0 (explicit)
export CUDA_DEVICE_ORDER=PCI_BUS_ID     # Order GPUs by PCIe bus (stable across runs)
export CUBLAS_WORKSPACE_CONFIG=:16:8    # For deterministic CUDA operations (optional)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # Reduce fragmentation -> fewer OOMs
# export CUDA_LAUNCH_BLOCKING=1         # Synchronous GPU ops (debug only; ~slows training, leave off)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK  # OpenMP threads = CPU cores
export PYTHONUNBUFFERED=1               # Flush Python output immediately to see logs in tail -f

# ============================================================================
# 2. VALIDATION & SYSTEM INFO
# ============================================================================
echo ""
echo "[validation] System Information:"
echo "  GPU Status (nvidia-smi):"
nvidia-smi --query-gpu=index,name,memory.total,memory.free,compute_cap --format=csv,noheader || echo "  (no NVIDIA GPU detected or nvidia-smi not available)"
echo ""
echo "[validation] Python & PyTorch:"
python --version
python << 'EOF'
import torch
print('  torch:', torch.__version__)
print('  CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('  CUDA device count:', torch.cuda.device_count())
    print('  Current CUDA device:', torch.cuda.current_device())
    print('  GPU 0 name:', torch.cuda.get_device_name(0))
    print('  GPU 0 capability:', torch.cuda.get_device_capability(0))
    print('  cudnn version:', torch.backends.cudnn.version())
    print('  cudnn enabled:', torch.backends.cudnn.enabled)
else:
    print('  WARNING: CUDA is not available - CPU-only mode!')
EOF

echo ""
echo "[validation] Checking data file..."
if [ ! -f "datasets/HRD_RAW_MinuteLevel.csv" ]; then
  echo "ERROR: datasets/HRD_RAW_MinuteLevel.csv not found!"
  exit 1
fi
echo "  [OK] Data file found: $(ls -lh datasets/HRD_RAW_MinuteLevel.csv | awk '{print $5, $9}')"

# ============================================================================
# 3. PRE-TRAINING GPU CHECK
# ============================================================================
echo ""
echo "[gpu-check] Verifying GPU allocation for training..."
if python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'" 2>/dev/null; then
  echo "  [OK] CUDA available - GPU training enabled"
  python << 'EOF'
import torch
total_mem = torch.cuda.get_device_properties(0).total_memory // int(1e9)
print(f'  [OK] GPU 0: {torch.cuda.get_device_name(0)} with {total_mem} GB')
EOF
  
  echo "  Testing GPU computation..."
  python << 'EOF'
import torch
x = torch.randn(1000, 1000, device='cuda')
y = torch.randn(1000, 1000, device='cuda')
z = torch.matmul(x, y)
print(f'  [OK] GPU computation successful - tensor on device: {z.device}')
EOF
  if [ $? -eq 0 ]; then
    echo "  [OK] GPU is ready for training!"
  else
    echo "  [FAIL] GPU computation failed!"
  fi
else
  echo "  [FAIL] CUDA not available!"
  echo "  ERROR: This script requires GPU. CPU-only training is NOT supported."
  exit 1
fi
echo ""

# ============================================================================
# 4. TRAINING EXECUTION
# ============================================================================
# CoST Paper (ICLR 2022) Implementation Details (Section 4.1):
#   - Batch size: 256 (paper standard for all datasets)
#   - Iterations: 600 (for datasets > 100K samples); 200 (for < 100K samples)
#   - Optimizer: SGD with momentum=0.9, weight_decay=1E-4, lr=1E-3
#   - Learning rate schedule: cosine annealing
#   - Backbone: TCN (paper default) or Transformer for comparison
#   - Trend Feature Disentangler (TFD): mixture of auto-regressive experts (time domain loss)
#   - Seasonal Feature Disentangler (SFD): learnable Fourier layer (frequency domain loss)
#   - Combined contrastive loss: L = L_time + (α/2)(L_amp + L_phase), α=5E-04
#   - Data augmentations: scale, shift, jitter (each prob=0.5)
#   - Ridge regression for downstream forecasting (validation set tunes regularization)

echo ""
echo "=========================================="
echo "RUNNING TRAINING PIPELINE"
echo "=========================================="

# ---- OPTION A: BOTH TCN & TRANSFORMER (RECOMMENDED) ----
# Runs both backbones sequentially per paper ablation (Table 4)
# HRD dataset > 100K samples → use 600 iterations (not epochs)
echo "[train] Running BOTH backbones (TCN + Transformer)..."
echo "[train] Training will use: GPU 0 (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
for BACKBONE in transformer tcn; do
  echo ""
  echo "=========================================="
  echo "Training: $BACKBONE backbone (GPU accelerated)"
  echo "=========================================="

  # Reduce model dims for Transformer to fit in GPU memory
  if [ "$BACKBONE" = "transformer" ]; then
    HIDDEN_DIMS=48
    REPR_DIMS=240
    DEPTH=6
  else
    HIDDEN_DIMS=64
    REPR_DIMS=320
    DEPTH=10
  fi

  srun python train_hrd.py \
    --sensor-csv datasets/HRD_RAW_MinuteLevel.csv \
    --backbone "$BACKBONE" \
    --iters 600 \
    --batch-size 128 \
    --hidden-dims $HIDDEN_DIMS \
    --repr-dims $REPR_DIMS \
    --depth $DEPTH \
    --output-dir results_hrd \
    --seed 42 \
    --gpu 0 \
    --max-threads "$SLURM_CPUS_PER_TASK"
  
  TRAIN_EXIT_CODE=$?
  if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "[OK] $BACKBONE training completed successfully (exit code: $TRAIN_EXIT_CODE)"
  else
    echo "[FAIL] $BACKBONE training FAILED (exit code: $TRAIN_EXIT_CODE)"
    exit 1
  fi
done

# ============================================================================
# 5. CLEANUP & SUMMARY
# ============================================================================
echo ""
echo "=========================================="
echo "TRAINING COMPLETE"
echo "=========================================="
echo "End Time:     $(date)"
echo "Results Dir:  $PROJECT/results_hrd/"
echo "GPU Used:     YES (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "=========================================="
echo ""
echo "Output files:"
ls -lh results_hrd/ 2>/dev/null || echo "  (no results yet)"
echo ""

# ============================================================================
# ALTERNATIVE RUN OPTIONS (uncomment ONE below to replace the OPTION A above)
# ============================================================================

# ---- OPTION B: TCN ONLY (Paper Default) ----
# Uncomment this block and comment out OPTION A above to run TCN only
# This is the backbone used in the CoST paper's main results
# echo "[train] Running TCN backbone (paper default)..."
# echo "[train] Training will use: GPU 0 (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
# echo "=========================================="
# echo "Training: TCN backbone (GPU accelerated)"
# echo "=========================================="
# srun python train_hrd.py \
#     --sensor-csv datasets/HRD_RAW_MinuteLevel.csv \
#     --backbone tcn \
#     --iters 600 \
#     --batch-size 256 \
#     --output-dir results_hrd \
#     --seed 42 \
#     --gpu 0 \
#     --max-threads "$SLURM_CPUS_PER_TASK"

# ---- OPTION C: TRANSFORMER ONLY ----
# Uncomment this block and comment out OPTION A above to run Transformer only
# echo "[train] Running Transformer backbone..."
# echo "[train] Training will use: GPU 0 (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
# echo "=========================================="
# echo "Training: Transformer backbone (GPU accelerated)"
# echo "=========================================="
# srun python train_hrd.py \
#     --sensor-csv datasets/HRD_RAW_MinuteLevel.csv \
#     --backbone transformer \
#     --iters 600 \
#     --batch-size 256 \
#     --output-dir results_hrd \
#     --seed 42 \
#     --gpu 0 \
#     --max-threads "$SLURM_CPUS_PER_TASK"

# ---- OPTION D: QUICK DEBUG RUN (FEW ITERATIONS) ----
# Uncomment this block and comment out OPTION A above for quick testing
# echo "[train] Quick debug run (20 pretrain iterations)..."
# echo "[train] Training will use: GPU 0 (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
# echo "=========================================="
# echo "Training: TCN backbone - DEBUG MODE (GPU accelerated)"
# echo "=========================================="
# srun python train_hrd.py \
#     --sensor-csv datasets/HRD_RAW_MinuteLevel.csv \
#     --backbone tcn \
#     --iters 20 \
#     --batch-size 256 \
#     --output-dir results_hrd_debug \
#     --seed 42 \
#     --gpu 0 \
#     --max-threads "$SLURM_CPUS_PER_TASK"

# ============================================================================
# KEY PARAMETERS REFERENCE (from paper Section 4.1):
# ============================================================================
# --sensor-csv PATH        Input time series CSV (HRD wearable data)
# --backbone {tcn|transformer}  Feature encoder backbone (tcn=paper default)
# --iters N                Pretrain iterations (600 for >100K samples, 200 for <100K)
# --batch-size N           Batch size for SGD (256 = paper standard, NOT 64)
# --output-dir DIR         Directory for metrics & loss plots (default: results_hrd)
# --seed N                 Random seed for reproducibility (default: 42)
# --gpu {0|1|-1}           GPU device: 0/1 (specific GPU) or -1 (CPU, slow)
# --max-threads N          CPU threads for data loading (SLURM_CPUS_PER_TASK)
# --window-hours H         Lookback window in hours (default: 168 = 1 week)
# --bin-minutes M          Temporal resolution in minutes (default: 1)
#
# PAPER HYPERPARAMETERS (implicit in train_hrd.py):
#   - learning rate: 1E-3 (SGD)
#   - momentum: 0.9
#   - weight decay: 1E-4
#   - scheduler: cosine annealing
#   - MoCo queue size: 256
#   - MoCo momentum: 0.999
#   - MoCo temperature: 0.07
#   - seasonal weight α: 5E-04
#
# OUTPUT:
#   metrics_<backbone>_seed42.json       AUC/F1/Accuracy at window & participant level
#   pretrain_loss_<backbone>_seed42.npy  Training loss curve (pretraining phase)
# ============================================================================
