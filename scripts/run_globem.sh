#!/bin/bash
# =====================================================================
# SLURM runner for CoST — configurable sweep
# GPU type is cluster-native for fast queueing: Rorqual = h100, Narval = a100.
# (Request a TYPED GPU, not a bare "=1" -- the untyped form queues for hours on Rorqual.)
#
# Trains CoST on HRD wearable data for depression-endpoint classification.
# Runs are controlled by the editable SEEDS / VARIANTS block below.
# Default mode is one single job that iterates over every configured seed and variant.
#
# Submit from the project root with:  sbatch scripts/run.sh
# Monitor with: squeue -u $USER  and  tail -f logs/cost_hrd-<arrayjobid>_<taskid>.out
# Results land under one folder: results_hrd/<arrayjobid>/{backbone}_{pe}_seed{SEED}/
# =====================================================================
#SBATCH --account=def-plago
#SBATCH --job-name=cost_hrd
#SBATCH --gpus-per-node=a100:1       # 1 GPU, cluster-native type. Narval: a100 (this is Narval). On Rorqual change this one word to h100:1.
#SBATCH --cpus-per-task=8            # 8 CPU cores for data loading (fits both Narval and Rorqual GPU nodes)
#SBATCH --mem=64G                    # Raw CSV ~4.5 GB + model + batch (well under both clusters' node max)
#SBATCH --time=24:00:00              # Wall-clock for the configured seed/variant sweep below
#SBATCH --array=0-19                 # one array task per (SEED, HOLDOUT) pair. MUST equal #SEEDS x #HOLDOUTS - 1: 5 seeds x 4 folds -> 0-19; with an empty HOLDOUTS it is #SEEDS - 1.
#SBATCH --output=logs/%x-%A_%a.out   # stdout+stderr per task: logs/cost_hrd-<arrayjobid>_<taskid>.out
#SBATCH --mail-user=melikaseyedi@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

set -e  # Exit on any error

# ============================================================================
# 0. PROJECT SETUP & VALIDATION
# ============================================================================
# Run IN the folder the job was submitted from (SLURM_SUBMIT_DIR), so results/logs land next
# to the code you actually uploaded -- no hardcoded folder name. Falls back to $PWD for a
# manual (non-sbatch) run. This is what lets rhythmssl_project_GLOBEM work without edits.
PROJECT="${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p "$PROJECT"
cd "$PROJECT" || { echo "Cannot cd to $PROJECT"; exit 1; }
mkdir -p logs results_hrd

echo "=========================================="
echo "CoST Time Series Forecasting Training"
echo "=========================================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Array Job/Task: ${SLURM_ARRAY_JOB_ID:-n/a} / ${SLURM_ARRAY_TASK_ID:-n/a}"
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
# setuptools/wheel are what --no-build-isolation reuses when CosinorPy is cached as a source
# archive rather than a wheel (see the CosinorPy block below); without them in the venv that
# install path cannot build offline.
pip install --no-index --upgrade pip setuptools wheel

echo "[env] Installing packages from Alliance wheelhouse..."
pip install --no-index torch numpy pandas scikit-learn einops matplotlib umap-learn

# seaborn is a HARD dependency of CosinorPy (it does `import seaborn` at IMPORT time), so
# without it CosinorPy fails to import and the paper-cosinor view is silently skipped -- this
# was exactly why "Cosinor (paper)" stayed missing even AFTER CosinorPy itself installed.
# Non-fatal (kept off the core line so a missing wheel can't abort the whole job): try the
# wheelhouse, then the wheel cache.
echo "[env] Installing seaborn (CosinorPy import dependency)..."
pip install --no-index seaborn \
  || pip install --no-index --find-links="$PROJECT/wheels" seaborn \
  || echo "  [WARN] seaborn not installed -- CosinorPy import will fail, paper-cosinor view skipped"

# statsmodels is CosinorPy's other import-time dependency and is not on the core line above.
pip install --no-index statsmodels scipy \
  || echo "  [WARN] statsmodels/scipy missing -- CosinorPy import will fail"

# CosinorPy (the paper's exact cosinor engine) is the ONE package not in the Alliance
# --no-index wheelhouse, and compute nodes have no internet, so it must come from a wheel
# cached once on the LOGIN node.
#
# ALWAYS --no-deps, on BOTH download and install. CosinorPy declares pandas/numpy/matplotlib/
# statsmodels/scipy/openpyxl/scikit-optimize, and letting pip resolve that tree makes it fetch
# scipy FROM PYPI and build it from source, which dies on the login node with:
#     ../scipy/meson.build:285:9: ERROR: Dependency "OpenBLAS" not found
# That build is pure waste here: every one of those deps is already installed above from the
# wheelhouse. CosinorPy itself is a pure-python wheel (py3-none-any), so --no-deps fetches one
# small file and compiles nothing. Its real import-time needs are matplotlib, numpy, pandas,
# scipy, seaborn and statsmodels -- all present by this point; openpyxl and scikit-optimize are
# NOT imported by `from CosinorPy import cosinor` and are safely skipped.
#
# Run ONCE on the LOGIN node, with the SAME modules as this script so the wheel matches:
#     module purge && module load StdEnv/2023 python/3.11
#     pip download --no-deps CosinorPy -d <project>/wheels
#
# Non-fatal: if it fails, train_hrd still runs and only the paper-cosinor baseline is skipped.
WHEEL_CACHE="$PROJECT/wheels"
mkdir -p "$WHEEL_CACHE"
# Accept a source archive too, not just .whl: `pip download` on the login node can hand back
# CosinorPy-3.1.tar.gz instead of the py3-none-any wheel (Alliance's pip config prefers sdists
# for packages outside their wheelhouse). The old check globbed only *.whl, so a perfectly good
# cached sdist looked like an empty cache and the job fell through to the indexed install --
# which then failed for having no internet. CosinorPy is pure python, so building the sdist
# needs no compiler; it only needs --no-build-isolation, otherwise pip tries to fetch its build
# backend (setuptools/wheel) from PyPI and dies offline for that reason instead.
# (matched by listing the directory, not by two globs: `ls a*.whl b*.tar.gz` exits non-zero as
#  soon as the FIRST pattern matches nothing, so a cache holding only the .tar.gz still read as
#  empty.)
# Per-task pip cache and build dir. THIS is what made the install flaky: on run 66517709 the
# array tasks for seeds 369 and 827 failed with ModuleNotFoundError while 156/267/454 succeeded
# from the SAME wheel cache -- 26 of 65 variants lost the cosinor row. All 5 tasks start within
# seconds of each other and, by default, share ~/.cache/pip and the same sdist unpack directory,
# so concurrent builds of the same source archive race each other. Giving every task its own
# scratch cache removes the shared state entirely.
export PIP_CACHE_DIR="$SLURM_TMPDIR/pipcache"
export TMPDIR="$SLURM_TMPDIR"
mkdir -p "$PIP_CACHE_DIR"

# Try, verify, retry. `pip install` can report success while the import still fails (a partially
# built sdist), and the reverse -- so the ONLY trustworthy check is importing it. Two attempts,
# then a direct install of the archive by path, which skips the resolver completely.
install_cosinorpy() {
  local src
  if ls "$WHEEL_CACHE" 2>/dev/null | grep -qiE '^cosinorpy.*\.(whl|tar\.gz)$'; then
    for attempt in 1 2; do
      echo "[env] Installing CosinorPy from local cache (attempt $attempt/2)..."
      pip install --no-index --no-deps --no-build-isolation \
                  --find-links="$WHEEL_CACHE" CosinorPy >/dev/null 2>&1
      python -c "import CosinorPy" 2>/dev/null && return 0
    done
    # last resort: name the file itself, no --find-links resolution at all
    src=$(ls "$WHEEL_CACHE"/[Cc]osinor[Pp]y*.whl "$WHEEL_CACHE"/[Cc]osinor[Pp]y*.tar.gz 2>/dev/null | head -1)
    if [ -n "$src" ]; then
      echo "[env] Retrying from the archive directly: $src"
      pip install --no-index --no-deps --no-build-isolation "$src" >/dev/null 2>&1
      python -c "import CosinorPy" 2>/dev/null && return 0
    fi
  else
    echo "[env] No cached CosinorPy archive; trying indexed install (needs internet)..."
    pip install --no-deps CosinorPy >/dev/null 2>&1
    python -c "import CosinorPy" 2>/dev/null && return 0
  fi
  return 1
}

if install_cosinorpy; then
  echo "[env] CosinorPy import verified -- paper-cosinor baseline enabled."
else
  echo "  [WARN] CosinorPy unavailable after 3 attempts -- paper-cosinor row will be MISSING"
  echo "         from every variant of this task. Reason:"
  python -c "import CosinorPy" 2>&1 | tail -2 | sed 's/^/           /'
  echo "         If the cache is empty, run ONCE on the LOGIN node:"
  echo "           module purge && module load StdEnv/2023 python/3.11"
  echo "           pip download --no-deps CosinorPy -d $WHEEL_CACHE"
  echo "         (--no-deps is required: without it pip tries to BUILD scipy and fails on OpenBLAS.)"
fi

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
# ---- PREFLIGHT: every project module the run needs must be present, BEFORE training ----
# Runs 66404249, 66440129 and 66465766 (195 variants total) each trained for hours and only
# then dropped the "Cosinor (paper)" baseline, because baselines/cosinor.py was missing from the upload
# and the failure was reported per-variant in paper_cosinor.FAILED.txt -- invisible until the
# results were collected. A missing module is an UPLOAD mistake, so catch it in seconds here.
# tasks/rhythm.py imports cosinor lazily on purpose (a dependency problem must not kill a sweep);
# this check is what makes that safe, by refusing to start the sweep in the first place.
echo ""
echo "[preflight] Checking required project modules in $PROJECT ..."
PREFLIGHT_MISSING=""
for f in cost.py train_hrd.py data_processing/data_preprocessing.py utils.py tasks/rhythm.py baselines/cosinor.py \
         tasks/decomposition.py; do
  [ -f "$PROJECT/$f" ] || PREFLIGHT_MISSING="$PREFLIGHT_MISSING $f"
done
[ -d "$PROJECT/models" ] || PREFLIGHT_MISSING="$PREFLIGHT_MISSING models/"
if [ -n "$PREFLIGHT_MISSING" ]; then
  echo "  [FATAL] missing from the project directory:$PREFLIGHT_MISSING"
  echo "          These are project files, not pip packages -- the upload skipped them."
  echo "          Fix from your LOCAL machine (repo root), then resubmit:"
  echo "            rsync -avP --exclude='.git' --exclude='__pycache__' --exclude='results_hrd' \\"
  echo "                  ./ <user>@<host>:$PROJECT/"
  echo "          Upload the whole tree ('./'), never a list of filenames -- that is how"
  echo "          these files went missing three sweeps in a row. See NARVAL.md Step 1."
  exit 1
fi
echo "  all required project modules present"

# CosinorPy is a pip package, not a project file, so it is a WARNING not a fatal: the sweep is
# still worth running without the paper-cosinor baseline row. Reported loudly and once, here,
# instead of 65x inside the per-variant result folders.
if python -c "import cosinor" 2>/dev/null; then
  echo "  [preflight] baselines/cosinor.py imports OK (paper-cosinor baseline enabled)"
else
  echo "  [preflight][WARN] baselines/cosinor.py present but does not import -- the paper-cosinor row"
  echo "                    will be MISSING from every variant of this run. Reason:"
  python -c "import cosinor" 2>&1 | tail -3 | sed 's/^/                      /'
  echo "                    Usually CosinorPy: run ONCE on a LOGIN node ('--no-deps' matters,"
  echo "                    without it pip tries to build scipy and fails on OpenBLAS):"
  echo "                      module purge && module load StdEnv/2023 python/3.11"
  echo "                      pip download --no-deps CosinorPy -d $PROJECT/wheels"
fi

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

# ---- Dataset selector -------------------------------------------------------
# "hrd"    -> minute-level HRD_RAW_MinuteLevel.csv (train_hrd.py default pipeline)
# "globem" -> segment-level GLOBEM_REDUCED.csv (4 segments/day; --dataset globem)
DATASET="globem"
if [ "$DATASET" = "globem" ]; then
  SENSOR_CSV="datasets/GLOBEM_REDUCED.csv"
else
  SENSOR_CSV="datasets/HRD_RAW_MinuteLevel.csv"
fi

echo ""
echo "[validation] Checking data file ($DATASET)..."
if [ ! -f "$SENSOR_CSV" ]; then
  echo "ERROR: $SENSOR_CSV not found!"
  exit 1
fi
echo "  [OK] Data file found: $(ls -lh "$SENSOR_CSV" | awk '{print $5, $9}')"

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

# ==========================================================================
# 4. TRAINING — CONFIGURABLE SWEEP
# ==========================================================================
# Edit only this block to define the exact run. The script uses these values
# directly, so the executed sweep matches the settings here.

echo "=========================================="
echo "RUNNING CONFIGURED SWEEP"
echo "=========================================="

# SENSOR_CSV is set by the DATASET selector above (hrd -> HRD csv, globem -> GLOBEM csv).
OUTPUT_DIR="results_hrd"
RUN_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"

# Seed control:
#   RUN_SEEDS_MODE=all   -> run every seed listed below in one job (sequential; only for a few runs)
#   RUN_SEEDS_MODE=array -> one seed per SLURM array task (parallel; needed for the full sweep)
# One array task per seed (parallel). Keep #SBATCH --array above in sync with the SEEDS count:
# 3 seeds -> --array=0-2, 5 seeds -> --array=0-4. Per task = (#VARIANTS x #DIS_MODE) runs.
RUN_SEEDS_MODE="array"
# TEN seeds this time, not five. Each seed re-draws the 100-participant test set, and that
# choice turned out to dominate everything: on run 66496147 the held-out split explained 50.9%
# of the variance in the amplitude probe AUC while the model choice explained 19.8%. With only
# 5 seeds the encoding comparison had no power -- one-way ANOVA across encoding families gave
# p=0.67 and p=0.93 on two independent runs, and the family ranking flipped between them.
# Doubling the seeds shrinks the standard error of each variant's mean by ~30%.
#
# Drawn to collide with NONE of the 15 seeds used so far (11 42 56 184 337 / 140 193 709 869
# 997 / 156 267 369 454 827), because collect_results.py pools every run under --results-dir:
# a repeated seed is counted twice and silently shrinks the reported sd. Runs 66404249 and
# 66440129 share one batch, and 66465766 and 66496147 share another -- so the pooled n=10 in
# those tables is really n=5.
SEEDS=(52 12 43 23 90)
# the other five from the 10-seed sweeps, held back for a follow-up: 82 45 14 21 351

# Cross-dataset evaluation, GLOBEM protocol A. Each entry is one fold; the held-out
# cohort is the test set and is excluded from pretraining as well as from the probe.
#   DS1 DS2 DS3 DS4  -> leave-one-dataset-out (the paper's Table 9)
#   pre post         -> pre/post-COVID, both directions (the paper's Table 10)
# EMPTY () -> the original random class-balanced holdout (--test-per-class).
# The array runs one task per (seed, holdout) PAIR, so #SBATCH --array must be
# 0-(#SEEDS x #HOLDOUTS - 1):  10 seeds x 6 folds -> --array=0-59.
HOLDOUTS=(DS1 DS2 DS3 DS4)

# Which GLOBEM label the probe is trained and scored on. "weekly" = each weekly PHQ-4 survey
# is a sample (a STATE measure that fluctuates week to week); "endpoint" = one BDI-II label
# per participant (a TRAIT measure). Circadian disruption is trait-like: with identical
# features, windows and participant-disjoint folds, classical circadian descriptors reach
# AUC 0.581 on the endpoint label against 0.544 on the weekly one. Both entries here -> every
# (seed, holdout) task runs both labels.
LABELS=(weekly endpoint)
# earlier batches, for reference / to reproduce those runs:
#   66404249, 66440129 -> SEEDS=(11 42 56 184 337)
#   66465766, 66496147 -> SEEDS=(140 193 709 869 997)
#   66517709           -> SEEDS=(156 267 369 454 827)

# Variant control: each entry is "backbone:pe". ALL valid backbone x PE combinations:
#   TCN accepts only {none, time2vec}; Transformer accepts all 8 PE methods + time2vec;
#   vit accepts only {none} -- the ViT-2D backbone (SensorLM-style [channel x time] grid)
#   has an INTRINSIC encoding: Learnable Fourier (Li et al. 2021) on the metric time axis
#   + a discrete embedding on the non-metric channel axis, so no swappable PE applies.
VARIANTS=(
  tcn:none tcn:time2vec
  transformer:sinusoidal transformer:learnable transformer:tape
  transformer:rpe transformer:erpe transformer:tupe
  transformer:convspe transformer:tpe transformer:time2vec
  vit:none
)

# Training hyperparameters.
ALPHA="0.005"
LR="5e-4"
ITERS="6000"
BATCH_SIZE=64
# Augmentation. Jitter is active (CoST default 0.1). Timestep MASKING is not: "none" matches
# upstream CoST, which never applied one -- its encoder's mask argument hard-defaulted to
# 'all_true', so mask_mode was unreachable and MASK_KEEP_PROB had NO effect. Set
# MASK_MODE=binomial to actually enable it; MASK_KEEP_PROB is read only in that case.
JITTER_SIGMA="0.1"
MASK_MODE="none"
MASK_KEEP_PROB="0.5"
# Seasonal-loss phase comparison:
#   circular      -- [sin, cos]; correct across the +/-pi branch cut
#   circular_amp  -- circular, additionally weighted by each channel's amplitude, so
#                    channels whose phase is undefined noise stop counting as much as
#                    real rhythms (strict generalisation; same logit scale)
#   raw           -- upstream CoST's raw angle; only to reproduce archived runs
PHASE_ENCODING="circular"
# Optional crossed-seed design. Leave EMPTY to keep the historical behaviour where --seed
# drives both the participant split and the model init. Set one or both to separate cohort
# variance from optimisation variance (see train_hrd.py --split-seed / --model-seed).
SPLIT_SEED=""
MODEL_SEED=""
# NB plain `if`, not `[ -n "$X" ] && ...`: under `set -e` (line 26) a false test is a non-zero
# exit status and would abort the whole job whenever the seeds are left unset -- i.e. always.
SEED_FLAGS=""
if [ -n "$SPLIT_SEED" ]; then SEED_FLAGS="$SEED_FLAGS --split-seed $SPLIT_SEED"; fi
if [ -n "$MODEL_SEED" ]; then SEED_FLAGS="$SEED_FLAGS --model-seed $MODEL_SEED"; fi
# GLOBEM windowing -- WEEKLY windows (per project agreement). IGNORED for the HRD dataset
# (which windows via --window-hours/--bin-minutes). 7 days x 4 segments/day -> T=28, so the
# seasonal-FFT longest resolvable period = the whole 168 h window (weekly), matching HRD.
# STRIDE_DAYS=7 slides one week at a time (non-overlapping); lower it for more, overlapping
# windows (more SSL samples) if the weekly-non-overlap count comes out too small.
WINDOW_DAYS=7
STRIDE_DAYS=7
# ViT-2D attends over the [channel x time] grid and is far heavier than the TCN/Transformer:
# measured forward+backward at the GLOBEM shape (T=28, C=12) it needs ~0.6/0.9/3.6 GB at batch
# 16/32/64 against ~0.1 GB for the others. The old limit of 16 was set for the HRD shape
# (T=672), where the grid is 14x larger and batch 64 OOM'd on the H100; at T=28 that reason no
# longer applies, so 32 is used here. MoCo's queue (K=4096) keeps the negatives large either way.
VIT_BATCH_SIZE=32
TEST_PER_CLASS=50
GPU_INDEX=0

# ---------------------------------------------------------------------------
# CLOCK / CALENDAR COVARIATES.  "yes" appends CoST's 7 calendar channels to the sensors and
# injects them as an ADDITIVE temporal encoding (encoder.time_fc), outside input_fc and outside
# the seasonal decomposition. "no" drops them; results then land in folders WITHOUT the _clock
# suffix, so they cannot be confused with clock-on runs.
#
# Set to "no" for the ablation this is currently configured for. Measured on GLOBEM, only 2 of
# the 7 channels are real temporal signal; the rest are dead or encode the calendar DATE:
#     minute      constant 0 everywhere (segments are 6 h, so there is no minute resolution)
#     hour        varies within the window  -> the 24 h cycle, {0,6,12,18}
#     dayofweek   varies within the window  -> the weekly cycle, 0..6
#     day         mixed
#     dayofyear   almost constant within a window, varies BETWEEN windows
#     month       almost constant within a window, varies BETWEEN windows
#     weekofyear  EXACTLY constant within a window, varies BETWEEN windows
# The last three do not encode time-within-the-window at all; they tell the model which calendar
# period a window came from. GLOBEM spans 2018-2021 and participants enrolled at different dates,
# so those channels are a season/cohort confound rather than a rhythm signal. And `hour` hands
# the circadian phase to the model directly, so part of the reported rhythm capture may come from
# the clock rather than from the sensors -- which is exactly what this ablation tests.
#
# The comparison is FREE: run 66840586 is the clock-ON arm with the SAME 10 seeds and the same
# config, so this run pairs against it directly. Note vit never took clock channels,
# so they are identical in both arms and only the other 11 variants carry information.
CLOCK_FEATURES="no"

# Loss balancing between the trend and seasonal branches:
#   fixed    -> weight seasonal by $ALPHA (original CoST behaviour) -- FAST
#   gradnorm -> adaptively balance the two losses (GradNorm), $ALPHA ignored -- ~2.3x SLOWER
# GradNorm consistently drove seasonal weight to ~0.001, so 'fixed' with a small alpha gives
# essentially the same model much faster. Flip to gradnorm only for the one-off ablation.
LOSS_BALANCE="fixed"

# Split cohort:  consistent -> only baseline==endpoint (clean label, fewer people)
#                labeled    -> every participant with an endpoint label (MORE samples)
COHORT="labeled"

# Probe/test on ONE last-week window per participant (no pseudo-replication, label-closest).
# "yes" / "no".  PROBE_C = inverse L2 of the probe (small = strong reg; use ~0.1 for last-window).
PROBE_LAST_WINDOW="yes"
PROBE_C="0.1"

# Participant-level k-fold CV WITHIN the probe pool (the 36-participant test set is untouched):
# the decision threshold is tuned on pooled out-of-fold predictions and the final probe is
# refit on ALL pool participants. Also reports an internal OOF-CV metric. 1 = single split.
CV_FOLDS="5"

# Which MODEL to run per variant (same backbone / PE / augmentations in both):
#   cost  -> CoST: trend MoCo + seasonal FFT disentangler (--disentangle). Produces the
#            full disentanglement/rhythm eval files (decomposition_recovery, hrd_rhythm, ...).
#   plain -> a PLAIN single-representation self-supervised encoder (--no-disentangle):
#            NO trend/seasonal split at all, one standard MoCo on the encoder output.
#            Tagged _plain (e.g. rpe_plain); ONLY the prediction result (metrics.json) --
#            no disentanglement eval files (there are no branches to probe).
#   both  -> run CoST AND the plain model for each variant (head-to-head prediction ablation)
DIS_MODE="cost"

# -- Always emit the summary table + summary.csv when the job ENDS, no matter how --
# Registered as an EXIT trap so the CSV is written on normal completion, on a variant crash,
# OR on a SLURM wall-clock timeout (SIGTERM). collect_results tolerates partial/failed runs,
# so whatever finished is never lost. summary.csv was previously missing because the job died
# (set -e) before reaching the collect step; via this trap it can no longer be skipped.
summarize() {
  echo ""
  echo "=========================================="
  echo "SUMMARY  (all results collected so far under run $RUN_ID)"
  echo "=========================================="
  python scripts/collect_results.py \
    --results-dir "$OUTPUT_DIR/$RUN_ID" \
    --csv "$OUTPUT_DIR/$RUN_ID/summary.csv" \
    || echo "  (collect_results found nothing yet / failed; safe to re-run manually)"
}
trap summarize EXIT

# An empty HOLDOUTS means "no cross-dataset split" -- represent it as a single empty-string
# entry so the array arithmetic below is identical either way.
[ ${#HOLDOUTS[@]} -eq 0 ] && HOLDOUTS=("")
N_HOLD=${#HOLDOUTS[@]}
N_TASKS=$(( ${#SEEDS[@]} * N_HOLD ))

if [ "$RUN_SEEDS_MODE" = "array" ]; then
  if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "ERROR: RUN_SEEDS_MODE=array requires sbatch --array."
    exit 1
  fi
  if [ "${SLURM_ARRAY_TASK_ID}" -ge "$N_TASKS" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} is out of range: ${#SEEDS[@]} seeds"
    echo "       x ${N_HOLD} holdout(s) = ${N_TASKS} task(s). Fix: #SBATCH --array=0-$(( N_TASKS - 1 ))."
    exit 1
  fi
  # Loud guard: SLURM parses '#SBATCH --array' at submission (before this script runs), so it
  # cannot be derived from SEEDS/HOLDOUTS. If the array has FEWER tasks than the sweep needs,
  # the tail silently never runs -- warn so the mismatch is visible in every task's log.
  if [ -n "${SLURM_ARRAY_TASK_COUNT:-}" ] && [ "${SLURM_ARRAY_TASK_COUNT}" -lt "$N_TASKS" ]; then
    echo "WARNING: this array has only ${SLURM_ARRAY_TASK_COUNT} task(s) but the sweep needs ${N_TASKS} "
    echo "         (${#SEEDS[@]} seeds x ${N_HOLD} holdout(s)) -> the tail will NOT run."
    echo "         Fix: #SBATCH --array=0-$(( N_TASKS - 1 ))."
  fi
  # One task per (seed, holdout) PAIR. Keeping the pair -- rather than looping holdouts inside a
  # task -- means every task does exactly one seed's worth of variants, so adding folds widens
  # the array instead of lengthening each task, and the 24 h wall-clock stays valid unchanged.
  TASK_SEEDS=("${SEEDS[$(( SLURM_ARRAY_TASK_ID / N_HOLD ))]}")
  TASK_HOLDOUTS=("${HOLDOUTS[$(( SLURM_ARRAY_TASK_ID % N_HOLD ))]}")
else
  TASK_SEEDS=("${SEEDS[@]}")
  TASK_HOLDOUTS=("${HOLDOUTS[@]}")
fi
echo "[sweep] seeds=${TASK_SEEDS[*]} | holdouts='${TASK_HOLDOUTS[*]}' | ${#VARIANTS[@]} variants"

for SEED in "${TASK_SEEDS[@]}"; do
for HOLD in "${TASK_HOLDOUTS[@]}"; do
echo ""
echo "################# SEED = $SEED${HOLD:+  HOLDOUT = $HOLD} #################"
# Protocol A: the held-out GLOBEM cohort is the test set AND is excluded from pretraining.
# Empty -> train_hrd's default random class-balanced holdout (--test-per-class).
HOLD_FLAG=""; [ -n "$HOLD" ] && HOLD_FLAG="--holdout $HOLD"
for GLABEL in "${LABELS[@]}"; do
echo "--- label = $GLABEL ---"
for V in "${VARIANTS[@]}"; do
  BACKBONE="${V%%:*}"; PE="${V##*:}"
  if [ "$BACKBONE" = "tcn" ]; then HID=64; REPR=320; DEPTH=10; VBATCH="$BATCH_SIZE"
  elif [ "$BACKBONE" = "vit" ]; then HID=48; REPR=240; DEPTH=4; VBATCH="$VIT_BATCH_SIZE"
  else HID=48; REPR=240; DEPTH=4; VBATCH="$BATCH_SIZE"; fi
  # EQUAL SAMPLE BUDGET, not equal iteration budget. --iters counts optimiser STEPS, so a
  # backbone forced onto a smaller batch sees proportionally less data: at batch 16 the ViT
  # saw 96k windows (14.5 epochs) against the TCN's 384k (58 epochs) -- and 70% of its runs
  # still had their best validation loss at the final epoch, i.e. it was under-trained, which
  # confounds every backbone comparison. Scaling iters by BATCH_SIZE/VBATCH equalises windows
  # seen; wall-clock is unchanged because each step is correspondingly cheaper.
  VITERS=$(( ITERS * BATCH_SIZE / VBATCH ))
  LW_FLAG=""; [ "$PROBE_LAST_WINDOW" = "yes" ] && LW_FLAG="--probe-last-window"
  # Clock/time features, controlled by $CLOCK_FEATURES above. The ViT backbones reject appended
  # clock channels, so they are always run without them regardless of the setting.
  CLOCK_FLAG=""
  [ "$CLOCK_FEATURES" = "yes" ] && CLOCK_FLAG="--with-clock-features"
  case "$BACKBONE" in vit) CLOCK_FLAG="" ;; esac
  # CoST (disentangled) and/or the plain-SSL (no disentangler) baseline, per DIS_MODE
  case "$DIS_MODE" in
    cost)  DIS_MODES="--disentangle" ;;
    plain) DIS_MODES="--no-disentangle" ;;
    both)  DIS_MODES="--disentangle --no-disentangle" ;;
    *) echo "ERROR: DIS_MODE must be cost|plain|both (got '$DIS_MODE')"; exit 1 ;;
  esac
  for DIS in $DIS_MODES; do
  echo "--- $BACKBONE / $PE  ($DIS, loss-balance=$LOSS_BALANCE) ---"
  # NON-FATAL per variant: one variant crashing (e.g. ViT running out of GPU memory at the
  # full 672-step sequence) must NOT abort the whole sweep or skip the summary/CSV below.
  # Without this guard, `set -e` kills the job on the first failure -- which is exactly how a
  # single failing variant previously left both its own results AND summary.csv missing.
  if python train_globem.py \
    --sensor-csv "$SENSOR_CSV" \
    --dataset "$DATASET" \
    --window-days "$WINDOW_DAYS" --stride-days "$STRIDE_DAYS" \
    --backbone "$BACKBONE" \
    --pe "$PE" \
    --hidden-dims $HID --repr-dims $REPR --depth $DEPTH \
    --alpha "$ALPHA" --loss-balance "$LOSS_BALANCE" $DIS \
    --jitter-sigma "$JITTER_SIGMA" \
    --mask-mode "$MASK_MODE" --mask-keep-prob "$MASK_KEEP_PROB" \
    --phase-encoding "$PHASE_ENCODING" \
    $SEED_FLAGS \
    --cohort "$COHORT" $LW_FLAG --probe-c "$PROBE_C" --cv-folds "$CV_FOLDS" \
    --globem-label "$GLABEL" \
    --lr "$LR" \
    --iters "$VITERS" --batch-size "$VBATCH" --test-per-class "$TEST_PER_CLASS" \
    $CLOCK_FLAG \
    --output-dir "$OUTPUT_DIR" --run-id "$RUN_ID" --seed "$SEED" --gpu "$GPU_INDEX" ; then
    echo "[OK]   $BACKBONE/$PE ($DIS) seed $SEED"
  else
    echo "[WARN] $BACKBONE/$PE ($DIS) seed $SEED FAILED -- continuing sweep (see traceback above)"
  fi
  done
done
done   # ===== end label loop (LABELS) =====
done   # ===== end holdout loop (TASK_HOLDOUTS) =====
done   # ===== end seed loop (SEEDS) =====

# (Clock/time features are controlled by $CLOCK_FEATURES above and applied via $CLOCK_FLAG for every
#  backbone except vit, to match salesforce/CoST: the 7 CoST calendar covariates
#  [minute, hour, dayofweek, day, dayofyear, month, weekofyear] are standardised and appended
#  to the 7 sensor channels -> model input C = 14. Drop the flag for the no-clock ablation, C = 7.)

# (Positional-encoding figure step removed: it needed per-variant net.pt
#  checkpoints, which are no longer saved -- to keep storage low. Build those
#  figures on demand from any run that still has net.pt, e.g.:
#    python scripts/plot_position_similarity.py --cosine --ckpt-dir results_hrd/<jobid>)

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
echo "Output files (shared folder for all seeds):"
ls -lhR "$OUTPUT_DIR/$RUN_ID/" 2>/dev/null || echo "  (no results yet)"

# --- Per-seed detail + aggregated mean +/- std -------------------------------
# The summary table + summary.csv are emitted by the summarize() EXIT trap (registered above),
# so they are produced exactly once when this task ends -- even if a variant crashed or the
# job timed out. Each array task writes to the SHARED folder; the LAST task to finish sees all
# seeds. For the definitive table after ALL tasks finish, re-run:
#   python scripts/collect_results.py --results-dir $OUTPUT_DIR/$RUN_ID
