#!/bin/bash
# =====================================================================
# SLURM runner: CoST on HRD wearable data (depression endpoint + energy probe).
# One (seed, variant) pair per array task -- a whole seed in one task needs ~40 h
# and gets cut off by the wall clock.
#
#   bash   scripts/run.sh                   # depression sweep (SEEDS x VARIANTS tasks)
#   bash   scripts/run.sh --dry-run         # print the sbatch command only
#   bash   scripts/run.sh --concurrent 6    # cap running tasks (default 12)
#   bash   scripts/run.sh -- --time=8:00:00 # after -- goes to sbatch verbatim
#   sbatch scripts/run.sh                   # depression sweep via the #SBATCH --array below
#
# Both work. `sbatch` uses the hard-coded --array, which goes stale when the sweep grows;
# task 0 notices and submits the missing tail into the same results folder. `bash` computes
# the range up front and needs no healing.
#
# TCN variants are ~1.5x the rest; --time cannot vary within one array, so submit
# them separately to keep the rest inside the 3 h partition bucket:
#   bash scripts/run.sh --dry-run           # shows the task ids
#   sbatch --time=8:00:00 --array=<tcn ids> scripts/run.sh
# TCN entries are variants 0,1 -> task ids {0,1} + N_VARIANTS*seed_index.
#
# Results: results_hrd/<arrayjobid>/{backbone}_{pe}_seed{SEED}/
# Monitor: squeue -u $USER ; tail -f logs/cost_hrd-<arrayjobid>_<taskid>.out
# =====================================================================
#SBATCH --account=def-plago
#SBATCH --job-name=cost_hrd
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1   # MIG slice: H100 speed, 40GB, many slices -> short queue (full h100:1 waits days)
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G                    # 32G OOM-killed 17/28 tasks in 18535364 before the energy rhythm analysis was subsampled; 64G will not schedule on this node type.
#SBATCH --time=8:00:00               # Was 3h (a <=3h job is eligible for BOTH the 3h and 12h buckets, ~20x less queued work ahead of it). One array cannot vary --time per task, so it must cover the WORST task: tcn:none and transformer:sinusoidal also pretrain the plain-SSL twin, i.e. two full pretrainings plus q1/q2/q3. That no longer fits 3h, so the whole sweep moves to the 12h bucket and queues longer. This is the price of one-command submission; splitting the array by task id (see header) buys the 3h bucket back for the other 60.
#SBATCH --array=0-71%12              # 6 seeds x 12 variants. Goes stale when the sweep grows,
#                                    # so task 0 self-heals: it detects a short array and
#                                    # submits the remainder (see SELF-HEALING ARRAY below).
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --mail-user=melikaseyedi@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

set -e

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

PROJECT=~/projects/def-plago/melikas/projects/rhythmssl_project
mkdir -p "$PROJECT"
cd "$PROJECT" || { echo "Cannot cd to $PROJECT"; exit 1; }
mkdir -p logs results_hrd results_hrd_energy

# ============================================================================
# SWEEP CONFIGURATION -- the only block you normally edit.
# Defined above the env setup so the self-submission block can read it on a
# login node without loading modules.
# ============================================================================
OUTPUT_DIR="results_hrd"

# Re-running the same seeds on unchanged code reproduces the same numbers bit for
# bit, so new seeds are the only way to add information.
SEEDS=(42 82 22 91 53 45)         # 6 seeds

# "backbone:pe". TCN accepts {none, time2vec, factorized}; Transformer accepts all 8 PE
# methods + time2vec + factorized.
#
# factorized = the calendar-anchored PE of WavesFM Eq. 2 (two learnable tables indexed by
# the bin's real time-of-day / day-of-week). Its controls are already in this sweep:
#   transformer:learnable  -> same learnable-table form, INDEX-anchored  (isolates the anchor)
#   tcn:none               -> relative position only, no absolute calendar
# circular = the hand-designed twin of factorized: same wall-clock anchor and the same
# additive site, but a FIXED sin/cos basis instead of learnable per-bin tables. Both are
# input-side encodings, so both run on either backbone -- which is what makes the wall-clock
# family readable free of the backbone (factorized alone was Transformer-only).
VARIANTS=(
  tcn:none tcn:time2vec tcn:circular
  transformer:sinusoidal transformer:learnable transformer:tpe
  transformer:rpe transformer:erpe transformer:tupe
  transformer:convspe
  transformer:factorized transformer:circular
)
#transformer:time2vec
#transformer:tpe
#tcn:factorized

N_SEEDS=${#SEEDS[@]}
N_VARIANTS=${#VARIANTS[@]}

N_JOBS=$(( N_SEEDS * N_VARIANTS ))                # one task per (seed, variant)

# ============================================================================
# SELF-SUBMISSION (skipped when SLURM is already running us)
# ============================================================================
if [ -z "${SLURM_JOB_ID:-}" ]; then
  _concurrent=12; _dry=0; _pass=()
  while [ $# -gt 0 ]; do
    case "$1" in
      --concurrent) _concurrent="$2"; shift 2 ;;
      --dry-run)    _dry=1;           shift ;;
      --)           shift; _pass+=("$@"); break ;;
      *)            _pass+=("$1");    shift ;;
    esac
  done
  _time="8:00:00"                         # must match the #SBATCH default above

  _cmd=(sbatch "--array=0-$(( N_JOBS - 1 ))%${_concurrent}" "--time=$_time"
        "--job-name=cost_hrd" "--export=ALL")
  # Plain `if`: under `set -e` a false one-liner test is itself a failing command.
  if [ ${#_pass[@]} -gt 0 ]; then _cmd+=("${_pass[@]}"); fi
  _cmd+=("$SELF")

  echo "HRD: $N_SEEDS seeds x $N_VARIANTS variants = $N_JOBS tasks | time $_time"
  echo "${_cmd[*]}"
  if [ "$_dry" -eq 1 ]; then echo "(dry run -- nothing submitted)"; exit 0; fi
  exec "${_cmd[@]}"
fi

echo "=== CoST/HRD | job $SLURM_JOB_ID | array ${SLURM_ARRAY_JOB_ID:-n/a}/${SLURM_ARRAY_TASK_ID:-n/a} | $(hostname) | $(date) ==="

# ============================================================================
# ENVIRONMENT
# ============================================================================
module purge
module load StdEnv/2023 python/3.11
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch numpy pandas scikit-learn einops matplotlib umap-learn

# seaborn is a HARD import-time dependency of CosinorPy; without it the
# "Cosinor (paper)" baseline is silently skipped. CosinorPy itself is not in the
# Alliance wheelhouse, so it comes from PyPI (Nibi compute nodes do reach it).
pip install --no-index seaborn || echo "  [WARN] no seaborn -> CosinorPy import will fail"
pip install CosinorPy || echo "  [WARN] CosinorPy install failed -> paper-cosinor view skipped"
python -c "from CosinorPy import cosinor" 2>/dev/null \
  && echo "[env] CosinorPy OK" \
  || echo "[env] [WARN] CosinorPy import FAILS -- every variant loses the 'Cosinor (paper)' row"

export MPLBACKEND=Agg
export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUBLAS_WORKSPACE_CONFIG=:16:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

# Catch a half-finished scp BEFORE burning the array: a missing module makes every
# task die on import in seconds and still report COMPLETED (array 18523312).
MISSING=""
for _f in train_hrd.py data_processing/data_preprocessing.py cost.py utils.py \
          baselines/cosinor.py baselines/supervised.py \
          tasks/rhythm.py tasks/decomposition.py tasks/_eval_protocols.py tasks/energy.py \
          tasks/_experiment_common.py \
          experiment_q1.py experiment_q2.py experiment_q3.py \
          scripts/collect_results.py scripts/_results.py scripts/_style.py \
          models datasets/HRD_RAW_MinuteLevel.csv; do
  [ -e "$_f" ] || MISSING="$MISSING $_f"
done
[ -n "$MISSING" ] && { echo "ERROR: missing from $PROJECT:$MISSING"; exit 1; }

# ============================================================================
# HYPERPARAMETERS
# ============================================================================
SENSOR_CSV="datasets/HRD_RAW_MinuteLevel.csv"
RUN_ID="${RUN_ID:-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}}"

ALPHA="0.005"; LR="5e-4"; ITERS="6000"; BATCH_SIZE=64
JITTER_SIGMA="0.1"
# Timestep-masking augmentation. "none" (default) matches upstream CoST, which never applied
# one -- its encoder's mask argument hard-defaulted to 'all_true', so mask_mode was
# unreachable and MASK_KEEP_PROB had NO effect. Set MASK_MODE=binomial to actually enable it;
# MASK_KEEP_PROB is read only in that case.
MASK_MODE="none"; MASK_KEEP_PROB="0.5"
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
SPLIT_SEED=""; MODEL_SEED=""
TEST_PER_CLASS=18; GPU_INDEX=0
COHORT="labeled"          # every participant with an endpoint label
PROBE_UNIT="last"         # one probe sample = participant's most recent window
PROBE_C="0.1"             # inverse L2; small, since last/persubject give 1 row per participant
CV_FOLDS="5"              # participant-level CV inside the probe pool; test set untouched

# Calendar covariates OFF: run 18676256 could not separate the encoding operator
# from the input, because its four clock-free configs were exactly its four best.
# With this empty, every variant sees the identical 4 sensor channels.
CLOCK_FLAG=""

# Also probe daily emotional_energy on the SAME held-out participants, reusing the
# frozen encoder (no extra pretraining). Writes to results_hrd_energy/<RUN_ID>/.
ENERGY_FLAG="--energy-probe --energy-output-dir results_hrd_energy"

# ============================================================================
# DISPATCH: task id -> one (seed, variant)
# ============================================================================
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
  echo "ERROR: no array task id. Submit with 'sbatch scripts/run.sh' or 'bash scripts/run.sh'."
  exit 1
fi
if [ "${SLURM_ARRAY_TASK_ID}" -ge "$N_JOBS" ]; then
  echo "NOTE: task ${SLURM_ARRAY_TASK_ID} is beyond the ${N_JOBS} configured pairs. Nothing to do."
  exit 0
fi
# SELF-HEALING ARRAY. The #SBATCH --array directive is read at submission time and cannot see
# SEEDS/VARIANTS, so it goes stale whenever the sweep grows -- that is how 19037752 silently
# ran 44 of 52 pairs. Rather than forbid `sbatch scripts/run.sh`, task 0 detects the shortfall
# and submits the remainder into the SAME results folder.
# Only for a CONTIGUOUS array starting at 0 (min=0 and max=count-1): a hand-picked sparse array
# like --array=0,1,2,13,... is deliberate and must never be extended.
if [ "${SLURM_ARRAY_TASK_ID}" = "0" ] && [ -n "${SLURM_ARRAY_TASK_COUNT:-}" ] \
   && [ "${SLURM_ARRAY_TASK_COUNT}" -lt "$N_JOBS" ] \
   && [ "${SLURM_ARRAY_TASK_MIN:-1}" = "0" ] \
   && [ "${SLURM_ARRAY_TASK_MAX:--1}" = "$(( SLURM_ARRAY_TASK_COUNT - 1 ))" ]; then
  echo "[self-heal] array covers ${SLURM_ARRAY_TASK_COUNT} of ${N_JOBS} pairs; submitting the rest"
  sbatch --array="${SLURM_ARRAY_TASK_COUNT}-$(( N_JOBS - 1 ))%12" \
         --time="8:00:00" --job-name="cost_hrd" \
         --export=ALL,RUN_ID="$RUN_ID" "$SELF" \
    && echo "[self-heal] queued tasks ${SLURM_ARRAY_TASK_COUNT}-$(( N_JOBS - 1 )) into $RUN_ID" \
    || echo "[self-heal] FAILED -- submit by hand: RUN_ID=$RUN_ID sbatch --array=${SLURM_ARRAY_TASK_COUNT}-$(( N_JOBS - 1 )) scripts/run.sh"
fi

# ------------------------------------------------------------------- HRD depression sweep
SEED="${SEEDS[$(( SLURM_ARRAY_TASK_ID / N_VARIANTS ))]}"
V="${VARIANTS[$(( SLURM_ARRAY_TASK_ID % N_VARIANTS ))]}"
BACKBONE="${V%%:*}"; PE="${V##*:}"
echo "[dispatch] task ${SLURM_ARRAY_TASK_ID}: seed $SEED | $BACKBONE / $PE"

# encoder.pt is ~235 MB (the complex SFD table alone is 138 MB), so all 70 pairs would be
# ~16 GB. It is now written ALWAYS, because experiment_q1/q2/q3 reload it to reuse this
# task's pretraining instead of retraining -- but it is deleted again at the end of the task
# unless this is the FIRST seed, which is kept for later re-analysis. Peak cost is therefore
# ~235 MB per RUNNING task, not 16 GB.
SAVE_ENC="--save-encoder"
KEEP_ENC=0
if [ "$SEED" = "${SEEDS[0]}" ]; then KEEP_ENC=1; fi

# Queue the summary ONCE, from task 0, depending on the whole array. It cannot be
# an EXIT trap (tasks would race on summary.csv) nor queued at submission time
# (the array job id is not known until sbatch returns).
if [ "${SLURM_ARRAY_TASK_ID}" = "0" ] && [ -n "${SLURM_ARRAY_JOB_ID:-}" ]; then
  _sum="cd $PROJECT && python scripts/collect_results.py --results-dir $OUTPUT_DIR/$RUN_ID --csv $OUTPUT_DIR/$RUN_ID/summary.csv"
  sbatch --dependency=afterany:"$SLURM_ARRAY_JOB_ID" --account="$SLURM_JOB_ACCOUNT" \
         --time=30:00 --mem=8G --cpus-per-task=1 --job-name=cost_sum \
         --output="logs/cost_sum-%j.out" --wrap "$_sum" 2>/dev/null \
    && echo "[summary] queued for after array $SLURM_ARRAY_JOB_ID" \
    || echo "[summary] queue failed; run by hand afterwards: $_sum"
fi

# ============================================================================
# TRAIN
# ============================================================================
if [ "$BACKBONE" = "tcn" ]; then HID=64; REPR=320; DEPTH=10
else HID=48; REPR=240; DEPTH=4; fi

# Only pass the crossed-seed flags when set, so the default invocation is byte-identical to
# before and keeps writing the historical `_seed<N>` folder name.
# NB plain `if`, not `[ -n "$X" ] && ...`: under `set -e` (line 39) a false test is a non-zero
# exit status and would abort the whole job whenever the seeds are left unset -- i.e. always.
SEED_FLAGS=""
if [ -n "$SPLIT_SEED" ]; then SEED_FLAGS="$SEED_FLAGS --split-seed $SPLIT_SEED"; fi
if [ -n "$MODEL_SEED" ]; then SEED_FLAGS="$SEED_FLAGS --model-seed $MODEL_SEED"; fi
# train_hrd.py names the folder `_seed<model>` when both seeds agree, `_seed<model>_split<split>`
# when they differ; mirror that here so the completion line points at a directory that exists.
_MS="${MODEL_SEED:-$SEED}"; _SS="${SPLIT_SEED:-$SEED}"
if [ "$_MS" = "$_SS" ]; then SEED_TAG="_seed${_MS}"; else SEED_TAG="_seed${_MS}_split${_SS}"; fi

python train_hrd.py \
  --sensor-csv "$SENSOR_CSV" \
  --backbone "$BACKBONE" --pe "$PE" \
  --hidden-dims $HID --repr-dims $REPR --depth $DEPTH \
  --alpha "$ALPHA" --loss-balance fixed --disentangle \
  --jitter-sigma "$JITTER_SIGMA" \
  --mask-mode "$MASK_MODE" --mask-keep-prob "$MASK_KEEP_PROB" \
  --phase-encoding "$PHASE_ENCODING" \
  --cohort "$COHORT" --probe-unit "$PROBE_UNIT" --probe-c "$PROBE_C" --cv-folds "$CV_FOLDS" \
  --lr "$LR" --iters "$ITERS" --batch-size "$BATCH_SIZE" --test-per-class "$TEST_PER_CLASS" \
  $CLOCK_FLAG $ENERGY_FLAG $SAVE_ENC $SEED_FLAGS \
  --output-dir "$OUTPUT_DIR" --run-id "$RUN_ID" --seed "$SEED" --gpu "$GPU_INDEX"

# ============================================================================
# RQ1-RQ3 EXPERIMENTS -- same (seed, variant), same frozen encoder, NO retraining.
# Each script reloads encoder.pt + metrics.json from the variant directory above, so the
# 70 pretrainings of this sweep stay 70. One shared dataset cache per node means the
# 53.5M-row CSV is parsed once for all three instead of three times.
# Non-fatal: a failed experiment must not throw away the training that just finished.
# ============================================================================
VARIANT_DIR="$OUTPUT_DIR/$RUN_ID/${BACKBONE}_${PE}${SEED_TAG}"
CACHE_DIR="${SLURM_TMPDIR:-/tmp}/hrd_cache"; mkdir -p "$CACHE_DIR"

for Q in 1 2 3; do
  echo "--- experiment_q$Q | seed $SEED | $BACKBONE / $PE | $(date +%H:%M:%S) ---"
  python "experiment_q$Q.py" --variant-dir "$VARIANT_DIR" \
      --cache-dir "$CACHE_DIR" --gpu "$GPU_INDEX" \
    || echo "[q$Q] FAILED (non-fatal) -- $VARIANT_DIR keeps its depression results"
done

# plain_encoder.pt is the non-disentangled SSL control, pretrained once by q1 and reloaded
# by q3. Same size and same reasoning as encoder.pt, so it is kept/dropped together.
if [ "$KEEP_ENC" -eq 0 ]; then rm -f "$VARIANT_DIR/encoder.pt" "$VARIANT_DIR/plain_encoder.pt"; fi

echo "=== done $(date) | $VARIANT_DIR ==="
