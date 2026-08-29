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
# This sweep is the TCN pair only: 6 seeds x 2 variants = 12 tasks, all of them heavy
# (both variants pretrain the plain-SSL twin AND run q1/q2/q3). Stage 0 first -- one seed of
# each -- so the real wall time is measured before 12 tasks are committed to a 3 h limit:
#   sbatch --time=3:00:00 --array=0,1%2 scripts/run.sh     # Stage 0 (seed 42, both variants)
#   sacct -j <jobid> --format=JobID%20,State,Elapsed        # then size Stage 1 from this
# task id -> (seed, variant): seed = SEEDS[id / 2], variant = VARIANTS[id % 2].
#
# Results: results_hrd/<arrayjobid>/{backbone}_{pe}_seed{SEED}/
# Monitor: squeue -u $USER ; tail -f logs/cost_hrd-<arrayjobid>_<taskid>.out
# =====================================================================
#SBATCH --account=def-plago
#SBATCH --job-name=cost_hrd
#SBATCH --gres=gpu:a100_3g.20gb:1    # NARVAL. Cluster-specific -- see GRES below.
#                                    # Rorqual: gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#                                    # Both are MIG slices on purpose: many slices -> a
#                                    # short queue, where a whole card waits days. SLURM
#                                    # reads this before the script runs, so it cannot be
#                                    # computed; override with `sbatch --gres=...`.
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G                    # 32G OOM-killed 17/28 tasks in 18535364 before the energy rhythm analysis was subsampled; 64G will not schedule on this node type.
#SBATCH --time=3:00:00               # A <=3h job is eligible for BOTH the 3h and 12h buckets (~20x less queued work ahead of it); at 8h it is locked into the 12h bucket. Run 19649817 was measured: the WORST task -- tcn:none, which pretrains the plain-SSL twin AND runs q1/q2/q3 -- finished in 2:23:41, and every transformer task in ~1:10. 8h was a guess that cost that array two days in ReqNodeNotAvail. Margin here is ~35 min; a rare TIMEOUT costs one rerun, which is far cheaper than the queue.
#SBATCH --array=0-63%12              # DO NOT EDIT WHEN THE SWEEP CHANGES. SLURM reads this
#                                    # before the script runs, so it cannot see SEEDS/VARIANTS.
#                                    # It is over-provisioned instead: any task id >= the real
#                                    # SEEDS x VARIANTS count exits in ~1 s at the OUT-OF-RANGE
#                                    # EXIT block, before the environment is built. 64 covers
#                                    # every sweep this file has ever run. Raise it only if a
#                                    # sweep exceeds 64 pairs; shrinking it is never necessary.
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
# ---------------------------------------------------------------------------
# DATASET. One sweep script, both cohorts. scripts/run_globem.sh is deleted rather than
# updated: it predated the RQ suite entirely -- it contained no experiment_q invocation at
# all -- so keeping it would have meant maintaining a second sweep that silently ran a
# different, smaller study. Everything below is shared; only these four knobs differ.
#
#   DATASET=globem SENSOR_CSV=datasets/GLOBEM_REDUCED.csv OUTPUT_DIR=results_globem #   ENERGY_FLAG= TEST_PER_CLASS=60 sbatch --array=0-23%12 scripts/run.sh
#
# ENERGY_FLAG must be EMPTY for globem: emotional energy is an HRD-only measurement, and the
# energy probe would fail on a cohort that never recorded it.
DATASET="${DATASET:-hrd}"
OUTPUT_DIR="${OUTPUT_DIR:-results_hrd}"

# Re-running the same seeds on unchanged code reproduces the same numbers bit for
# bit, so new seeds are the only way to add information.
# 12 seeds x 2 variants = 24 paired runs. Sized from the observed contrast: DSSL(Full) minus
# Random-init on the canonical persubject estimator was +0.0507 with sd(diff) 0.0862 over 6
# runs (p=0.209), which needs ~24 paired runs for 80% power. Six runs cannot resolve the top
# of the ladder -- its first nine rungs span 0.020 with every pairwise p >= 0.31.
SEEDS=(42 82 22 7 13 29 31 45 53 61 73 91 17 23 37 41 59 67 71 79 83 89 97 101)  # 24 seeds
# 24, not 12. The paired t over variants has SD(difference) = 0.0495 measured across this
# project's runs, so the minimum detectable effect is 2.8016*0.0495/sqrt(n): 0.040 at n=12
# and 0.028 at n=24. The subject-negatives study needs to resolve +0.030, so n=12 cannot
# answer it and n=24 barely can.

# "backbone:pe". The sweep is deliberately the TCN PAIR only.
#
#   tcn:none      relative position from the convolutions alone, no time encoding at all.
#                 This is the F0 REFERENCE: pe_contrast(ref=("tcn","none")) measures every
#                 other variant's Delta_pi against it, so it can never be dropped.
#   tcn:time2vec  Kazemi et al. 2019 fed as an INPUT feature (1 linear + 64 learnable-
#                 frequency sines). Elapsed-time structure, not calendar phase -- windows
#                 start at an arbitrary hour (align_midnight=False), so tau maps to a
#                 different wall-clock time in every window.
#
# Both are in PLAIN_REF, so both carry the plain-SSL control, and both are in Q23_VARIANTS,
# so both run RQ1, RQ2 and RQ3 in full. That makes every task in this array a "worst case" --
# see the Stage 0 note in the header before trusting the 3 h limit.
#
# CONSEQUENCE for E1.4: with two variants, Delta_pi has exactly ONE contrast arm
# (time2vec vs none). That is a valid comparison but not a family, so the Holm correction in
# pe_contrast is a no-op and the "does the temporal reference frame matter?" half of RQ1
# rests on a single pair. Add the wall-clock arms back (tcn:circular, transformer:circular,
# transformer:factorized) when that half is the claim being made.
#
# tcn:time2vec never converged in 19649817 (pretrain loss fell only 48%, to 3.50, vs ~0.10
# everywhere else) because --time2vec-dim was 16, i.e. k=15 sines -- below the smallest
# configuration the paper ever tested, and its frequency grid started 10.9% away from the
# circadian 2*pi/96 so no sine began near 24 h. The default is now 65 (k=64, the paper's
# best; nearest frequency 0.6% away). CHECK IT IN STAGE 0: if the loss still plateaus above
# ~3 this variant is not reportable and should come out before Stage 1 spends six tasks.
#   tcn:circular  wall-clock calendar phase (fixed sin/cos basis over [time-of-day,
#                 day-of-week]) added at the input projection. Runs on the TCN because it is
#                 an INPUT-side encoding, not an attention bias -- which is exactly what lets
#                 the reference-frame contrast be read free of the backbone.
#
# tcn:circular is here ONLY to give E1.4 a contrast arm. With tcn:none alone,
# Delta_pi = rho_pi - rho_F0 has nothing to subtract -- tcn:none IS F0 -- so pe_contrast
# returns empty and the "does the temporal reference frame matter?" half of RQ1 has no data
# at all. One arm is not a family, so the Holm correction stays a no-op, but one valid
# paired comparison across 8 seeds is a reportable result where zero arms is not.
VARIANTS=(
  tcn:none
)
# tcn:circular is dropped, not lost. The circular calendar PE was measured to HURT: paired
# over 12 seeds it costs DIS 0.124 (p=0.0098) and worsens the acrophase error in 12 of 12
# seeds (p=0.0024), the mechanism being that it injects a deterministic sinusoid at exactly
# the circadian bin the SFD reads. Spending half a sweep on a condition already shown to be
# harmful buys nothing; the seeds bought instead go into the paired n above.
# BOTH arms are extended, and that is the current cost model:
#   Q23_VARIANTS  -- line 394 lists tcn:circular, so q2/q3 run on the circular arm too.
#   PLAIN_REF     -- tasks/_experiment_common.py:31 contains ("tcn","circular"), so the
#                    circular arm pretrains its own plain-SSL twin. That twin is no longer a
#                    Delta_pi-only control: it is a rung of the ladder in BOTH separability
#                    tables, so dropping it would leave half the runs without one.
# CONSEQUENCE: every task in this array is a worst case -- two full pretrainings (DSSL +
# plain twin), the supervised rungs, and the RQ3 degradation grid. There is no cheap arm to
# hide behind against the 3 h wall clock; see the Stage 0 note in the header before trusting
# it. (An earlier version of this comment claimed the opposite -- that neither list was
# extended. It was stale; both lists have contained tcn:circular since the sweep was cut to
# the TCN pair.)
#
# Dropped from this sweep, not from the codebase -- add back for the full PE study:
#   tcn:time2vec tcn:factorized transformer:sinusoidal transformer:learnable transformer:tpe
#   transformer:rpe transformer:erpe transformer:tupe transformer:convspe
#   transformer:factorized transformer:circular

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
  _time="3:00:00"                         # must match the #SBATCH default above

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
# OUT-OF-RANGE EXIT -- deliberately the FIRST thing after the banner.
#
# The #SBATCH --array directive is read by SLURM before this script runs, so it can never be
# computed from SEEDS/VARIANTS. Instead of editing it every time the sweep changes, it is
# OVER-provisioned (see the header) and the surplus tasks fall out here. That only works if
# they fall out cheaply: this used to sit below the ENVIRONMENT block, so every surplus task
# first built a virtualenv and pip-installed torch -- ~2-3 min of allocated GPU each, times
# however many spare slots the array had. Now it costs one second.
#
# Keep this ABOVE `module purge`. Anything expensive placed before it silently reintroduces
# that waste.
# ============================================================================
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
  echo "ERROR: no array task id. Submit with 'sbatch scripts/run.sh' or 'bash scripts/run.sh'."
  exit 1
fi
if [ "${SLURM_ARRAY_TASK_ID}" -ge "$N_JOBS" ]; then
  echo "NOTE: task ${SLURM_ARRAY_TASK_ID} is beyond the ${N_JOBS} configured pairs "
  echo "      (${N_SEEDS} seeds x ${N_VARIANTS} variants). Nothing to do -- exiting before setup."
  exit 0
fi

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
# Alliance wheelhouse, so it has to come from a wheel.
#
# PORTABILITY: Nibi compute nodes reach PyPI; Narval/Beluga/Graham/Cedar compute nodes do
# NOT. Downloading on PyPI only would make the "Cosinor (paper)" row appear on one cluster
# and vanish on another -- the two runs would then not be comparable, and the difference
# would show up only as a missing table row. So a local wheelhouse is tried FIRST and PyPI
# is the fallback. Populate it once, from a LOGIN node, on any cluster without compute-node
# internet:
#     module load StdEnv/2023 python/3.11
#     pip download --no-deps -d "$PROJECT/wheels" CosinorPy
pip install --no-index seaborn || echo "  [WARN] no seaborn -> CosinorPy import will fail"
if pip install --no-index --find-links "$PROJECT/wheels" CosinorPy 2>/dev/null; then
  echo "[env] CosinorPy from the local wheelhouse"
else
  pip install CosinorPy || echo "  [WARN] CosinorPy install failed -> paper-cosinor view skipped"
fi
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
for _f in train_hrd.py train_hrd_energy.py data_processing/data_preprocessing.py cost.py utils.py \
          model_build.py tasks/rq_paths.py \
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
# Extra flags appended verbatim to train_hrd.py, for the three objective ablations. Empty by
# default, so an ordinary sweep is byte-identical to run 1239199. sbatch exports the caller's
# environment, so:
#     ABLATE="--shift-sigma 0"        sbatch --array=0,1%2 scripts/run.sh   # A
#     ABLATE="--trend-pool mean"      sbatch --array=0,1%2 scripts/run.sh   # B
#     ABLATE="--moco-k 1024"          sbatch --array=0,1%2 scripts/run.sh   # C
# Run each as its OWN sbatch: RUN_ID is the array job id, so the three land in separate
# results folders and cannot overwrite one another or the baseline.
ABLATE="${ABLATE:-}"

# GPU type for the SELF-HEAL resubmission. The #SBATCH line at the top covers the FIRST
# submission only, and `sbatch --gres=...` on the command line overrides that one only --
# healed tasks re-read this file and would take the stale value. Keep the two in step:
#     sbatch --gres=gpu:a100_3g.20gb:1 scripts/run.sh                    # Narval
#     GRES=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1 sbatch --gres=$GRES ...   # Rorqual
GRES="${GRES:-gpu:a100_3g.20gb:1}"
Q_FLAGS="${Q_FLAGS:-}"
Q3_FLAGS="${Q3_FLAGS:-}"

SENSOR_CSV="${SENSOR_CSV:-datasets/HRD_RAW_MinuteLevel.csv}"
RUN_ID="${RUN_ID:-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}}"

ALPHA="0.005"; LR="5e-4"; ITERS="6000"; BATCH_SIZE=64
JITTER_SIGMA="0.1"
# Timestep-masking augmentation. "none" (default) matches upstream CoST, which never applied
# one -- its encoder's mask argument hard-defaulted to 'all_true', so mask_mode was
# unreachable and MASK_KEEP_PROB had NO effect. Set MASK_MODE=binomial to actually enable it;
# MASK_KEEP_PROB is read only in that case.
MASK_MODE="none"; MASK_KEEP_PROB="0.5"
# REVERTED after measurement. Setting this to "binomial" for runs 19891667 and 19937323 was a
# mistake, and it is the single change separating them from the 17 healthy runs before them:
#   pretrain loss  0.079-0.537 (17 runs, mask=none)  ->  0.986-1.080   (12x worse)
#   tail SD        0.0072-0.0220                     ->  0.0496-0.0672 (7x less stable)
#   V^S -> sigma   ~0.89                             ->  0.74
#   DIS sigma-half ~0.00 (oscillating both signs)    ->  -0.12 (consistent)
# and, decisively, the seasonal branch's own cosine-similarity structure:
#   19649817 (none):     V^S spans +1.00 .. -1.00 -- a clean 7-cycle circadian sinusoid
#   19937323 (binomial): V^S spans  0.988 ..  1.000 -- collapsed into a ~6 degree cone
# The CAUTION in models/encoder.py (mask stacks on nan_mask, and the SFD contrasts the rFFT
# amplitude AND phase of the representation, so timestep dropout injects broadband noise into
# exactly the quantity the seasonal objective fits) predicted this, and demanded an A/B on one
# seed first. That A/B was never run. Do not re-enable without it.
# Was "none". With scale deliberately dropped (amplitude IS the biomarker) and no random
# crop (max_train_length == seq_len, so the crop branch in cost.py never fires), the two
# contrastive views differed only by jitter sigma=0.1 and a per-channel DC offset. Run
# 19649817 solved that pretext task outright -- InfoNCE fell to 0.05-0.10 (98%+ drop) on
# every variant -- and the resulting encoder lost to a RANDOM-INIT encoder on Full->sigma
# in 72/72 runs. Timestep masking restores a non-trivial task without touching amplitude,
# and is training-only (encode() always runs mask='none'), so nothing downstream changes.
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
# GLOBEM has 669 labelled participants against HRD's 114, so its held-out set can be far
# larger -- which is the entire reason for running it.
TEST_PER_CLASS="${TEST_PER_CLASS:-18}"; GPU_INDEX=0
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
# Overridable so a DIAGNOSTIC run can drop the whole energy probe: ENERGY_FLAG="" turns it
# off, including its four supervised trainings. Never do this for a reported run -- the energy
# probe IS the emotional-energy half of the project.
ENERGY_FLAG="${ENERGY_FLAG-"--energy-probe --energy-output-dir results_hrd_energy"}"
# The end-to-end supervised rung on the two BINARY energy tasks is the only part of the energy
# probe that TRAINS a network -- twice, on ~6,700 windows. Everything else is a linear probe on
# a frozen encoder. Set ENERGY_SUPERVISED=0 to drop it: that is what buys the task back under
# the 3 h limit, and a <=3h job is eligible for BOTH the 3h and the 12h bucket (~20x less
# queued work ahead of it). The energy ladder then stops at CoST instead of at the supervised
# ceiling -- state that in the paper rather than leaving the rung silently missing.
ENERGY_SUPERVISED="${ENERGY_SUPERVISED:-1}"
if [ "$ENERGY_SUPERVISED" -eq 0 ]; then ENERGY_FLAG="$ENERGY_FLAG --no-energy-supervised"; fi

# ============================================================================
# DISPATCH: task id -> one (seed, variant)
# ============================================================================
# (the task-id sanity checks ran before the ENVIRONMENT block -- see OUT-OF-RANGE EXIT)
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
  # 3:00:00, NOT 8:00:00. An <=3h job is eligible for both the 3h and the 12h bucket; at 8h it
  # is locked into the 12h one, which is what left run 19649817 sitting in ReqNodeNotAvail for
  # two days. The healed tasks are the same work as the rest of the array, so they get the
  # same limit as the #SBATCH directive above.
  sbatch --array="${SLURM_ARRAY_TASK_COUNT}-$(( N_JOBS - 1 ))%12" \
         --time="3:00:00" --job-name="cost_hrd" --gres="$GRES" \
         --export=ALL,RUN_ID="$RUN_ID",GRES="$GRES" "$SELF" \
    && echo "[self-heal] queued tasks ${SLURM_ARRAY_TASK_COUNT}-$(( N_JOBS - 1 )) into $RUN_ID" \
    || echo "[self-heal] FAILED -- submit by hand: RUN_ID=$RUN_ID sbatch --gres=$GRES --array=${SLURM_ARRAY_TASK_COUNT}-$(( N_JOBS - 1 )) scripts/run.sh"
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
# Which tasks keep encoder.pt. The default keeps only the first seed, which is what left
# every readout re-analysis stuck at n=1 -- a corrected probe could then be demonstrated but
# never tested. KEEP_ENC_ALL=1 keeps all of them; at 320 repr-dims that is ~176 MB per task.
KEEP_ENC="${KEEP_ENC_ALL:-0}"
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
  --dataset "$DATASET" --sensor-csv "$SENSOR_CSV" \
  --backbone "$BACKBONE" --pe "$PE" \
  --hidden-dims $HID --repr-dims $REPR --depth $DEPTH \
  --alpha "$ALPHA" --loss-balance fixed --disentangle \
  --jitter-sigma "$JITTER_SIGMA" \
  --mask-mode "$MASK_MODE" --mask-keep-prob "$MASK_KEEP_PROB" \
  --phase-encoding "$PHASE_ENCODING" \
  --cohort "$COHORT" --probe-unit "$PROBE_UNIT" --probe-c "$PROBE_C" --cv-folds "$CV_FOLDS" \
  --lr "$LR" --iters "$ITERS" --batch-size "$BATCH_SIZE" --test-per-class "$TEST_PER_CLASS" \
  $CLOCK_FLAG $ENERGY_FLAG $SAVE_ENC $SEED_FLAGS $ABLATE \
  --output-dir "$OUTPUT_DIR" --run-id "$RUN_ID" --seed "$SEED" --gpu "$GPU_INDEX"

# ============================================================================
# RQ1-RQ3 EXPERIMENTS -- same (seed, variant), same frozen encoder, NO retraining.
# Each script reloads encoder.pt + metrics.json from the variant directory above, so the
# 70 pretrainings of this sweep stay 70. One shared dataset cache per node means the
# 53.5M-row CSV is parsed once for all three instead of three times.
# Non-fatal: a failed experiment must not throw away the training that just finished.
# ============================================================================
# train_hrd.py appends "_plain" to the variant directory when the disentangler is off
# (train_hrd.py: dis_tag). run.sh has to append the SAME tag or the RQ scripts are pointed at a
# directory that was never written -- which is exactly how run 1421825 pretrained 24 encoders
# and then failed every q1/q2/q3 with FileNotFoundError on metrics.json.
DIS_TAG=""
case " $ABLATE " in *" --no-disentangle "*) DIS_TAG="_plain" ;; esac
VARIANT_DIR="$OUTPUT_DIR/$RUN_ID/${BACKBONE}_${PE}${DIS_TAG}${SEED_TAG}"
CACHE_DIR="${SLURM_TMPDIR:-/tmp}/hrd_cache"; mkdir -p "$CACHE_DIR"

# RQ1 runs on every variant -- it IS the PE sweep. RQ2/RQ3 run on a PRE-DECLARED pair only
# (docs/RQ_Minimal_Experiment_Design.md:304), for two distinct reasons:
#   multiplicity -- 12 variants x 3 perturbations x 7 levels is a large family, and every
#                   extra variant widens it without answering a new question;
#   selection    -- choosing the pair AFTER seeing RQ1 would make every RQ2/RQ3 number
#                   conditional on that choice. Fixing the list in advance, here, is what
#                   keeps them unconditional.
# One no-calendar reference and one wall-clock variant: the contrast RQ1 is actually about.
# Every variant in this sweep, so the gate below never fires. The design restricts RQ2/RQ3 to
# a pre-declared subset to keep the multiple-comparison surface small; with a two-variant
# sweep the subset IS the sweep, so the restriction costs nothing and both arms get the full
# RQ1/RQ2/RQ3 treatment. Declared here, before the run -- not chosen after seeing results.
#
# The previous sweep pre-declared transformer:circular and it came back useless: AUC 0.548,
# below the breakdown_valid floor of 0.60 (experiment_q3.py:245), so its c* was undefined by
# construction and half of RQ3 was empty. Both arms here clear that floor on past runs.
Q23_VARIANTS="tcn:none tcn:circular tcn:time2vec"

# Which experiments to run, and extra flags for them. Defaults are the full set, so an
# ordinary sweep is unchanged. A diagnostic that only needs RQ1 and the RQ3 ladder can say
#   QS="1 3" Q_FLAGS="--no-plain-ssl" Q3_FLAGS="--no-supervised --no-cosinor"
QS="${QS:-1 2 3}"
for Q in $QS; do
  if [ "$Q" != "1" ] && ! printf '%s\n' $Q23_VARIANTS | grep -qx "$BACKBONE:$PE"; then
    echo "--- experiment_q$Q | $BACKBONE/$PE not in Q23_VARIANTS -> skipped ---"
    continue
  fi
  echo "--- experiment_q$Q | seed $SEED | $BACKBONE / $PE | $(date +%H:%M:%S) ---"
  _qx="$Q_FLAGS"
  [ "$Q" = "3" ] && _qx="$_qx $Q3_FLAGS"
  python "experiment_q$Q.py" --variant-dir "$VARIANT_DIR" \
      --cache-dir "$CACHE_DIR" --gpu "$GPU_INDEX" $_qx \
    || echo "[q$Q] FAILED (non-fatal) -- $VARIANT_DIR keeps its depression results"
done

# plain_encoder.pt is the non-disentangled SSL control, pretrained once by q1 and reloaded
# by q3. Same size and same reasoning as encoder.pt, so it is kept/dropped together.
if [ "$KEEP_ENC" -eq 0 ]; then rm -f "$VARIANT_DIR/encoder.pt" "$VARIANT_DIR/plain_encoder.pt" "$VARIANT_DIR/plain_encoder.key.json"; fi

echo "=== done $(date) | $VARIANT_DIR | ABLATE='${ABLATE}'"
echo "    QS='${QS}' Q_FLAGS='${Q_FLAGS}' Q3_FLAGS='${Q3_FLAGS}' ENERGY='${ENERGY_FLAG}' ==="
