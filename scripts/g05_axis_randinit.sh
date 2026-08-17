#!/bin/bash
# =====================================================================
# GATE G0.5 -- E1.3 (chronobiological markers) : CoST vs RANDOM-INIT
#
# Answers the one question that decides whether RQ1 has ANY positive claim:
# does PRETRAINING beat the untrained architecture on the markers, or does a
# wide frozen random conv map already recover IS and acrophase on its own?
#
# Costs NO retraining. It reloads the two surviving encoders from run 19937323
# (run.sh keeps encoder.pt for SEEDS[0]=42 only) and reuses the cached cosinor
# fit, so the whole gate is one extra encode per variant.
#
#   sbatch scripts/g05_axis_randinit.sh
#   tail -f logs/g05_axis-<jobid>.out
# =====================================================================
#SBATCH --account=def-plago
#SBATCH --job-name=g05_axis
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1   # same MIG slice run.sh uses: short queue
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-user=melikaseyedi@gmail.com
#SBATCH --mail-type=END,FAIL

set -e

PROJECT=~/projects/def-plago/melikas/projects/rhythmssl_project
cd "$PROJECT" || { echo "Cannot cd to $PROJECT"; exit 1; }
mkdir -p logs

RUN=19937323
VARIANTS="tcn_none_seed42 tcn_time2vec_seed42"

# --- preflight: fail in seconds, not after the environment is built ----------
grep -q "axis_probe_random_init" experiment_q1.py \
  || { echo "FATAL: experiment_q1.py on the cluster is the OLD copy (no random-init arm). Re-upload it."; exit 1; }
for V in $VARIANTS; do
  for F in encoder.pt metrics.json rq1/rq1.json rq1/cosinor_cache.npz; do
    [ -f "results_hrd/$RUN/$V/$F" ] \
      || { echo "FATAL: missing results_hrd/$RUN/$V/$F"; exit 1; }
  done
done
echo "[preflight] patched script + both encoders + cosinor caches present"

# --- environment (identical to scripts/run.sh) -------------------------------
module purge
module load StdEnv/2023 python/3.11
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch numpy pandas scikit-learn einops matplotlib umap-learn
pip install --no-index seaborn || echo "  [WARN] no seaborn"
pip install CosinorPy || echo "  [WARN] CosinorPy install failed"
# E1.3 IS the gate here, and it needs CosinorPy. Abort rather than measure nothing.
python -c "from CosinorPy import cosinor" 2>/dev/null \
  && echo "[env] CosinorPy OK" \
  || { echo "FATAL: CosinorPy unimportable -- G0.5 cannot be evaluated."; exit 1; }

export MPLBACKEND=Agg
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

CACHE="$SLURM_TMPDIR/hrd_cache"; mkdir -p "$CACHE"

# --- run ---------------------------------------------------------------------
# --skip-controls gates the E1.2/E1.5 block INCLUDING the plain-SSL twin, so
# nothing pretrains. It also means rq1.json is rewritten without `controls`,
# hence the backup beside it (kept in the variant dir, not $SLURM_TMPDIR, so it
# survives the job).
for V in $VARIANTS; do
  D="results_hrd/$RUN/$V"
  cp "$D/rq1/rq1.json" "$D/rq1/rq1.pre_g05.json.bak"
  echo "=============== $V ==============="
  python experiment_q1.py --variant-dir "$D" --cache-dir "$CACHE" \
         --skip-controls --gpu 0
done

# --- verdict -----------------------------------------------------------------
python - "$RUN" $VARIANTS <<'EOF'
import json, sys
run, variants = sys.argv[1], sys.argv[2:]
KEY = ["interdaily stability [is_asleep]", "interdaily stability [HR]",
       "acrophase [HR]", "acrophase [is_asleep]"]
for V in variants:
    r = json.load(open(f"results_hrd/{run}/{V}/rq1/rq1.json"))
    h, a, b = r["headline"], r["axis_probe"], r["axis_probe_random_init"]
    frac, med = h["frac_markers_beating_random_init"], h["median_gain_over_random_init"]
    d = {k: a[k]["value"] - b[k]["value"] for k in KEY if k in a and k in b}
    print(f"\n================ {V} ================")
    print(f"  frac markers beating random-init : {frac}  (of {h['n_markers_vs_random_init']})")
    print(f"  median gain over random-init     : {med:+.4f}")
    for k, v in d.items():
        print(f"  {k:34s} CoST {a[k]['value']:+.3f} | rand {b[k]['value']:+.3f} | D {v:+.3f}")
    dIS, dAC = d.get("interdaily stability [is_asleep]"), d.get("acrophase [HR]")
    # Thresholds are the 6-seed SDs of run 19937323: IS[is_asleep] +-0.007,
    # acrophase[HR] +-0.055. PASS margins are ~7 sigma and ~2 sigma at n=1.
    PASS = (frac is not None and frac >= 0.625 and med > 0
            and dIS is not None and dIS >= 0.05 and dAC is not None and dAC >= 0.10)
    FAIL = sum([frac is not None and frac <= 0.50,
                med is not None and med <= 0,
                dIS is not None and dIS <= 0.02,
                dAC is not None and dAC <= 0.0]) >= 2
    print(f"  >>> VERDICT: {'PASS' if PASS else 'FAIL' if FAIL else 'AMBIGUOUS -- escalate to 8 seeds'}")
EOF

echo "=== G0.5 done $(date) ==="
