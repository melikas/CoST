#!/bin/bash
# The v3 matrix: CoST objective with harmonic-centred bands, width selected per fold. From the repo root:
#   bash slurm/submit_v3.sh ACCOUNT RUN_NAME CONFIG [FOLDS]
#   bash slurm/submit_v3.sh def-plago narval_v3 configs/hrd_v3.json                       # HRD, 5 folds
#   DATASET=globem bash slurm/submit_v3.sh def-plago loyo_v3 configs/globem_loyo_v3.json 4  # GLOBEM, 4 years
# Chain for one dataset (each arrow is afterok):
#   CoST reference ------------------------------------------------+
#   width sweep: one variant array per width in DIMS (no supervised) +-> select width (CPU)
#   select --> supervised control at the selected width --> main variant at the selected width
#   select --> ablations at the selected width: no amplitude scaling (<config>_noscale.json) and
#              one full-spectrum band (<config>_fullband.json)
#   summaries: every sweep width, the selected main variant, each ablation (CPU)
#   main --> dynamics export of the selected variant (CPU array, tasks/dynamics.py), which
#            scripts/representation_figures.py turns into the representation figures
#   TIME=12:00:00 / GPU=gpu:a100:1 as in slurm/submit.sh.
set -euo pipefail
account="${1:?usage: bash slurm/submit_v3.sh ACCOUNT RUN_NAME CONFIG [FOLDS]}"
run="${2:?RUN_NAME}"
config="${3:?CONFIG}"
folds="${4:-5}"
dataset="${DATASET:-hrd}"
dims="${DIMS:-32 64 128 320}"
ablations=("${config%.json}_noscale.json" "${config%.json}_fullband.json")
for f in "$config" "${ablations[@]}"; do
    [ -f "$f" ] || { echo "missing $f" >&2; exit 2; }
done
mkdir -p logs
tasks="0-$((3 * folds - 1))"
gpu=(--account="$account" --gres="${GPU:-gpu:a100_3g.20gb:1}" --time="${TIME:-03:00:00}" --array="$tasks")
cpu=(--account="$account")
env="ALL,RUN_NAME=$run,DATASET=$dataset,FOLDS=$folds"
submit() { local id; id=$(sbatch --parsable "$@"); echo "${id%%;*}"; }

ref=$(submit "${gpu[@]}" --export="$env,CONFIG=$config,STAGE=reference" slurm/rq123.sbatch)
echo "$dataset: CoST reference $ref"
sweep=()
for d in $dims; do
    id=$(submit "${gpu[@]}" --dependency=afterok:"$ref" \
        --export="$env,CONFIG=$config,STAGE=variant,OUTPUT_DIMS=$d,NO_SUPERVISED=1" slurm/rq123.sbatch)
    sweep+=("$id")
    submit "${cpu[@]}" --dependency=afterok:"$id" \
        --export="$env,CONFIG=$config,OUTPUT_DIMS=$d" slurm/summarize.sbatch >/dev/null
    echo "$dataset: width $d sweep $id"
done
select=$(submit "${cpu[@]}" --dependency=afterok:"$(IFS=:; echo "${sweep[*]}")" \
    --export="$env,CONFIG=$config,SELECT=1" slurm/summarize.sbatch)
sup=$(submit "${gpu[@]}" --dependency=afterok:"$select" \
    --export="$env,CONFIG=$config,STAGE=supervised,OUTPUT_DIMS=selected" slurm/rq123.sbatch)
main=$(submit "${gpu[@]}" --dependency=afterok:"$sup" \
    --export="$env,CONFIG=$config,STAGE=variant,OUTPUT_DIMS=selected" slurm/rq123.sbatch)
s=$(submit "${cpu[@]}" --dependency=afterok:"$main" \
    --export="$env,CONFIG=$config,OUTPUT_DIMS=selected" slurm/summarize.sbatch)
exp=$(submit "${cpu[@]}" --array="$tasks" --dependency=afterok:"$main" \
    --export="$env,CONFIG=$config,OUTPUT_DIMS=selected" slurm/export.sbatch)
echo "$dataset: select $select -> supervised $sup -> main $main (summary $s, dynamics export $exp)"
for ab in "${ablations[@]}"; do
    id=$(submit "${gpu[@]}" --dependency=afterok:"$select" \
        --export="$env,CONFIG=$ab,STAGE=variant,OUTPUT_DIMS=selected,NO_SUPERVISED=1" slurm/rq123.sbatch)
    s=$(submit "${cpu[@]}" --dependency=afterok:"$id" \
        --export="$env,CONFIG=$ab,OUTPUT_DIMS=selected" slurm/summarize.sbatch)
    echo "$dataset: ablation $ab $id (summary $s)"
done
echo "Results: results/$dataset/$run/{tcn_none_d*,tcn_none_selected,tcn_none_noscale_selected,tcn_none_fullband_selected}"
