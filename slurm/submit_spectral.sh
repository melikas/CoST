#!/bin/bash
# Re-read the finished v3 encoders with the amplitude/phase readout. From the repository root:
#   bash slurm/submit_spectral.sh ACCOUNT RUN_NAME CONFIG [FOLDS]
#   bash slurm/submit_spectral.sh def-plago narval_v3 configs/hrd_v3_spectral.json
#   DATASET=globem bash slurm/submit_spectral.sh def-plago loyo_v3 configs/globem_loyo_v3_spectral.json 4
# The run name is the SAME as the finished v3 run: the widths chosen there, its trained DSSL
# encoders and its CoST reference are reused, and the new variant is written beside them as
# tcn_none_spec_selected. DSSL and CoST are re-read, not retrained, because the objective never
# uses the readout. Only the supervised control is retrained: its head reads the representation.
set -euo pipefail
account="${1:?usage: bash slurm/submit_spectral.sh ACCOUNT RUN_NAME CONFIG [FOLDS]}"
run="${2:?RUN_NAME}"
config="${3:?CONFIG}"
folds="${4:-5}"
dataset="${DATASET:-hrd}"
[ -f "$config" ] || { echo "missing $config" >&2; exit 2; }
sel="results/$dataset/$run/tcn_none_selection.csv"
[ -f "$sel" ] || { echo "missing $sel: run slurm/submit_v3.sh for this run first" >&2; exit 2; }
mkdir -p logs
tasks="0-$((3 * folds - 1))"
gpu=(--account="$account" --gres="${GPU:-gpu:a100_3g.20gb:1}" --time="${TIME:-03:00:00}" --array="$tasks")
cpu=(--account="$account")
env="ALL,RUN_NAME=$run,DATASET=$dataset,FOLDS=$folds,CONFIG=$config,OUTPUT_DIMS=selected"
submit() { local id; id=$(sbatch --parsable "$@"); echo "${id%%;*}"; }

sup=$(submit "${gpu[@]}" --export="$env,STAGE=supervised" slurm/rq123.sbatch)
var=$(submit "${gpu[@]}" --dependency=afterok:"$sup" --export="$env,STAGE=variant" slurm/rq123.sbatch)
sum=$(submit "${cpu[@]}" --dependency=afterok:"$var" --export="$env" slurm/summarize.sbatch)
exp=$(submit "${cpu[@]}" --array="$tasks" --dependency=afterok:"$var" --export="$env" slurm/export.sbatch)
echo "$dataset: supervised $sup -> variant $var (summary $sum, dynamics export $exp)"
echo "Results: results/$dataset/$run/tcn_none_spec_selected"
