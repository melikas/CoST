#!/bin/bash
# Readout-only comparison on the narval_v2 encoders. Submit from the repository root:
#   bash slurm/submit_readout.sh ACCOUNT [SOURCE_RUN]
#
# Reads the SAME trained encoders out under two further readouts and evaluates the full protocol
# for each, so the only thing that differs between the three arms is the readout:
#   narval_v2            readout_norm="none"      (already run; amplitude and phase unnormalised)
#   narval_v2_timestep   readout_norm="timestep"  (both normalised -- v1's readout, v2's weights)
#   narval_v2_split      readout_norm="split"     (amplitude unnormalised, phase normalised)
#
# Nothing is retrained: the readout never enters training (seasonal_loss normalises independently
# of readout_norm, and _log_readout_amp is inert at w_ac=0), so all three arms share identical
# weights. narval_v1 is NOT this comparison -- it also had w_eq=1.0 and eps=1e-6.
#
# Stages per dataset x arm mirror slurm/submit.sh: reference, then variants, then the CPU summary.
set -euo pipefail
account="${1:?usage: bash slurm/submit_readout.sh ACCOUNT [SOURCE_RUN]}"
source_run="${2:-narval_v2}"
mkdir -p logs
gpu_job=(--account="$account" --gres="${GPU:-gpu:a100_3g.20gb:1}" --time="${TIME:-03:00:00}")
for dataset in hrd globem; do
    for arm in timestep split; do
        run="${source_run}_${arm}"
        config="configs/readout/${dataset}_${arm}.json"
        common="ALL,RUN_NAME=$run,DATASET=$dataset,CONFIG=$config,REUSE=$source_run"
        reference=$(sbatch --parsable "${gpu_job[@]}" --export="$common,STAGE=reference" slurm/rq123.sbatch)
        reference=${reference%%;*}
        array=$(sbatch --parsable "${gpu_job[@]}" --dependency=afterok:"$reference" \
            --export="$common,STAGE=variant" slurm/rq123.sbatch)
        array=${array%%;*}
        summary=$(sbatch --parsable --account="$account" --dependency=afterok:"$array" \
            --export="$common" slurm/summarize.sbatch)
        echo "$dataset $arm: reference $reference -> variants $array -> summary ${summary%%;*}"
    done
done
echo "Compare: results/<dataset>/{${source_run},${source_run}_timestep,${source_run}_split}/tcn_none/SUMMARY.md"
