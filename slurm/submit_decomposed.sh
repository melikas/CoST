#!/bin/bash
# Development run for the pre-encoder decomposition. Submit from the repository root:
#   bash slurm/submit_decomposed.sh ACCOUNT [RUN_NAME]
#
# The architecture changed, so the encoders must be retrained -- --reuse-encoders is NOT valid
# here (it is only sound when configurations differ solely in readout settings).
#
# --skip-reference: this is a DEVELOPMENT comparison against narval_v2 and against the matched
# untrained control, and none of the eight pre-committed criteria involves the CoST reference.
# Skipping it halves the GPU cost. The reference returns for the final protocol run, if the
# architecture is adopted.
#
# RQ3 is computed by the pipeline but MUST NOT be read during development
# (docs/DECOMPOSITION_PRECOMMIT.md).
set -euo pipefail
account="${1:?usage: bash slurm/submit_decomposed.sh ACCOUNT [RUN_NAME]}"
run="${2:-narval_v3_decomposed}"
mkdir -p logs
gpu_job=(--account="$account" --gres="${GPU:-gpu:a100_3g.20gb:1}" --time="${TIME:-12:00:00}")
for dataset in hrd globem; do
    config="configs/arch/${dataset}_decomposed.json"
    common="ALL,RUN_NAME=$run,DATASET=$dataset,CONFIG=$config,SKIP_REFERENCE=1"
    array=$(sbatch --parsable "${gpu_job[@]}" --export="$common,STAGE=variant" slurm/rq123.sbatch)
    array=${array%%;*}
    summary=$(sbatch --parsable --account="$account" --dependency=afterok:"$array" \
        --export="$common" slurm/summarize.sbatch)
    echo "$dataset: variant array ${array} (tasks 0-14) -> summary ${summary%%;*}"
done
echo "Compare against results/<dataset>/narval_v2/tcn_none/SUMMARY.md"
