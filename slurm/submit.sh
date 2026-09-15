#!/bin/bash
# Submit the matrix from the repository root on a login node:
#   bash slurm/submit.sh ACCOUNT [RUN_NAME]
# Per dataset: the CoST reference array, then the variant array (starts when every reference
# task succeeded), then a CPU summary (starts when every variant task succeeded).
#   GPU=gpu:a100_3g.20gb:1 bash slurm/submit.sh ...     MIG slices instead of whole A100s
#   BACKBONE=transformer ENCODING=sinusoidal SKIP_REFERENCE=1 bash slurm/submit.sh ACCOUNT RUN
#                                                        another variant, reusing the reference
set -euo pipefail
account="${1:?usage: bash slurm/submit.sh ACCOUNT [RUN_NAME]}"
run="${2:-narval_v1}"
mkdir -p logs
export_vars="ALL,RUN_NAME=$run,BACKBONE=${BACKBONE:-},ENCODING=${ENCODING:-}"
for dataset in hrd globem; do
    dependency=()
    if [ -z "${SKIP_REFERENCE:-}" ]; then
        reference=$(sbatch --parsable --account="$account" --gres="${GPU:-gpu:1}" \
            --export="$export_vars,DATASET=$dataset,STAGE=reference" slurm/rq123.sbatch)
        reference=${reference%%;*}
        dependency=(--dependency=afterok:"$reference")
        echo "$dataset: reference array $reference (tasks 0-14)"
    fi
    array=$(sbatch --parsable --account="$account" --gres="${GPU:-gpu:1}" "${dependency[@]}" \
        --export="$export_vars,DATASET=$dataset,STAGE=variant" slurm/rq123.sbatch)
    array=${array%%;*}
    summary=$(sbatch --parsable --account="$account" --dependency=afterok:"$array" \
        --export="$export_vars,DATASET=$dataset" slurm/summarize.sbatch)
    echo "$dataset: variant array ${array} (tasks 0-14) -> summary job ${summary%%;*}"
done
echo "Results: results/<dataset>/$run/<backbone>_<encoding>/ ; combined summary: results/SUMMARY_$run.md"
