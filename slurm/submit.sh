#!/bin/bash
# Submit the matrix from the repository root on a login node:
#   bash slurm/submit.sh ACCOUNT [RUN_NAME]
# Per dataset: the CoST reference array and the supervised-control array (in parallel), then the
# variant array (starts when every task of both succeeded), then a CPU summary (starts when every
# variant task succeeded).
#   TIME=12:00:00 bash slurm/submit.sh ...              longer limit per task (default 3 h)
#   GPU=gpu:a100:1 bash slurm/submit.sh ...             whole A100s instead of 3g.20gb slices
#   BACKBONE=transformer ENCODING=sinusoidal SKIP_REFERENCE=1 bash slurm/submit.sh ACCOUNT RUN
#                                                       another variant, reusing the reference
#   DATASETS=globem CONFIG=configs/globem_loyo.json FOLDS=4 bash slurm/submit.sh ACCOUNT loyo_v1
#                                                       GLOBEM leave-one-year-out (12 tasks)
set -euo pipefail
account="${1:?usage: bash slurm/submit.sh ACCOUNT [RUN_NAME]}"
run="${2:-narval_v2}"
mkdir -p logs
gpu_job=(--account="$account" --gres="${GPU:-gpu:a100_3g.20gb:1}" --time="${TIME:-03:00:00}")
folds=${FOLDS:-5}
tasks="0-$((3 * folds - 1))"
export_vars="ALL,RUN_NAME=$run,BACKBONE=${BACKBONE:-},ENCODING=${ENCODING:-},CONFIG=${CONFIG:-},FOLDS=$folds"
for dataset in ${DATASETS:-hrd globem}; do
    dependency=()
    if [ -z "${SKIP_REFERENCE:-}" ]; then
        reference=$(sbatch --parsable "${gpu_job[@]}" --array="$tasks" \
            --export="$export_vars,DATASET=$dataset,STAGE=reference" slurm/rq123.sbatch)
        reference=${reference%%;*}
        supervised=$(sbatch --parsable "${gpu_job[@]}" --array="$tasks" \
            --export="$export_vars,DATASET=$dataset,STAGE=supervised" slurm/rq123.sbatch)
        supervised=${supervised%%;*}
        dependency=(--dependency=afterok:"$reference":"$supervised")
        echo "$dataset: CoST reference array $reference, supervised array $supervised (tasks $tasks)"
    fi
    array=$(sbatch --parsable "${gpu_job[@]}" --array="$tasks" "${dependency[@]}" \
        --export="$export_vars,DATASET=$dataset,STAGE=variant" slurm/rq123.sbatch)
    array=${array%%;*}
    summary=$(sbatch --parsable --account="$account" --dependency=afterok:"$array" \
        --export="$export_vars,DATASET=$dataset" slurm/summarize.sbatch)
    echo "$dataset: variant array ${array} (tasks $tasks) -> summary job ${summary%%;*}"
done
echo "Results: results/<dataset>/$run/<backbone>_<encoding>/ ; combined summary: results/SUMMARY_$run.md"
