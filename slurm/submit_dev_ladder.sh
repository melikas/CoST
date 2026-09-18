#!/bin/bash
# DSSL-v2 development ladder. Submit from the repository root:
#   bash slurm/submit_dev_ladder.sh ACCOUNT [RUN_PREFIX]
#
# Runs inside locked protocol fold 0's TRAINING participants only (--dev-cohort). Fold 0's test
# participants are never seen and the locked 3-seed x 5-fold evaluation is untouched, so every
# previous run stays comparable.
#
#   a0_shared        7.45M, current shared encoder          (baseline)
#   a1lo_decomposed  4.03M, decomposition at the v3 width   (capacity control)
#   a1_decomposed    7.51M, decomposition at matched capacity
#
# A0 vs A1 isolates architecture at equal capacity; A1lo vs A1 isolates capacity at equal
# architecture -- so the v3 confound cannot recur.
#
# Development matrix is 2 seeds x 5 folds (array 0-9). Selection must use the LABEL-FREE criteria
# (RQ1 recovery, own-vs-leakage, RQ2). The RQ3 in these summaries is a development probe on
# development participants; the locked RQ3 test is never read here.
set -euo pipefail
account="${1:?usage: bash slurm/submit_dev_ladder.sh ACCOUNT [RUN_PREFIX]}"
prefix="${2:-dev}"
mkdir -p logs
gpu_job=(--account="$account" --gres="${GPU:-gpu:a100_3g.20gb:1}" --time="${TIME:-12:00:00}")
for dataset in hrd globem; do
    for arm in a0_shared a1lo_decomposed a1_decomposed; do
        run="${prefix}_${arm}"
        config="configs/v2/${dataset}_${arm}.json"
        common="ALL,RUN_NAME=$run,DATASET=$dataset,CONFIG=$config,SKIP_REFERENCE=1,DEV_COHORT=1"
        array=$(sbatch --parsable "${gpu_job[@]}" --array=0-9 \
            --export="$common,STAGE=variant" slurm/rq123.sbatch)
        array=${array%%;*}
        summary=$(sbatch --parsable --account="$account" --dependency=afterok:"$array" \
            --export="$common" slurm/summarize.sbatch)
        echo "$dataset $arm: variants ${array} (0-9) -> summary ${summary%%;*}"
    done
done
echo "Score with: python scripts/dev_scoreboard.py ${prefix}_a0_shared ${prefix}_a1lo_decomposed ${prefix}_a1_decomposed"
