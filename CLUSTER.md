# Narval: every command, in order

Commands marked LOCAL run in Git Bash at this repository's root; NARVAL commands run on a
Narval login node (`ssh melikas@narval.alliancecan.ca`, password + MFA). The code goes into a
fresh directory, `~/projects/def-plago/melikas/projects/rhythmssl_rescue`, so nothing in the
older `rhythmssl_project` directory is touched. Design and outputs: [docs/RUNNING.md](docs/RUNNING.md).

## 1. Package the committed code and the two caches (LOCAL)

```bash
git status --short                   # must print nothing: runs refuse uncommitted code
python scripts/stamp_version.py      # prints: CODE_VERSION: <commit> (clean)
git archive --format=tar.gz -o rescue_upload.tgz \
    --prefix=rhythmssl_rescue/datasets/cache/ \
    --add-file=datasets/cache/hrd_rescue_v1.npz --add-file=datasets/cache/globem_rescue_v1.npz \
    --prefix=rhythmssl_rescue/ --add-file=CODE_VERSION HEAD
scp rescue_upload.tgz melikas@narval.alliancecan.ca:~/projects/def-plago/melikas/projects/
```

The archive holds exactly the committed files, the commit stamp and the caches (~40 MB); no raw
CSV is needed on the cluster.

## 2. Unpack and build the environment once (NARVAL)

```bash
cd ~/projects/def-plago/melikas/projects
tar xzf rescue_upload.tgz            # creates rhythmssl_rescue/
cd rhythmssl_rescue
cat CODE_VERSION                     # the commit you packaged, "git_dirty": false
bash slurm/setup_env.sh              # login node only: needs the internet for CosinorPy
```

If `setup_env.sh` stops because `~/venvs/dssl` already exists, check that environment instead:

```bash
source slurm/env.sh && python -c "import torch, sklearn, statsmodels, seaborn, tasks.yan_cosinor; from CosinorPy import cosinor; print('ok', torch.__version__)"
```

## 3. GPU smoke test (NARVAL, ~15–30 min)

```bash
mkdir -p logs
sbatch --account=def-plago slurm/smoke.sbatch        # prints: Submitted batch job <id>
```

When `squeue -u $USER` no longer lists it:

```bash
sacct -j <id> --format=JobID%18,State,Elapsed        # State must be COMPLETED
grep -iE 'traceback|error' logs/dssl-smoke_<id>.err  # must print nothing
ls results/SUMMARY_narval_v2_smoke_cuda.md           # the whole output chain ran
tail -n 5 logs/dssl-smoke_<id>.out                   # measured time per update
```

The last lines read like `hrd cost_reference: 1.234 s/update -> 2.06 h for 6000 updates
(training only)`. Each task also needs time for evaluation, so if the largest projection
(normally `hrd cost_reference`) is above 2.5 h, submit with `TIME=12:00:00` in step 4.

## 4. Submit the study (NARVAL)

```bash
bash slurm/submit.sh def-plago
```

This queues, per dataset, 15 reference tasks, then 15 variant tasks once every reference task
succeeded, then a CPU summary once every variant task succeeded (3 seeds × 5 folds). Each task
uses one `a100_3g.20gb` GPU slice, 4 CPUs, 32 GB and at most 3 h. Options:

```bash
TIME=12:00:00 bash slurm/submit.sh def-plago         # longer limit (12 h partition, slower queue)
GPU=gpu:a100:1 bash slurm/submit.sh def-plago        # whole A100s
bash slurm/submit.sh def-plago narval_v2             # a new run name; required after any code change
```

## 5. Monitor (NARVAL)

```bash
squeue -u $USER                                        # pending (PD) / running (R)
sacct -j <array_id> --format=JobID%18,State,Elapsed,MaxRSS
tail -n 20 logs/dssl-rq123_<array_id>_<task>.out       # training progress of one task
scancel <jobid>                                        # cancel one job or array
scancel -u melikas                                     # cancel ALL your jobs
```

Task `t` is seed `t // 5 + 1`, fold `t % 5`.

## 6. If a task fails or times out (NARVAL)

Resubmitting a task resumes it from its last checkpoint (every 100 updates); finished tasks
are skipped. The jobs waiting on the failed array can never start, so cancel them and chain new
ones (`hrd` shown; use `globem` for the other dataset):

```bash
scancel <waiting_variant_array_id> <waiting_summary_id>
# a failed REFERENCE task, then the variant array and summary behind it:
ref=$(sbatch --parsable --account=def-plago --array=<failed_tasks> --export=ALL,DATASET=hrd,RUN_NAME=narval_v2,STAGE=reference slurm/rq123.sbatch)
var=$(sbatch --parsable --account=def-plago --dependency=afterok:$ref --export=ALL,DATASET=hrd,RUN_NAME=narval_v2,STAGE=variant slurm/rq123.sbatch)
sbatch --account=def-plago --dependency=afterok:$var --export=ALL,DATASET=hrd,RUN_NAME=narval_v2 slurm/summarize.sbatch
# a failed VARIANT task, then the summary behind it:
var=$(sbatch --parsable --account=def-plago --array=<failed_tasks> --export=ALL,DATASET=hrd,RUN_NAME=narval_v2,STAGE=variant slurm/rq123.sbatch)
sbatch --account=def-plago --dependency=afterok:$var --export=ALL,DATASET=hrd,RUN_NAME=narval_v2 slurm/summarize.sbatch
```

Add `--time=12:00:00` to the `sbatch` lines if the task timed out. Do not run the same task
twice at once, and do not change the code within one run name.

## 7. Bring the results home

NARVAL, once both summary jobs are COMPLETED (training checkpoints `*.pt` stay behind):

```bash
tar czf ~/rescue_results_narval_v2.tgz --exclude='*.pt' \
    results/SUMMARY_narval_v2.md results/hrd/narval_v2 results/globem/narval_v2
```

LOCAL, at this repository's root:

```bash
scp melikas@narval.alliancecan.ca:~/rescue_results_narval_v2.tgz .
tar xzf rescue_results_narval_v2.tgz                  # unpacks into results/
```

Open `results/SUMMARY_narval_v2.md` first; per dataset,
`results/<dataset>/narval_v2/tcn_none/REPORT.html` has every table and figure.
