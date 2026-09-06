# Running on the cluster

Merged from `CC.md` and `CC_globem.md`, which were ~85% duplicates of each other. Both
cohorts use the same code path, so they use the same commands — only the environment
variables differ (see [HRD vs GLOBEM](#hrd-vs-globem)).

## Where do I type each command?

- **LOCAL** — your PC (PowerShell or Git Bash), in `c:\Users\umroot\Documents\CoST - Rorqual`.
  Used for upload and download.
- **CLUSTER** — after `ssh`. Used to submit and watch jobs.

Every block below is labelled.

## Constants

| | |
|---|---|
| user | `melikas` |
| clusters | `narval` · `rorqual` · `nibi` — all `.alliancecan.ca` |
| project path | `~/projects/def-plago/melikas/projects/rhythmssl_project` |
| account | `def-plago` |

GRES differs per cluster and is set in `scripts/run.sh`:
Narval `gpu:a100_3g.20gb:1`, Rorqual `gpu:nvidia_h100_80gb_hbm3_3g.40gb:1`.

`<jobid>` is the number SLURM prints after `sbatch`. Write it down — results land in
`results_hrd/<jobid>/<backbone>_<pe>_seed<seed>/`.

---

## 1 — Log in

```bash
ssh melikas@rorqual.alliancecan.ca     # or narval / nibi
```

MFA is mandatory: https://ccdb.alliancecan.ca/multi_factor_authentications

---

## 2 — Upload

> **Upload the whole tree. Never list files by hand.**
>
> Three sweeps were lost this way. An `scp` line that named files individually omitted
> `cosinor.py`; runs **66404249, 66440129 and 66465766** — 195 variants, many GPU-hours —
> each trained to completion and silently dropped the "Cosinor (paper)" baseline with
> `ModuleNotFoundError`. Every *other* edited file uploaded fine, so the runs looked
> healthy; the loss showed up only as `n/a` in `summary_models.csv`.
>
> `scripts/run.sh` now refuses to start if a project module is missing. That preflight is
> the fix, and it only works if you keep its manifest current when files move.

```bash
# LOCAL, from the repo root. Whole tree minus junk; names no file by hand.
rsync -avP --exclude='.git' --exclude='results_*' --exclude='__pycache__' \
      ./ melikas@rorqual.alliancecan.ca:~/projects/def-plago/melikas/projects/rhythmssl_project/
```

Add `--exclude='datasets'` to skip the ~3.4 GB HRD CSV once it is already up there.
The first upload takes several minutes.

`-a` keeps structure, `-v` shows progress, `-P` resumes partial files.

> Historical note: `CC_globem.md` pointed at `bash scripts/upload.sh`. **That script does
> not exist in this repository** and appears never to have been committed. Use the `rsync`
> line above.

### Verify it landed

```bash
# CLUSTER
cd ~/projects/def-plago/melikas/projects/rhythmssl_project
ls -lh datasets/HRD_RAW_MinuteLevel.csv
```

### CosinorPy

Needed for the paper-cosinor baseline, and not on the cluster by default:

```bash
# CLUSTER
module purge && module load StdEnv/2023 python/3.11
pip download --no-deps CosinorPy -d wheels
ls wheels/                              # expect CosinorPy-3.1-py3-none-any.whl
python -c "import CosinorPy; print('OK')"
```

---

## 3 — Submit

```bash
# CLUSTER
cd ~/projects/def-plago/melikas/projects/rhythmssl_project

sbatch scripts/run.sh                          # the sweep
sbatch --time=3:00:00 --array=0,1%2 scripts/run.sh   # stage 0: one seed of each variant

sacct -j <jobid> --format=JobID%20,State,Elapsed     # size the real sweep from stage 0
```

Run stage 0 first. A full task needs hours, and sizing the array from a guess is how jobs
get cut off by the wall clock.

### HRD vs GLOBEM

Same scripts; GLOBEM is selected by environment:

```bash
# CLUSTER — GLOBEM. ENERGY_FLAG must be EMPTY: emotional energy is HRD-only.
DATASET=globem SENSOR_CSV=datasets/GLOBEM_REDUCED.csv \
OUTPUT_DIR=results_globem ENERGY_FLAG= TEST_PER_CLASS=60 \
    sbatch --array=0-23%12 scripts/run.sh
```

`KEEP_ENC_ALL=1` keeps every task's encoder, which is what lets a readout question be
re-tested later without retraining. At 320 repr-dims that is ~176 MB per task.

---

## 4 — Watch

Stdout and errors go to one file; there is no separate `.err`.

```bash
# CLUSTER
squeue -u melikas                       # ST: PD = pending, R = running
tail -f logs/cost_hrd-<jobid>_<taskid>.out
less  logs/cost_hrd-<jobid>_<taskid>.out
scontrol show job <jobid>
```

A Python traceback, if any, is at the bottom of that same `.out`. When the job leaves
`squeue` it has finished — check the log, not the queue.

```bash
scancel <jobid>          # cancel one
scancel -u melikas       # cancel everything
```

---

## 5 — Download

```bash
# LOCAL
rsync -avP \
  "melikas@rorqual.alliancecan.ca:~/projects/def-plago/melikas/projects/rhythmssl_project/results_hrd/<jobid>" \
  ./results_hrd/
```

`scp -r` works too if `rsync` is unavailable:

```bash
# LOCAL
scp -r "melikas@nibi.alliancecan.ca:~/.../rhythmssl_project/results_hrd/<jobid>" \
       "c:\Users\umroot\Documents\CoST - Rorqual\results_hrd\"
```

---

## 6 — Free space on the cluster

**`rm -rf` is permanent. Read the path twice.**

The bulk is always checkpoints, not results: `encoder.pt` is ~150-235 MB each, while all
the json/csv/md a run produces come to a few MB. Delete the weights and keep the numbers:

```bash
# CLUSTER
cd ~/projects/def-plago/melikas/projects/rhythmssl_project
du -sh results_hrd/*
diskusage_report

find results_hrd/<jobid> -name "*.pt" -delete     # option A: weights only (preferred)
rm -rf results_hrd/<jobid>                        # option B: the whole job, irreversible
rm -f  logs/cost_hrd-<jobid>*.out
```

> این همان کاری است که به‌صورت محلی هم انجام شد: حجمِ اصلی از فایل‌های `.pt` است
> (هر کدام ~۱۵۰MB)، و خودِ نتایج (json/csv/metrics/png) فقط چند مگابایت‌اند. با پاک‌کردنِ
> وزن‌ها بیشترِ فضا آزاد می‌شود و همهٔ نتایجِ تحلیل می‌مانند. برای تأیید: `diskusage_report`

---

## Cheat sheet

| I want to… | Where | Command |
|---|---|---|
| Log in | LOCAL | `ssh melikas@rorqual.alliancecan.ca` |
| Upload | LOCAL | `rsync -avP --exclude='.git' --exclude='results_*' ./ …:…/rhythmssl_project/` |
| Submit | CLUSTER | `sbatch scripts/run.sh` |
| Watch | CLUSTER | `tail -f logs/cost_hrd-<jobid>_<taskid>.out` |
| Queue | CLUSTER | `squeue -u melikas` |
| Cancel | CLUSTER | `scancel <jobid>` |
| Download | LOCAL | `rsync -avP …:…/results_hrd/<jobid> ./results_hrd/` |
| Free space | CLUSTER | `find results_hrd/<jobid> -name "*.pt" -delete` |

## Notes

- One A100 (or an H100 MIG slice) is enough; 64 GB RAM covers the ~3.4 GB CSV plus the model.
- Change hyperparameters by editing the `python train_hrd.py …` line in `scripts/run.sh`.
- The cross-variant summary job was removed in the cleanup — it ran
  `scripts/collect_results.py`, which is deleted. Each task still writes its own results
  under its variant directory; the sweep-level `summary.csv` is folded into `eval.py`.

## Recent job ids

Kept for traceability against `archive_logs.txt` section 1.

| cluster | job id | note |
|---|---|---|
| nibi | 19649817 | transformer/PE sweep — out of scope, code removed |
| nibi | 19606825 | |
| nibi | 19422314 | smoke test |
| — | 20093940 | **the sweep kept whole locally, with its encoders** |
| narval | 66404249 · 66440129 · 66465766 | lost the cosinor baseline to the upload bug above |
