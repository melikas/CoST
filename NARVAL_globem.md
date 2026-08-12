

## Where do I type each command?

There are **two places**:

- **LOCAL** = your own PC (Windows PowerShell or Git Bash), in the folder
  `c:\Users\umroot\Documents\CoST`. Used for upload/download.
- **NARVAL** = the cluster, after you log in with `ssh`. Used to submit and watch jobs.

Each command block below is labelled `# LOCAL` or `# NARVAL`.

## Placeholders (replace these with real values)

| Placeholder | Means | Example |
|-------------|-------|---------|
| `<jobid>` | the number SLURM gives you after `sbatch` | `62642322` |
| `<backbone>_<pe>_seed<seed>` | one experiment variant folder | `transformer_tupe_seed42` |

Useful constants for copy-paste:

- Login: `melikas@narval.alliancecan.ca`
- Project on Narval: `~/projects/def-plago/melikas/projects/rhythmssl_project`

---

## Step 0 — Log in to Narval

```bash
# NARVAL: open a session on the cluster (asks for password + MFA code)
ssh melikas@narval.alliancecan.ca
```
scp -r ./cost.py ./train_hrd.py ./data_preprocessing.py ./hrd_rhythm.py ./cosinor.py ./decomposition_recovery.py ./globem_preprocessing.py ./utils.py ./models ./scripts ./requirements.txt melikas@narval.alliancecan.ca:~/projects/def-plago/melikas/projects/rhythmssl_project_GLOBEM/

Multifactor authentication (MFA) is mandatory. If you have not set it up:
https://ccdb.alliancecan.ca/multi_factor_authentications

Type `exit` any time to close the Narval session and return to your PC.

---

#
```

# LOCAL
scp -r ./cost.py ./train_hrd.py ./data_preprocessing.py ./hrd_rhythm.py ./cosinor.py ./globem_preprocessing.py ./datasets ./decomposition_recovery.py ./utils.py ./models ./scripts ./requirements.txt melikas@rorqual.alliancecan.ca:~/projects/def-plago/melikas/projects/rhythmssl_project/


### >>> USE THIS, not the scp lines above <<<

```bash
# LOCAL, from the repo root. Sends the WHOLE tree minus junk, names no file by hand,
# then verifies on the cluster that every required module actually landed.
bash scripts/upload.sh narval            # code + datasets/
bash scripts/upload.sh narval --code      # code only, skips the ~4 GB datasets/
```

> **This is where three sweeps were lost.** The `scp` line that used to sit here —
> targeting `rhythmssl_project_GLOBEM/` — listed files by name and did **not** include
> `cosinor.py`. Runs 66404249, 66440129 and 66465766 (195 variants, many GPU-hours) each
> trained to completion and silently dropped the "Cosinor (paper)" baseline with
> `ModuleNotFoundError: No module named 'cosinor'`, while every other edited file
> (`hrd_rhythm.py`, `run.sh`, `collect_results.py`) uploaded fine — which is why the runs
> looked healthy. `scripts/run.sh` now refuses to start if a project module is missing.
>
> The old command, corrected, for reference only:
>
> ```bash
> scp -r ./cost.py ./train_hrd.py ./data_preprocessing.py ./hrd_rhythm.py ./cosinor.py \
>        ./globem_preprocessing.py ./datasets ./decomposition_recovery.py ./utils.py \
>        ./models ./scripts ./requirements.txt \
>        melikas@narval.alliancecan.ca:~/projects/def-plago/melikas/projects/rhythmssl_project_GLOBEM/
> ```

What the flags mean: `-a` keep file structure, `-v` show progress, `-P` resume
partial files, `--exclude=...` do **not** upload those (the `.git` folder is huge

> Do not have `rsync` on Windows? Use Git Bash, or this `scp` fallback. It uploads
> the `.git` folder too (slower) and may print a harmless `.git/...rev failed`
> warning at the end — that file is not needed, so you can ignore it:
>
> ```bash
> # LOCAL: simpler but copies extra junk; the dataset still uploads fully
> scp -r ./cost.py ./train_hrd.py ./data_preprocessing.py ./hrd_rhythm.py ./cosinor.py \
>        ./decomposition_recovery.py ./globem_preprocessing.py ./utils.py \
>        ./models ./scripts ./datasets ./requirements.txt \
>        melikas@narval.alliancecan.ca:~/projects/def-plago/melikas/projects/rhythmssl_project/
> ```

> **Every one of these lists must name `cosinor.py`.** It is the paper-cosinor baseline
> engine, imported by `hrd_rhythm.py` as a plain project module. Runs 66404249 and 66440129
> (130/130 variants) lost the "Cosinor (paper)" row to `ModuleNotFoundError: No module named
> 'cosinor'` purely because these `scp` lists omitted the file — the job itself succeeded, so
> the loss showed up only as `n/a` in `summary_models.csv`. Prefer the `rsync ./` command
> above, which uploads the whole tree and cannot miss a module.

The dataset is ~4 GB, so the first upload takes several minutes.

---

## Step 2 — Verify the upload (NARVAL)

```bash
# NARVAL: go into the project and confirm the dataset is there
ssh melikas@narval.alliancecan.ca
ssh melikas@rorqual.alliancecan.ca

```

---

# NARVAL
ssh melikas@narval.alliancecan.ca
cd ~/projects/def-plago/melikas/projects/rhythmssl_project_GLOBEM
sbatch scripts/run.sh
squeue -u melikas

nar: 67041999
117122
scp -r melikas@narval.alliancecan.ca:~/projects/def-plago/melikas/projects/rhythmssl_project_GLOBEM/results_hrd/117122 "c:\Users\umroot\Documents\CoST - GLOBEM\results_hrd\"
scp -r melikas@narval.alliancecan.ca:~/projects/def-plago/melikas/projects/rhythmssl_project_GLOBEM/results_hrd/66474858 "c:\Users\umroot\Documents\CoST - GLOBEM\results_hrd\"



"ssh melikas@narval.alliancecan.ca
cd ~/projects/def-plago/melikas/projects/rhythmssl_project_GLOBEM

module purge
module load StdEnv/2023 python/3.11
pip download --no-deps CosinorPy -d wheels

ls wheels/          # باید CosinorPy-3.1-py3-none-any.whl را ببینید
python -c "import CosinorPy; print('OK')"
"
## Watch the job live 
tail -f logs/cost_hrd-65772176.out
```
logs/cost_hrd-65772176.out


```bash
# NARVAL: open the full log, scroll with arrows, press q to quit
less logs/cost_hrd-<jobid>.out
```


---

## Step 5 — See the job queue (NARVAL)

```bash
# NARVAL: list YOUR jobs and their state
squeue -u melikas
```

The `ST` column means: `PD` = pending (waiting for a GPU), `R` = running.
When your job disappears from this list, it has finished (check the log).

```bash
# NARVAL: full details of one job (node, time used, why it is pending, ...)
scontrol show job <jobid>
```

---

## Step 6 — Cancel a job (NARVAL)

```bash
# NARVAL: cancel one job by its id
scancel 63756625

# NARVAL: cancel ALL of your jobs at once
scancel -u melikas
```

---

## Step 7 — Download the results (Narval → LOCAL)

Results are grouped by job id:
`results_hrd/<jobid>/<backbone>_<pe>_seed<seed>/` containing `metrics.json`
(the scores) and `pretrain_loss.npy` (the loss curve).

```powershell
# LOCAL: back to the project folder
cd c:\Users\umroot\Documents\CoST
```

```bash
# LOCAL: download the results of one job (replace <jobid>)
rsync -avP \
  melikas@narval.alliancecan.ca:~/projects/def-plago/melikas/projects/rhythmssl_project/results_hrd/62952884 \
  ./results_hrd/
```

scp -r melikas@rorqual.alliancecan.ca:~/projects/def-plago/melikas/projects/rhythmssl_project/results_hrd/16439461 c:\Users\umroot\Documents\CoST\results_hrd\

# LOCAL
scp -r melikas@narval.alliancecan.ca:~/projects/def-plago/melikas/projects/rhythmssl_project_GLOBEM/results_hrd/66372999 "c:\Users\umroot\Documents\CoST - GLOBEM\results_hrd\"

# LOCAL — دقت کن: بدون _GLOBEM
scp -r melikas@narval.alliancecan.ca:~/projects/def-plago/melikas/projects/rhythmssl_project/results_hrd/66372999 "c:\Users\umroot\Documents\CoST - GLOBEM\results_hrd\"



```powershell
# LOCAL: print a sorted table and also save a CSV
python scripts/collect_results.py --results-dir results_hrd --csv pe_summary.csv
```

---

## Step 8 — Delete old outputs on Narval (cleanup)

After you have downloaded results you no longer need on the cluster, free up
space. **`rm -rf` permanently deletes — double-check the path first.**

```bash
# NARVAL: see how big each job's results are
cd ~/projects/def-plago/melikas/projects/rhythmssl_project
du -sh results_hrd/*

# NARVAL: delete ONE job's results
rm -rf results_hrd/<jobid>

# NARVAL: delete old log files too (optional)
rm -f logs/cost_hrd-<jobid>.out
```

To clear **everything** under results (be careful):

```bash
# NARVAL: wipe all results (cannot be undone)
rm -rf results_hrd/*
```

---

## Cheat sheet

| I want to... | Where | Command |
|--------------|-------|---------|
| Log in | LOCAL | `ssh melikas@narval.alliancecan.ca` |
| Upload project | LOCAL | `rsync -avP --exclude='.git' ... ./ ...:.../rhythmssl_project/` |
| Submit baseline job | NARVAL | `sbatch scripts/run.sh` |
| Watch live / errors | NARVAL | `tail -f logs/cost_hrd-<jobid>.out` |
| Job queue | NARVAL | `squeue -u melikas` |
| Cancel a job | NARVAL | `scancel <jobid>` |
| Download results | LOCAL | `rsync -avP ...:.../results_hrd/<jobid> ./results_hrd/` |
| Compare results | LOCAL | `python scripts/collect_results.py --results-dir results_hrd` |
| Delete old results | NARVAL | `rm -rf results_hrd/<jobid>` |

## Notes

- One A100 GPU is enough; 64 GB RAM covers the ~4 GB CSV plus the model.
- Baseline (2 backbones) ≈ 2–3 h. The full 11-variant sweep takes proportionally longer.
- Change settings by editing the `srun python train_hrd.py ...` line in
  `scripts/run.sh` (e.g. `--iters`, `--seed`, `--window-hours`, `--bin-minutes`).

du -sh 65772176
diskusage_report
گزینهٔ A — فقط فایل‌های بزرگِ net.pt را پاک کنید (توصیه‌شده)
حجمِ اصلی از net.ptهاست (هر کدام ۱۵۰MB × ~۶۰ ≈ ۹GB). خودِ نتایج (json/csv/metrics/png) فقط چند KB‌اند. این‌طور بیشترِ فضا آزاد می‌شود ولی همهٔ نتایجِ تحلیل می‌مانند:


find 65772176 -name "net.pt" -delete
du -sh 65772176        # حالا باید خیلی کوچک باشد
گزینهٔ B — کلِ پوشهٔ job را کامل پاک کنید

rm -rf 65772176
(بازگشت‌ناپذیر — همهٔ نتایج، لاگ‌ها، شکل‌ها و مدل‌ها می‌روند.)

تأیید آزادسازیِ فضا:


diskusage_report