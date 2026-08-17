

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
scp -r ./cost.py ./train_hrd.py ./data_preprocessing.py ./hrd_rhythm.py ./decomposition_recovery.py ./utils.py ./models ./scripts ./requirements.txt melikas@narval.alliancecan.ca:~/projects/def-plago/melikas/projects/rhythmssl_project/


scp -r ./cost.py ./utils.py ./train_hrd.py ./train_globem.py ./experiment_q1.py ./experiment_q2.py ./experiment_q3.py ./requirements.txt ./models ./tasks ./tasks_globem ./baselines ./data_processing ./scripts melikas@nibi.alliancecan.ca:~/projects/def-plago/melikas/projects/rhythmssl_project/





What the flags mean: `-a` keep file structure, `-v` show progress, `-P` resume
partial files, `--exclude=...` do **not** upload those (the `.git` folder is huge

> Do not have `rsync` on Windows? Use Git Bash, or this `scp` fallback. It uploads
> the `.git` folder too (slower) and may print a harmless `.git/...rev failed`
> warning at the end — that file is not needed, so you can ignore it:
>
> ```bash
> # LOCAL: simpler but copies extra junk; the dataset still uploads fully
> scp -r ./cost.py ./train_hrd.py ./data_preprocessing.py ./hrd_rhythm.py ./decomposition_recovery.py ./utils.py \
>        ./models ./scripts ./datasets ./requirements.txt \
>        melikas@narval.alliancecan.ca:~/projects/def-plago/melikas/projects/rhythmssl_project/
> ```

The dataset is ~4 GB, so the first upload takes several minutes.

---

## Step 2 — Verify the upload (NARVAL)

```bash
# NARVAL: go into the project and confirm the dataset is there
ssh melikas@narval.alliancecan.ca
ssh melikas@rorqual.alliancecan.ca
ssh melikas@nibi.alliancecan.ca
cd ~/projects/def-plago/melikas/projects/rhythmssl_project
ls -lh datasets/HRD_RAW_MinuteLevel.csv
```

---

## Step 3 — Submit a job (NARVAL)

`sbatch` puts your job in the queue; the cluster runs it when a GPU is free.
You do **not** wait at the terminal — the job runs in the background.

```bash
# NARVAL: run the BASELINE experiment (TCN + Transformer/sinusoidal)
cd ~/projects/def-plago/melikas/projects/rhythmssl_project
sbatch scripts/run.sh
# 2. smoke test -- the heaviest task, no self-heal
sbatch --array=12 scripts/run.sh
```
ror: 

nibi: 19649817

logs/cost_rq1-19421181_0.out       (seed 43)
logs/cost_rq1-19422314_[1-6].out   (بقیهٔ seedها)

It prints something like `Submitted batch job 62977394`. **Write down that
number** — it is your `<jobid>`.


---

## Step 4 — Watch the job live / see errors (NARVAL)

This job writes **both normal output and errors into one file**:
`logs/cost_hrd-<jobid>.out` (there is no separate `.err` file).

```bash
# NARVAL: follow the log live as the job runs (press Ctrl+C to stop watching)
tail -f logs/cost_hrd-18115719.out
```
logs/rhythmssl_project-16829002.out

`tail -f` keeps printing new lines as they appear. Stopping it with `Ctrl+C`
does **not** stop the job — only stops watching. To read the whole log at once:

```bash
# NARVAL: open the full log, scroll with arrows, press q to quit
less logs/cost_hrd-<jobid>.out
```

If the job crashed, the Python error message (traceback) is at the bottom of
this same `.out` file.

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
scancel 18944108

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


scp -r melikas@nibi.alliancecan.ca:~/projects/def-plago/melikas/projects/rhythmssl_project/results_hrd/19606825 "c:\Users\umroot\Documents\CoST - Rorqual\results_hrd\"

scp -r melikas@nibi.alliancecan.ca:~/projects/def-plago/melikas/projects/rhythmssl_project/results_hrd_energy/19314126 "c:\Users\umroot\Documents\CoST - Rorqual\results_hrd_energy\"


smoke download 
# LOCAL
scp -r "melikas@nibi.alliancecan.ca:~/projects/def-plago/melikas/projects/rhythmssl_project/results_hrd/19422314" "c:\Users\umroot\Documents\CoST - Rorqual\results_hrd\"




Then build one comparison table of every variant:

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