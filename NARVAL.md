

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

Multifactor authentication (MFA) is mandatory. If you have not set it up:
https://ccdb.alliancecan.ca/multi_factor_authentications

Type `exit` any time to close the Narval session and return to your PC.

---

## Step 1 — Upload your project (LOCAL → Narval)

Do this from your PC. It copies the code **and** the dataset to Narval.

```powershell
# LOCAL: go to the project folder
cd c:\Users\umroot\Documents\CoST
```

# LOCAL
scp scripts/run_debug.sh melikas@narval.alliancecan.ca:~/projects/def-plago/melikas/projects/rhythmssl_project/scripts/


62733138


```bash
# LOCAL: create the destination folder on Narval (one time)
ssh melikas@narval.alliancecan.ca "mkdir -p ~/projects/def-plago/melikas/projects/rhythmssl_project"
```

Now upload. **Prefer `rsync`** — it skips junk (`.git`, caches, old results) and
can resume if interrupted:

```bash
# LOCAL: upload everything needed (code + datasets/HRD_RAW_MinuteLevel.csv)
rsync -avP \
  --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='results_hrd' \
  ./ melikas@narval.alliancecan.ca:~/projects/def-plago/melikas/projects/rhythmssl_project/
```
scp -r ./ melikas@narval.alliancecan.ca:~/projects/def-plago/melikas/projects/rhythmssl_project/

What the flags mean: `-a` keep file structure, `-v` show progress, `-P` resume
partial files, `--exclude=...` do **not** upload those (the `.git` folder is huge
and useless on the cluster).

> Do not have `rsync` on Windows? Use Git Bash, or this `scp` fallback. It uploads
> the `.git` folder too (slower) and may print a harmless `.git/...rev failed`
> warning at the end — that file is not needed, so you can ignore it:
>
> ```bash
> # LOCAL: simpler but copies extra junk; the dataset still uploads fully
> scp -r ./cost.py ./train_hrd.py ./data_preprocessing.py ./utils.py ./datautils.py \
>        ./models ./scripts ./datasets ./requirements.txt \
>        melikas@narval.alliancecan.ca:~/projects/def-plago/melikas/projects/rhythmssl_project/
> ```

The dataset is ~4 GB, so the first upload takes several minutes.

---

## Step 2 — Verify the upload (NARVAL)

```bash
# NARVAL: go into the project and confirm the dataset is there
ssh melikas@narval.alliancecan.ca
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
```
62749320


It prints something like `Submitted batch job 62642322`. **Write down that
number** — it is your `<jobid>`.

**Want all positional-encoding variants instead** (8 PEs + Time2Vec + TCN variants)?
Open `scripts/run.sh`, find the block marked `OPTION E`, remove the `#` at the
start of its two lines (and comment out OPTION A above it), then `sbatch scripts/run.sh`.

---

## Step 4 — Watch the job live / see errors (NARVAL)

This job writes **both normal output and errors into one file**:
`logs/cost_hrd-<jobid>.out` (there is no separate `.err` file).

```bash
# NARVAL: follow the log live as the job runs (press Ctrl+C to stop watching)
tail -f logs/cost_hrd-62749320.out
```
logs/cost_hrd-62749320.out

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
scancel 62744632

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
  melikas@narval.alliancecan.ca:~/projects/def-plago/melikas/projects/rhythmssl_project/results_hrd/<jobid> \
  ./results_hrd/
```

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
