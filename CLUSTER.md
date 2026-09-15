- Login: `melikas@NARVAL.alliancecan.ca`
- Project on nibi: `~/projects/def-plago/melikas/projects/rhythmssl_project`

---

## Step 0 — Log in to nibi

```bash
# NARVAL: open a session on the cluster (asks for password + MFA code)
ssh melikas@narval.alliancecan.ca
```
scp -r ./cost.py ./utils.py ./train_hrd.py ./train_hrd_energy.py ./train_globem.py ./model_build.py ./experiment_q1.py ./experiment_q2.py ./experiment_q3.py ./requirements.txt ./models ./tasks ./baselines ./scripts melikas@narval.alliancecan.ca:~/projects/def-plago/melikas/projects/rhythmssl_project/



> Do not have `rsync` on Windows? Use Git Bash, or this `scp` fallback. It uploads
> the `.git` folder too (slower) and may print a harmless `.git/...rev failed`
> warning at the end — that file is not needed, so you can ignore it:
>
> ```bash
> # LOCAL: simpler but copies extra junk; the dataset still uploads fully
> scp -r 
>        melikas@narval.alliancecan.ca:~/projects/def-plago/melikas/projects/rhythmssl_project/
> ```

## Step 2 — Verify the upload (NARVAL)

```bash
# NARVAL: go into the project and confirm the dataset is there
ssh melikas@narval.alliancecan.ca
ssh melikas@rorqual.alliancecan.ca
ssh melikas@nibi.alliancecan.ca
cd ~/projects/def-plago/melikas/projects/rhythmssl_project
ls -lh datasets/HRD_RAW_MinuteLevel.csv
```


## Step 3 — Submit a job (NARVAL)

`sbatch` puts your job in the queue; the cluster runs it when a GPU is free.
You do **not** wait at the terminal — the job runs in the background.

```bash
# NARVAL: run the BASELINE experiment (TCN + Transformer/sinusoidal)
cd ~/projects/def-plago/melikas/projects/rhythmssl_project
sbatch scripts/run.sh
# 2. smoke test -- the heaviest task, no self-heal
sbatch --array=12 scripts/run.sh
"
narval: 


logs/cost_rq1-19421181_0.out       (seed 43)
number** — it is your `<jobid>`.


# NARVAL: follow the log live as the job runs 
tail -n 40 logs/cost_hrd-20071782_0.out
```
logs/rhythmssl_project-16829002.out
less logs/cost_hrd-<jobid>.out


squeue -u melikas

```bash
# NARVAL: full details of one job (node, time used, why it is pending, ...)
scontrol show job <jobid>
```

---

## Step 6 — Cancel a job (NARVAL)

```bash
# NARVAL: cancel one job by its id
scancel 

# NARVAL: cancel ALL of your jobs at once
scancel -u melikas
```

---

## Step 7 — Download the results (Narval → LOCAL)

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

## Notes

- One A100 GPU is enough; 64 GB RAM covers the ~4 GB CSV plus the model.