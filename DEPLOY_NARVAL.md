# Deploy CoST to Narval (Compute Canada) - Simple Guide

## موارد مورد نیاز
- **Local:** Windows PowerShell یا Git Bash
- **Remote:** `melikas@narval.alliancecan.ca`
- **Project Path:** `~/projects/def-plago/melikas/projects/rhythmssl_project`

---

## Step 1: آپلود پروژه

**از روی کامپیوتر شخصی خود، داخل پوشه CoST:**

```bash
cd c:\Users\umroot\Documents\CoST
```

**سپس (Windows PowerShell یا Git Bash):**

```bash
scp -r ./* melikas@narval.alliancecan.ca:"~/projects/def-plago/melikas/projects/rhythmssl_project/"
```

**یا rsync (اگر داشتید):**
```bash
rsync -avP --exclude='__pycache__' --exclude='*.pyc' --exclude='.git' \
      --exclude='results_hrd' --exclude='RhythmSSL' \
      ./ melikas@narval.alliancecan.ca:"~/projects/def-plago/melikas/projects/rhythmssl_project/"
```

**اگر همچنان fail شود، از full path استفاده کنید:**
```bash
scp -r ./* melikas@narval.alliancecan.ca:/home/melikas/projects/def-plago/melikas/projects/rhythmssl_project/
```
scp c:\Users\umroot\Documents\CoST\run.sh melikas@narval.alliancecan.ca:~/projects/def-plago/melikas/projects/rhythmssl_project/run.sh
---

## Step 2: تایید آپلود

```bash
ssh melikas@narval.alliancecan.ca
cd ~/projects/def-plago/melikas/projects/rhythmssl_project
ls -lh                                                # فایل‌های موجود
ls -lh datasets/HRD_RAW_MinuteLevel.csv              # تایید دیتا (~4.5 GB)
```

---

## Step 3: اجرای Training

**داخل Narval:**

```bash
cd ~/projects/def-plago/melikas/projects/rhythmssl_project
sbatch run.sh
```
62642322
``````

---

## Step 4: مانیتورینگ

**وضعیت:**
```bash
squeue -u melikas
```

**لاگ‌های live:**
```bash
tail -f logs/cost_hrd-62642322.out
```

**جزئیات کامل:**
```bash
scontrol show job 62638207
```

**لغو جاب:**
```bash
scancel 62638207
```

---

## Step 5: دانلود نتایج

**از روی کامپیوتر شخصی:**

```bash
cd c:\Users\umroot\Documents\CoST

scp -r melikas@narval.alliancecan.ca:~/projects/def-plago/melikas/projects/rhythmssl_project/results_hrd ./
```

**یا rsync:**
```bash
rsync -avP melikas@narval.alliancecan.ca:~/projects/def-plago/melikas/projects/rhythmssl_project/results_hrd ./
```

---

## Files Output

در `results_hrd/`:
```
metrics_tcn_seed42.json              # TCN نتایج (AUC/F1/Accuracy)
metrics_transformer_seed42.json      # Transformer نتایج
pretrain_loss_tcn_seed42.npy         # TCN loss curve
pretrain_loss_transformer_seed42.npy # Transformer loss curve
```

---

## Quick Cheat Sheet

| کار | دستور |
|-----|--------|
| آپلود | `scp -r ./ melikas@narval.alliancecan.ca:.../rhythmssl_project/` |
| اجرا | `sbatch run.sh` |
| وضعیت | `squeue -u melikas` |
| لاگ | `tail -f logs/cost_hrd-*.out` |
| دانلود | `scp -r melikas@narval.alliancecan.ca:.../results_hrd ./` |
| لغو | `scancel <jobid>` |
| خروج | `exit` |

---

## نکات مهم

✓ **دستورات rsync/scp** را از **کامپیوتر شخصی** اجرا کنید  
✓ **sbatch run.sh** را داخل **Narval** اجرا کنید  
✓ **tail -f** برای مشاهده لاگ live (Ctrl+C برای خروج)  
✓ آپلود 4.5 GB ممکن است **30 دقیقه تا 1 ساعت** طول بکشد  
✓ Training بعد از اجرا **۲-۳ ساعت** طول می‌کشد  

# SLURM uses %A → filename has NO underscore
tail -f ~/projects/def-plago/melikas/projects/rhythmssl_project/logs/cost_hrd-62560916.out

tail -f ~/projects/def-plago/melikas/projects/rhythmssl_project/logs/cost_hrd-62560916.err

tail -f ~/projects/def-plago/melikas/projects/rhythmssl_project/logs/cost_hrd-62642322.{out,err}

result
"cd c:\Users\umroot\Documents\CoST

scp -r melikas@narval.alliancecan.ca:~/projects/def-plago/melikas/projects/rhythmssl_project/results_hrd ./
"
scp -r melikas@narval.alliancecan.ca:~/projects/def-plago/melikas/projects/rhythmssl_project/results_hrd ./
