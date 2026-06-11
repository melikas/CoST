# CoST: Contrastive Learning of Disentangled Seasonal-Trend Representations for Time Series Forecasting (ICLR 2022)

<p align="center">
<img src=".\pics\CoST.png" width = "700" alt="" align=center />
<br><br>
<b>Figure 1.</b> Overall CoST Architecture.
</p>

Official PyTorch code repository for the [CoST paper](https://openreview.net/forum?id=PilZY3omXV2).

* CoST is a contrastive learning method for learning disentangled seasonal-trend representations for time series forecasting.
* CoST consistently outperforms state-of-the-art methods by a considerable margin, achieveing a 21.3% improvement in MSE on multivariate benchmarks.
  
## Requirements
1. Install Python 3.8, and the required dependencies.
2. Required dependencies can be installed by: ```pip install -r requirements.txt```

## Data

The datasets can be obtained and put into `datasets/` folder in the following way:

* [3 ETT datasets](https://github.com/zhouhaoyi/ETDataset) should be placed at `datasets/ETTh1.csv`, `datasets/ETTh2.csv` and `datasets/ETTm1.csv`.
* [Electricity dataset](https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014) placed at `datasets/LD2011_2014.txt` and run `electricity.py`.
* [Weather dataset](https://drive.google.com/drive/folders/1ohGYWWohJlOlb2gsGTeEq3Wii2egnEPR) (link from [Informer repository](https://github.com/zhouhaoyi/Informer2020)) placed at `datasets/WTH.csv`
* [M5 dataset](https://drive.google.com/drive/folders/1D6EWdVSaOtrP1LEFh1REjI3vej6iUS_4) place `calendar.csv`, `sales_train_validation.csv`, `sales_train_evaluation.csv`, `sales_test_validation.csv` and `sales_test_evaluation.csv` at `datasets/` and run m5.py.

## Usage
To train and evaluate CoST on a dataset, run the script from the scripts folder: ```./scripts/ETT_CoST.sh``` (edit file permissions via ```chmod u+x scripts/*```).

After training and evaluation, the trained encoder, output and evaluation metrics can be found in `training/<DatasetName>/<RunName>_<Date>_<Time>/`.

Alternatively, you can directly run the python scripts:
```train & evaluate
python train.py <dataset_name> <run_name> --archive <archive> --batch-size <batch_size> --repr-dims <repr_dims> --gpu <gpu> --eval
```
The detailed descriptions about the arguments are as following:
| Parameter name | Description of parameter |
| --- | --- |
| dataset_name | The dataset name |
| run_name | The folder name used to save model, output and evaluation metrics. This can be set to any word |
| archive | The archive name that the dataset belongs to. This can be set to `forecast_csv` or `forecast_csv_univar` |
| batch_size | The batch size (defaults to 8) |
| repr_dims | The representation dimensions (defaults to 320) |
| gpu | The gpu no. used for training and inference (defaults to 0) |
| eval | Whether to perform evaluation after training |
| kernels | Kernel sizes for mixture of AR experts module |
| alpha | Weight for loss function |

(For descriptions of more arguments, run `python train.py -h`.)

## Positional Encoding Experiments (HRD)

This fork adds swappable **positional encodings (PE)** to the CoST Transformer
backbone and extends **Time2Vec** to the TCN backbone, trained on the HRD
wearable data for depression-endpoint classification via
[train_hrd.py](train_hrd.py).

**Code added**

| File | Purpose |
|------|---------|
| [models/positional_encoding.py](models/positional_encoding.py) | All PE variants, the `Time2Vec` embedding, and the PE-aware self-attention / encoder layer. |
| [models/encoder.py](models/encoder.py) | `TransformerFeatureExtractor` takes a `pe` argument; `CoSTEncoder` injects Time2Vec into the TCN hidden stream. |
| [cost.py](cost.py) / [train_hrd.py](train_hrd.py) | `CoST(...)` and the `--pe` flag forward the choice to the encoder. |
| [scripts/run_pe_experiments.sh](scripts/run_pe_experiments.sh) | Runs the full variant sweep. |
| [scripts/collect_results.py](scripts/collect_results.py) | Aggregates every `metrics.json` into one comparison table. |
| [scripts/test_pe_variants.py](scripts/test_pe_variants.py) | CPU smoke test for all variants. |

**Supported encodings**

*Absolute* (added to the embeddings): `sinusoidal` (Transformer baseline),
`learnable`, `tape` (tAPE), `time2vec`. *Attention* (injected inside each
self-attention layer): `rpe`, `erpe`, `tupe`, `convspe`, `tpe`.

| Backbone | Valid `--pe` | Default |
|----------|-------------|---------|
| `transformer` | the 8 methods above **+** `time2vec` | `sinusoidal` |
| `tcn` | `none` (baseline), `time2vec` | `none` |

The TCN is position-aware through its dilated convolutions, so its baseline uses
no PE; `time2vec` adds a learnable time embedding to the hidden stream.

**Run a single variant**

```bash
python train_hrd.py --sensor-csv datasets/HRD_RAW_MinuteLevel.csv --backbone transformer --pe tupe
python train_hrd.py --sensor-csv datasets/HRD_RAW_MinuteLevel.csv --backbone tcn --pe none       # TCN baseline
python train_hrd.py --sensor-csv datasets/HRD_RAW_MinuteLevel.csv --backbone tcn --pe time2vec    # TCN + Time2Vec
```

Each run writes to `results_hrd/<run-id>/<backbone>_<pe>_seed<seed>/` (run-id is
`$SLURM_JOB_ID` on the cluster, else `local`), containing `metrics.json` and
`pretrain_loss.npy`, so variants never overwrite one another.

**Run the full sweep** (Transformer×8 + Transformer/Time2Vec + TCN baseline + TCN/Time2Vec):

```bash
bash scripts/run_pe_experiments.sh datasets/HRD_RAW_MinuteLevel.csv results_hrd 42
```

**Compare results** (one table, sorted by participant-level AUC):

```bash
python scripts/collect_results.py --results-dir results_hrd --csv pe_summary.csv
```

**Verify the implementation** (CPU, seconds — builds, pretrains and encodes every variant):

```bash
python scripts/test_pe_variants.py
```

**Add a new PE:** implement it in
[models/positional_encoding.py](models/positional_encoding.py) (an absolute term
in `add_absolute_pe`, or a branch in `PESelfAttention.forward`) and list it in
`ABSOLUTE_PES` / `ATTENTION_PES` plus the `--pe` choices in
[train_hrd.py](train_hrd.py). No other wiring is needed.

## Main Results
We perform experiments on five real-world public benchmark datasets, comparing against both state-of-the-art representation learning and end-to-end forecasting approaches. 
CoST achieves state-of-the-art performance, beating the best performing end-to-end forecasting approach by 39.3% and 18.22% (MSE) in the multivariate and univariate settings
respectively. CoST also beats next best performing feature-based approach by 21.3% and 4.71% (MSE) in the multivariate and univariate settings respectively (refer to main paper for full results).

<p align="center">
<img src=".\pics\results.png" width = "700" alt="" align=center />
</p>

## FAQs
**Q**: ValueError: Found array with dim 4. StandardScaler expected <= 2.

**A**: Please install the appropriate package requirements as found in ```requirements.txt```, in particular, ```scikit_learn==0.24.1```.

**Q**: How to set the ``--kernels`` parameter?

**A**: It should be list of space separated integers, e.g. ```--kernels 1 2 4```. See the `scripts` folder for further examples.

## Acknowledgements
The implementation of CoST relies on resources from the following codebases and repositories, we thank the original authors for open-sourcing their work.
* https://github.com/yuezhihan/ts2vec
* https://github.com/zhouhaoyi/Informer2020

## Citation
Please consider citing if you find this code useful to your research.
<pre>@inproceedings{
    woo2022cost,
    title={Co{ST}: Contrastive Learning of Disentangled Seasonal-Trend Representations for Time Series Forecasting},
    author={Gerald Woo and Chenghao Liu and Doyen Sahoo and Akshat Kumar and Steven Hoi},
    booktitle={International Conference on Learning Representations},
    year={2022},
    url={https://openreview.net/forum?id=PilZY3omXV2}
}</pre>
