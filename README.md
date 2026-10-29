# Personalized expert routing for PPG-based stress detection

Training and evaluation code for a 10 s wrist-PPG stress / non-stress
classifier: an InceptionTime backbone, K LSTM experts, and a router that mixes a
gate computed from 10 handcrafted subject descriptors with a gate computed from
the expert features. Everything runs from `train.py`.

---

## Installation

Python 3.10+, PyTorch 2.x, single GPU recommended (CPU works but is slow).

```bash
conda create -n ppg-stress python=3.10
conda activate ppg-stress
pip install -r requirements.txt
```

---

## Data

The datasets are not included. Download them from their sources:

| Dataset | Source |
|---|---|
| WESAD | UCI Machine Learning Repository, dataset 465 |
| CLAS  | Markova, Ganchev & Kalinkov, *BIA* 2019, doi:10.1109/BIA48344.2019.8967457 |
| CATSA | Motin et al., *Data in Brief* 63 (2025) 112295, doi:10.6084/m9.figshare.28690988 |

Convert them to the pipeline's 125 Hz segment files:

```bash
export WESAD_RAW=/path/to/WESAD        # contains S2/ ... S17/
export CLAS_RAW=/path/to/CLAS          # contains Block_details/ and Participants/
export CATSA_RAW=/path/to/CATSA        # contains Sub1/ ... Sub50/

bash scripts/prepare_data.sh
```

This writes `dataset/WESAD_125Hz.pkl`, `dataset/CLAS_125Hz.pkl` and
`dataset/CATSA_125Hz.pkl`. Any of the three can be omitted; that dataset is then
skipped. The docstring at the top of each `dataset/build_*.py` lists the exact
input layout, which recordings are used for each class, and how the signal is
resampled.

---

## Training

```bash
# 5 folds x 3 datasets with the reported configuration
python train.py --config configs/final.yaml
```

Equivalently, without the config file:

```bash
python train.py \
    --stage finetune \
    --data-dir dataset --out-dir runs \
    --pretrain-ckpt checkpoints/pretrain_inception_time_vital.pth \
    --datasets WESAD CLAS CATSA \
    --backbone inception_time --expert-num 4 --expert-type lstm --alpha 0.3 \
    --lambda1 1.0 --lambda2 0.1 \
    --lr 1e-4 --backbone-lr 5e-5 --weight-decay 2e-3 \
    --batch-size 128 --epochs 1000 --patience 50
```

Command-line flags override values from `--config`. Useful flags:

| Flag | Meaning |
|---|---|
| `--fold-idx N` | run only fold `N` (0-4); default `-1` runs all five |
| `--datasets WESAD` | restrict to a subset of datasets |
| `--scratch` | random-init backbone instead of the shipped checkpoint |
| `--tag NAME` | suffix for output filenames (keeps runs apart) |
| `--enrollment-s N` | length of the enrolment segment in seconds (default 120) |
| `--no-balance-classes` | fixed-stride windows instead of class-balanced sampling |
| `--feature-gate {expert0,per_expert,backbone_gap,expert_mean,expert_concat}` | gate input |
| `--no-domain-adapter` / `--no-head-skip` / `--no-gate-norm` / `--no-mixstyle` | ablation switches |
| `--global-ind-norm` | fit the descriptor min-max scaler over all subjects, held-out folds included (transductive; only the earlier runs used this) |
| `--freeze-backbone` | train the head and experts only |

Outputs under `--out-dir` (default `runs/`):

```
runs/
├── cache/segments_125Hz_e120s_f4cheby.pkl     # windowed, filtered, cached
├── checkpoints/Finetune_<DS>_K4_f<fold>_best.pth
├── preds/<DS>_<tag>_f<fold>_{test,val}_{y_true,y_pred_proba,subject}.npy
└── results.json                                # per-fold metrics + full config
```

### Pretraining (optional)

The shipped backbone was pretrained on VitalDB: 10 s / 125 Hz PPG segments,
each window regressed to its own ten handcrafted individual descriptors. To
rebuild the corpus and retrain the backbone:

```bash
# corpus build: separate Python 3.10 env (pyPPG and vitaldb pins conflict
# with the training environment - see requirements-corpus.txt)
conda create -n ppg-corpus python=3.10
conda activate ppg-corpus
pip install --no-deps -r requirements-corpus.txt
python dataset/build_vital_ppg.py --vital-dir /data/vital/1.0.0/vital_files \
                                  --out dataset/VITALDB_10s

# back in the training env
python train.py --stage pretrain --vital-dir dataset/VITALDB_10s \
                --pretrain-epochs 100 --pretrain-batch-size 512
```

Download VitalDB 1.0.0 from https://vitaldb.net and unpack `vital_files/`;
`build_vital_ppg.py` keeps the `SNUADC/PLETH` track of every recording. The
build takes hours of CPU and produces roughly 6M segments. Training splits the
subjects 85/15 into train/validation, keeps at most 192 segments per subject
and restores the epoch with the lowest validation loss. `--stage both` runs
pretraining and finetuning in sequence. A different corpus produces a
different backbone.

---

## Evaluation

Per-window predictions are dumped at the end of every fold, so metrics can be
recomputed without retraining:

```bash
python evaluate.py runs/preds --out runs/metrics.md
```

This prints per-dataset, per-fold and pooled accuracy / precision / recall /
specificity / F1 / AUC, plus a subject-level bootstrap confidence interval, and
writes a Markdown table alongside a JSON copy. Restrict to one run with
`--tag NAME`, or recompute on the validation split with `--split val`.

---

## Configuration

All hyperparameters live in [`configs/final.yaml`](configs/final.yaml), whose
keys are the argparse option names with `--` removed and dashes written as
underscores. Unknown keys are rejected. Anything not listed keeps its default,
so a single file describes a whole experiment:

```bash
python train.py --config configs/final.yaml --fold-idx 0 --tag smoke
```

---

## License

Code: MIT (see [`LICENSE`](LICENSE)). The datasets keep their own licences;
redistributing the segment files built from them is governed by their terms.
