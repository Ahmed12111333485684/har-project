# Wearable Human Activity Recognition (UCI-HAR) - AI403 Deep Learning Final Project

**Team:** _<team name / members>_ · **Idea chosen:** #9 Wearable human activity recognition (inertial sensors -> activity)

## 1. Task and best result
Classify a 2.56 s window (128 samples @ 50 Hz, 9 channels: total_acc, body_acc, body_gyro x/y/z) of a
waist-worn smartphone into one of 6 activities (Walking, Upstairs, Downstairs, Sitting, Standing, Laying).
Primary metric: **macro-F1** on **unseen subjects**. _Best result: fill in from `experiments/tables/tables.md`._

Models (one per required family, all under the same protocol):

| Role | Model | File |
|---|---|---|
| Baseline | hand-crafted features + Random Forest | `src/models/baseline.py`, `src/datasets/features.py` |
| Classic | 1D-CNN + BiLSTM | `src/models/cnn_bilstm.py` |
| Modern | Transformer encoder (**self-attention written from scratch**, optional masked-patch SSL pre-training) | `src/models/transformer.py` |
| Advanced | ResNet-1D + Squeeze-Excitation + temporal attention pooling | `src/models/resnet1d.py` |

### Experimental-protocol decisions (and why)
* **Subject-wise, leakage-free split** (`src/datasets/har.py`): the 21 official train subjects -> 15 train + 6 validation (fixed seed); 9 official test subjects only for final evaluation. A leakage assertion runs on every load. Normalisation stats come from train subjects only.
* **Test set touched only by final runs** (`--test`). Tuning, sweeps and early stopping use validation only. Ablations report test but are never used to choose anything.
* **Fair comparison:** identical split, preprocessing, epochs/patience, and tuning budget (3 trials, seed 0) for every model; final configs run with seeds 0, 1, 2.
* **Confidence intervals:** subject-level bootstrap (resamples people, not windows - windows of one person are correlated).
* **Augmentations written by us** (`src/datasets/augment.py`): jitter, scaling, time-warp, magnitude-warp.

## 2. Environment setup
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Google Colab: upload the repo, `!pip install -r requirements.txt`, then run the same commands with `!` in front.

## 3. Dataset download and preprocessing
```bash
python data/download.py          # -> data/raw/UCI HAR Dataset  (manual link inside the script if it fails)
```
Preprocessing (windowing is already done by the dataset authors: 128 samples, 50 % overlap) happens in
`get_splits()`: subject split -> channel selection -> z-score. A cache is written to `data/processed/`.
Source: Anguita et al., UCI ML Repository, DOI 10.24432/C54S4K. Licence: CC BY 4.0.

## 4. Training (one model, one seed)
```bash
python -m src.train --config configs/rf.yaml          --seed 0
python -m src.train --config configs/cnn_bilstm.yaml  --seed 0
python -m src.train --config configs/transformer.yaml --seed 0
python -m src.train --config configs/resnet1d.yaml    --seed 0
```
Whole project (tuning -> 3 seeds -> studies -> ablations -> tables), resumable:
```bash
bash scripts/run_all.sh
```
Pieces: `python -m src.run_experiments {tune|final|sweep|ablate}`, `python -m src.pretrain_ssl`.
Add `--set train.epochs=2 --out-root /tmp/smoke` for a 2-minute smoke test of everything.

## 5. Evaluation
```bash
python -m src.evaluate --run-dir experiments/runs/main/transformer__full/seed0 --split test
python -m src.error_analysis --seed 0      # confusion matrices, Table 6, example plots
python -m src.report_tables                # Tables 4, 5, 7 + sweeps -> experiments/tables/tables.md
```

## 6. Demo
`demo/` - Gradio app (3D visualisation of the whole pipeline). _Built in the next step._

## 7. Hardware and expected runtimes
Measured on a **1-vCPU Xeon CPU** sandbox (no GPU): ~3-5 s/epoch per DL model, RF ~7 s.
Full `run_all.sh` takes a few hours on 1 CPU and well under an hour on a Colab GPU.
_Fill in your own hardware/runtimes from Table 7 (`experiments/tables/table7_cost.csv`)._

## 8. Credits
* Dataset: Anguita, Ghio, Oneto, Parra, Reyes-Ortiz, "A Public Domain Dataset for Human Activity Recognition Using Smartphones", ESANN 2013.
* DeepConvLSTM idea: Ordonez & Roggen 2016. Transformer: Vaswani et al. 2017. ResNet: He et al. 2016. SE: Hu et al. 2018. Masked reconstruction: He et al. 2022 (MAE) - all re-implemented by us, no external model code or pretrained weights used.
* Libraries: PyTorch, scikit-learn, SciPy, pandas, matplotlib, Gradio.
