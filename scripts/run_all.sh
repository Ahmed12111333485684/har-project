#!/usr/bin/env bash
# Full pipeline. Resumable: re-running skips finished runs.  (Colab: prefix lines with '!')
set -e
DL="cnn_bilstm transformer resnet1d"
python data/download.py

# 1) hyper-parameter tuning: identical budget (3 trials, seed 0, VALIDATION only) for every model
python -m src.run_experiments tune

# 2) final runs: best config x 3 seeds, TEST evaluated once
python -m src.run_experiments final

# 3) data-strategy study (validation only)
python -m src.run_experiments sweep --models $DL --tag aug \
   --grid 'augment.names=[],[jitter],[scaling],[time_warp],[magnitude_warp],[jitter,scaling,time_warp]'
python -m src.run_experiments sweep --models $DL --tag classweights --grid train.class_weights=false,true
python -m src.run_experiments sweep --models $DL --tag window --grid data.window_len=64,96,128   # representation

# 4) more hyper-parameters
python -m src.run_experiments sweep --models $DL --tag batchsize --grid train.batch_size=32,64,128

# 5) self-supervised pre-training vs scratch, with few labels (H4)
python -m src.pretrain_ssl --epochs 40 --out checkpoints/transformer_ssl.pt
python -m src.run_experiments sweep --models transformer --tag ssl \
   --grid pretrained=null,checkpoints/transformer_ssl.pt --grid label_fraction=0.05,0.1,0.25,1.0

# 6) ablations (test evaluated; not used for any selection)
python -m src.run_experiments ablate

# 7) analysis + tables
python -m src.error_analysis --seed 0
python -m src.report_tables
