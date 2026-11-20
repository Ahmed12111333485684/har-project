"""
UCI-HAR loading and LEAKAGE-FREE, SUBJECT-WISE splitting.

Why subject-wise?  Windows overlap by 50 % and one person produces hundreds of
near-identical windows.  A random window split puts neighbours of the same
person in train and test -> inflated accuracy.  Here every subject lives in
exactly ONE of train / val / test and this is asserted in code.

  official train subjects (21)  ->  train (15) + validation (6, fixed seed)
  official test  subjects (9)   ->  test  (touched only for final evaluation)
Normalisation statistics are computed on the TRAIN subjects only.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

ACTIVITIES = ["Walking", "Upstairs", "Downstairs", "Sitting", "Standing", "Laying"]
SENSORS = ["total_acc", "body_acc", "body_gyro"]
AXES = "xyz"
CHANNEL_NAMES = [f"{s}_{a}" for s in SENSORS for a in AXES]
CHANNEL_GROUPS = {
    "all": list(range(9)),
    "acc_only": [0, 1, 2, 3, 4, 5],      # ablation: no gyroscope
    "gyro_only": [6, 7, 8],
    "total_acc_gyro": [0, 1, 2, 6, 7, 8],
}
FULL_LEN = 128


def find_root(root) -> Path:
    root = Path(root)
    for p in sorted(root.rglob("Inertial Signals")):
        if "__MACOSX" in p.parts:
            continue
        if p.parent.name == "train":
            return p.parent.parent
    raise FileNotFoundError(
        f"UCI-HAR not found under '{root}'. Run: python data/download.py")


def _read(path: Path) -> np.ndarray:
    return pd.read_csv(path, sep=r"\s+", header=None).to_numpy(dtype=np.float32)


def load_official(root, cache_dir="data/processed"):
    """Returns {'train': (X, y, subj), 'test': (X, y, subj)} with X (N,128,9)."""
    root = find_root(root)
    cache = Path(cache_dir) / "uci_har_raw.npz"
    if cache.exists():
        z = np.load(cache)
        return {s: (z[f"X_{s}"], z[f"y_{s}"], z[f"s_{s}"]) for s in ("train", "test")}
    out = {}
    for split in ("train", "test"):
        sig = root / split / "Inertial Signals"
        chans = [_read(sig / f"{s}_{a}_{split}.txt") for s in SENSORS for a in AXES]
        X = np.stack(chans, axis=-1)                                   # (N,128,9)
        y = _read(root / split / f"y_{split}.txt").astype(int).ravel() - 1
        subj = _read(root / split / f"subject_{split}.txt").astype(int).ravel()
        out[split] = (X, y, subj)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **{f"{k}_{s}": v for s, t in out.items()
                                  for k, v in zip(("X", "y", "s"), t)})
    return out


@dataclass
class Splits:
    X_train: np.ndarray; y_train: np.ndarray; s_train: np.ndarray
    X_val: np.ndarray;   y_val: np.ndarray;   s_val: np.ndarray
    X_test: np.ndarray;  y_test: np.ndarray;  s_test: np.ndarray
    mean: np.ndarray; std: np.ndarray
    channel_idx: list; channel_names: list
    train_subjects: list; val_subjects: list; test_subjects: list


def get_splits(root, channels="all", window_len=FULL_LEN, val_subjects=6,
               split_seed=0, label_fraction=1.0, label_seed=0) -> Splits:
    data = load_official(root)
    Xtr, ytr, str_ = data["train"]
    Xte, yte, ste = data["test"]

    # ---- subject-wise train/val split (fixed seed => identical for every model)
    subs = np.unique(str_)
    val_s = np.sort(np.random.RandomState(split_seed).choice(subs, val_subjects, replace=False))
    is_val = np.isin(str_, val_s)
    Xva, yva, sva = Xtr[is_val], ytr[is_val], str_[is_val]
    Xtr, ytr, str_ = Xtr[~is_val], ytr[~is_val], str_[~is_val]

    # ---- hard leakage guard
    a, b, c = set(np.unique(str_)), set(np.unique(sva)), set(np.unique(ste))
    assert not (a & b) and not (a & c) and not (b & c), "SUBJECT LEAKAGE between splits!"

    # ---- channel selection (ablation) and window cropping (hyper-parameter study)
    idx = CHANNEL_GROUPS[channels] if isinstance(channels, str) else list(channels)
    start = (FULL_LEN - window_len) // 2
    sel = lambda X: np.ascontiguousarray(X[:, start:start + window_len][:, :, idx])
    Xtr, Xva, Xte = sel(Xtr), sel(Xva), sel(Xte)

    # ---- normalisation: statistics from TRAIN ONLY
    mean = Xtr.mean(axis=(0, 1), keepdims=True)
    std = Xtr.std(axis=(0, 1), keepdims=True) + 1e-6
    Xtr, Xva, Xte = [((X - mean) / std).astype(np.float32) for X in (Xtr, Xva, Xte)]

    # ---- optional reduced-label regime (stratified), used for the SSL study
    if label_fraction < 1.0:
        rng = np.random.RandomState(label_seed)
        keep = np.concatenate([
            rng.choice(np.where(ytr == k)[0],
                       max(2, int(round(label_fraction * (ytr == k).sum()))), replace=False)
            for k in np.unique(ytr)])
        keep.sort()
        Xtr, ytr, str_ = Xtr[keep], ytr[keep], str_[keep]

    return Splits(Xtr, ytr, str_, Xva, yva, sva, Xte, yte, ste,
                  mean, std, idx, [CHANNEL_NAMES[i] for i in idx],
                  sorted(map(int, a)), sorted(map(int, b)), sorted(map(int, c)))


class HARDataset(Dataset):
    """Returns (T, C) float tensor and int label. Augmentation is on-the-fly."""
    def __init__(self, X, y, augmenter=None, seed=0):
        self.X, self.y, self.aug = X, y, augmenter
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        import torch
        x = self.X[i]
        if self.aug is not None:
            x = self.aug(x, self.rng)
        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)), int(self.y[i])
