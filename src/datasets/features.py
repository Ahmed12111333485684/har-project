"""Hand-crafted features for the classical baseline (vectorised over windows)."""
from __future__ import annotations
import numpy as np
from scipy.stats import kurtosis, skew


def extract_features(X: np.ndarray) -> np.ndarray:
    """X: (N, T, C) -> (N, F). Time stats + FFT + inter-axis correlation + magnitude."""
    N, T, C = X.shape
    f = [X.mean(1), X.std(1), X.min(1), X.max(1), np.median(X, 1),
         np.percentile(X, 75, axis=1) - np.percentile(X, 25, axis=1),
         (X ** 2).mean(1), skew(X, axis=1), kurtosis(X, axis=1)]
    F = np.abs(np.fft.rfft(X, axis=1)) / T
    F[:, 0, :] = 0.0                                        # drop DC component
    p = F ** 2 / (np.sum(F ** 2, axis=1, keepdims=True) + 1e-12)
    f += [np.argmax(F, axis=1).astype(np.float32),          # dominant frequency bin
          np.sum(F ** 2, axis=1),                           # spectral energy
          -np.sum(p * np.log(p + 1e-12), axis=1)]           # spectral entropy
    for s in range(C // 3):                                 # per 3-axis sensor
        t = X[:, :, 3 * s:3 * s + 3]
        tc = t - t.mean(1, keepdims=True)
        for i, j in ((0, 1), (0, 2), (1, 2)):
            num = (tc[:, :, i] * tc[:, :, j]).sum(1)
            den = np.sqrt((tc[:, :, i] ** 2).sum(1) * (tc[:, :, j] ** 2).sum(1)) + 1e-8
            f.append((num / den)[:, None])
        mag = np.sqrt((t ** 2).sum(2))
        f += [mag.mean(1, keepdims=True), mag.std(1, keepdims=True),
              np.abs(t).sum(2).mean(1, keepdims=True)]       # signal magnitude area
    return np.concatenate(f, axis=1).astype(np.float32)
