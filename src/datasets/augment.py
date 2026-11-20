"""
Time-series augmentations written from scratch (NumPy + a cubic spline).
They operate on ONE z-scored window x of shape (T, C).
(Sigmas are therefore in "standard deviations" of each channel.)
"""
from __future__ import annotations
import numpy as np
from scipy.interpolate import CubicSpline


def jitter(x, rng, sigma=0.05):
    """Additive Gaussian sensor noise."""
    return x + rng.normal(0.0, sigma, x.shape)


def scaling(x, rng, sigma=0.1):
    """Random per-channel gain (different sensor sensitivity / strength of movement)."""
    return x * rng.normal(1.0, sigma, (1, x.shape[1]))


def magnitude_warp(x, rng, sigma=0.2, knots=4):
    """Smoothly time-varying gain, one smooth curve per channel."""
    T, C = x.shape
    knot_pos = np.linspace(0, T - 1, knots + 2)
    curve = CubicSpline(knot_pos, rng.normal(1.0, sigma, (knots + 2, C)))(np.arange(T))
    return x * curve


def time_warp(x, rng, sigma=0.2, knots=4):
    """Smoothly speed up / slow down the movement (different walking cadence)."""
    T, C = x.shape
    knot_pos = np.linspace(0, T - 1, knots + 2)
    speed = CubicSpline(knot_pos, rng.normal(1.0, sigma, knots + 2))(np.arange(T))
    speed = np.clip(speed, 0.1, None)
    cum = np.cumsum(speed)
    new_t = (cum - cum[0]) / (cum[-1] - cum[0]) * (T - 1)      # monotone, 0 .. T-1
    grid = np.arange(T)
    return np.stack([np.interp(new_t, grid, x[:, c]) for c in range(C)], axis=1)


AUGMENTATIONS = {"jitter": jitter, "scaling": scaling,
                 "magnitude_warp": magnitude_warp, "time_warp": time_warp}


class Augmenter:
    """Applies each named augmentation independently with probability p."""
    def __init__(self, names, p=0.5):
        if isinstance(names, str):
            names = [n.strip() for n in names.strip("[]").split(",") if n.strip()]
        self.fns = [AUGMENTATIONS[n] for n in names]
        self.p = p

    def __call__(self, x, rng):
        for fn in self.fns:
            if rng.random() < self.p:
                x = fn(x, rng)
        return x.astype(np.float32)


def build_augmenter(cfg):
    names = cfg.get("names") or []
    return Augmenter(names, cfg.get("p", 0.5)) if names else None
