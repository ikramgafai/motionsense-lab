"""Additive Gaussian sensor-noise perturbation, applied at test time only.

Noise is added in standardised units, i.e. sigma is a fraction of each
channel's *training-partition* standard deviation. ``target`` restricts the
noise to accelerometer-derived channels, gyroscope channels, or all channels,
which lets us ask whether a fused model degrades when only one sensor does.
"""

from __future__ import annotations

import numpy as np

from .data import SENSOR_CONDITIONS


def channel_mask(condition: str, target: str) -> np.ndarray:
    chans = SENSOR_CONDITIONS[condition]
    if target == "all":
        return np.ones(len(chans), dtype=bool)
    if target == "acc":
        return np.array(["acc" in c for c in chans])
    if target == "gyro":
        return np.array(["gyro" in c for c in chans])
    raise ValueError(target)


def add_noise(X_std: np.ndarray, sigma: float, mask: np.ndarray,
              rng: np.random.Generator) -> np.ndarray:
    noise = rng.standard_normal(X_std.shape).astype(np.float32) * sigma
    noise[:, ~mask, :] = 0.0
    return X_std + noise
