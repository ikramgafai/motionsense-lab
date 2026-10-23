"""Per-channel standardisation fitted on the training partition only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ChannelScaler:
    """Standardise each channel with a mean/std pooled over windows and time.

    Pooling over time (rather than per timestep) keeps the scaler independent
    of window phase and matches what a streaming deployment could apply.
    """

    mean: np.ndarray | None = None  # (C, 1)
    std: np.ndarray | None = None   # (C, 1)
    eps: float = 1e-8

    def fit(self, X: np.ndarray) -> "ChannelScaler":
        self.mean = X.mean(axis=(0, 2), keepdims=True)[0].astype(np.float32)
        self.std = X.std(axis=(0, 2), keepdims=True)[0].astype(np.float32)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None:
            raise RuntimeError("ChannelScaler.transform called before fit")
        return ((X - self.mean) / (self.std + self.eps)).astype(np.float32)

    def to_dict(self) -> dict:
        return {"mean": self.mean.ravel().tolist(), "std": self.std.ravel().tolist()}
