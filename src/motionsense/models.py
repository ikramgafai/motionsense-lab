"""Models: a flattened-window logistic regression baseline and a small 1D CNN."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn


def make_logreg(C: float, max_iter: int, seed: int) -> LogisticRegression:
    # lbfgs on a convex objective: the seed only matters for tie-breaking.
    return LogisticRegression(C=C, max_iter=max_iter, random_state=seed)


def flatten(X: np.ndarray) -> np.ndarray:
    return X.reshape(len(X), -1)


def logreg_param_count(model: LogisticRegression) -> int:
    return int(model.coef_.size + model.intercept_.size)


class SmallCNN(nn.Module):
    """Three conv blocks, global average pooling, linear head.

    Global pooling makes the classifier shift-tolerant within the 2.56 s window,
    which a flattened linear model cannot be.
    """

    def __init__(self, in_channels: int, n_classes: int = 6,
                 widths: tuple[int, ...] = (32, 64, 64),
                 kernels: tuple[int, ...] = (7, 5, 3), dropout: float = 0.3):
        super().__init__()
        assert len(widths) == len(kernels)
        layers: list[nn.Module] = []
        c_in = in_channels
        for i, (w, k) in enumerate(zip(widths, kernels)):
            layers += [nn.Conv1d(c_in, w, k, padding=k // 2, bias=False),
                       nn.BatchNorm1d(w), nn.ReLU(inplace=True)]
            if i < len(widths) - 1:
                layers.append(nn.MaxPool1d(2))
            c_in = w
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(c_in, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.pool(self.features(x)).squeeze(-1)
        return self.head(self.dropout(z))


def torch_param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
