"""Classification metrics, overall and per test subject."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from .data import ACTIVITIES


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    per_class = f1_score(y_true, y_pred, average=None, labels=range(len(ACTIVITIES)))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "per_class_f1": {a: float(f) for a, f in zip(ACTIVITIES, per_class)},
        "confusion": confusion_matrix(y_true, y_pred, labels=range(len(ACTIVITIES))).tolist(),
    }


def per_subject_metrics(y_true: np.ndarray, y_pred: np.ndarray, subjects: np.ndarray) -> dict:
    out = {}
    for s in np.unique(subjects):
        m = subjects == s
        out[int(s)] = {
            "n": int(m.sum()),
            "accuracy": float(accuracy_score(y_true[m], y_pred[m])),
            "macro_f1": float(f1_score(y_true[m], y_pred[m], average="macro",
                                       labels=range(len(ACTIVITIES)), zero_division=0)),
        }
    return out
