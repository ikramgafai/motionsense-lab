"""Checks against the real dataset; skipped if it has not been downloaded yet."""

from pathlib import Path

import numpy as np
import pytest

from motionsense import data as D

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
pytestmark = pytest.mark.skipif(
    not (DATA_DIR / "processed" / "uci_har_inertial.npz").exists(),
    reason="run `python -m motionsense download` first",
)


@pytest.fixture(scope="module")
def splits():
    return D.load(DATA_DIR, download_if_missing=False)


def test_official_split_shapes_and_disjointness(splits):
    train, test = splits
    D.validate(train, test)
    assert len(np.unique(train.subjects)) == 21
    assert len(np.unique(test.subjects)) == 9


def test_validation_subjects_come_from_training_subjects(splits):
    train, test = splits
    tr, val = D.subject_split(train.subjects, 5, seed=0)
    assert set(train.subjects[val]) <= set(train.subjects)
    assert set(train.subjects[val]).isdisjoint(test.subjects)
    assert set(train.subjects[val]).isdisjoint(train.subjects[tr])


def test_units_are_physical(splits):
    """total_acc is in g: gravity dominates, so mean magnitude ~1 g."""
    train, _ = splits
    mag = np.linalg.norm(train.X[:, 0:3, :], axis=1).mean()
    assert 0.9 < mag < 1.1
