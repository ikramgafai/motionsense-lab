"""Unit tests for the leakage-sensitive parts of the protocol (no download needed)."""

import numpy as np
import pytest
import torch

from motionsense import data as D
from motionsense.experiment import prepare
from motionsense.models import SmallCNN, torch_param_count
from motionsense.noise import add_noise, channel_mask
from motionsense.preprocess import ChannelScaler


def synthetic_split(n_subjects=21, per_subject=20, seed=0, first_id=1):
    rng = np.random.default_rng(seed)
    subjects = np.repeat(np.arange(first_id, first_id + n_subjects), per_subject)
    X = rng.normal(size=(len(subjects), 9, 128)).astype(np.float32)
    y = np.tile(np.arange(6), len(subjects) // 6 + 1)[:len(subjects)]
    return D.HARSplit(X, y, subjects)


def test_subject_split_is_disjoint_and_complete():
    s = synthetic_split().subjects
    tr, val = D.subject_split(s, n_val_subjects=5, seed=0)
    assert not (tr & val).any()
    assert (tr | val).all()
    assert set(s[tr]).isdisjoint(set(s[val]))
    assert len(set(s[val])) == 5


def test_subject_split_is_deterministic():
    s = synthetic_split().subjects
    a = D.subject_split(s, 5, seed=3)[1]
    b = D.subject_split(s, 5, seed=3)[1]
    assert (a == b).all()


def test_scaler_is_fit_on_train_partition_only():
    train = synthetic_split()
    test = synthetic_split(n_subjects=9, seed=1, first_id=100)
    # Make val and test windows wildly different; train-only stats must ignore them.
    cfg = {"split": {"n_val_subjects": 5, "seed": 0}}
    tr_mask, val_mask = D.subject_split(train.subjects, 5, 0)
    train.X[val_mask] += 1000.0
    test.X += -500.0
    p = prepare(cfg, train, test, "acc_gyro")
    expected_mean = train.X[tr_mask].mean(axis=(0, 2))
    np.testing.assert_allclose(p["scaler"].mean.ravel(), expected_mean, rtol=1e-4)
    assert abs(p["X_tr"].mean()) < 1e-3
    assert p["X_val"].mean() > 100  # val is transformed, never fitted
    assert set(p["s_tr"]).isdisjoint(set(p["s_val"]))
    assert set(p["s_tr"]).isdisjoint(set(p["s_te"]))


def test_scaler_requires_fit():
    with pytest.raises(RuntimeError):
        ChannelScaler().transform(np.zeros((1, 3, 128), dtype=np.float32))


@pytest.mark.parametrize("cond,n", [("acc", 6), ("gyro", 3), ("acc_gyro", 9),
                                    ("acc_raw3", 3), ("acc_raw3_gyro", 6)])
def test_channel_selection(cond, n):
    X = np.arange(9)[None, :, None] * np.ones((2, 9, 128))
    Xs = D.select_channels(X, cond)
    assert Xs.shape == (2, n, 128)
    expected = [D.CHANNELS.index(c) for c in D.SENSOR_CONDITIONS[cond]]
    assert Xs[0, :, 0].tolist() == expected


def test_gyro_condition_contains_no_accelerometer_channel():
    assert all("gyro" in c for c in D.SENSOR_CONDITIONS["gyro"])
    assert not any("gyro" in c for c in D.SENSOR_CONDITIONS["acc"])


@pytest.mark.parametrize("c_in", [3, 6, 9])
def test_cnn_shapes(c_in):
    model = SmallCNN(c_in)
    out = model(torch.zeros(4, c_in, 128))
    assert out.shape == (4, 6)
    assert torch_param_count(model) < 50_000


def test_noise_targets_only_selected_channels():
    X = np.zeros((10, 9, 128), dtype=np.float32)
    rng = np.random.default_rng(0)
    Xn = add_noise(X, 0.5, channel_mask("acc_gyro", "gyro"), rng)
    assert np.all(Xn[:, :6] == 0)
    assert np.std(Xn[:, 6:]) == pytest.approx(0.5, rel=0.05)
    assert channel_mask("gyro", "acc").sum() == 0


def test_holm_adjustment():
    from motionsense.analysis import holm
    adj = holm([0.01, 0.04, 0.03])
    assert adj == pytest.approx([0.03, 0.06, 0.06])
