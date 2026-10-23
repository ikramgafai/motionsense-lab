"""Download, extract and load the raw inertial windows of UCI HAR.

The UCI archive is a ZIP that contains a second ZIP (``UCI HAR Dataset.zip``).
We read the inner archive from memory and extract only the whitelisted files we
need (raw inertial signals, labels, subject ids), which also avoids any
path-traversal risk from arbitrary archive members.

Each window is 128 samples at 50 Hz = 2.56 s, with 50% overlap between
consecutive windows of the same recording.
"""

from __future__ import annotations

import hashlib
import io
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

UCI_URL = (
    "https://archive.ics.uci.edu/static/public/240/"
    "human+activity+recognition+using+smartphones.zip"
)
# SHA-256 of the outer archive as downloaded from UCI on 2026-10-23.
UCI_SHA256 = "c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031"
INNER_ZIP = "UCI HAR Dataset.zip"
ROOT = "UCI HAR Dataset"

# Fixed channel order used everywhere in the project.
CHANNELS = [
    "total_acc_x", "total_acc_y", "total_acc_z",
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
]

# Sensor conditions for the ablation. "acc" contains every accelerometer-derived
# channel (body_acc = total_acc minus a 0.3 Hz Butterworth low-pass gravity
# estimate, per the dataset description: derived, not a separate sensor).
# The "*_raw3" conditions are channel-count controls: 3-channel accelerometer
# vs. 3-channel gyroscope, and 6-channel acc+gyro vs. 6-channel acc.
SENSOR_CONDITIONS = {
    "acc": ["total_acc_x", "total_acc_y", "total_acc_z",
            "body_acc_x", "body_acc_y", "body_acc_z"],
    "gyro": ["body_gyro_x", "body_gyro_y", "body_gyro_z"],
    "acc_gyro": list(CHANNELS),
    "acc_raw3": ["total_acc_x", "total_acc_y", "total_acc_z"],
    "acc_raw3_gyro": ["total_acc_x", "total_acc_y", "total_acc_z",
                      "body_gyro_x", "body_gyro_y", "body_gyro_z"],
}

ACTIVITIES = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
              "SITTING", "STANDING", "LAYING"]

WINDOW_SECONDS = 2.56
SAMPLE_RATE_HZ = 50


@dataclass
class HARSplit:
    """One official split. ``X`` is (n, 9, 128) float32 in CHANNELS order."""

    X: np.ndarray
    y: np.ndarray        # int64 labels in 0..5
    subjects: np.ndarray  # int64 subject ids in 1..30


def _members(split: str) -> list[str]:
    files = [f"{ROOT}/{split}/Inertial Signals/{c}_{split}.txt" for c in CHANNELS]
    files += [f"{ROOT}/{split}/y_{split}.txt", f"{ROOT}/{split}/subject_{split}.txt"]
    return files


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(data_dir: Path, verify_hash: bool = True) -> Path:
    """Download the official archive (if missing) and extract needed files."""
    data_dir = Path(data_dir)
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / "uci_har.zip"

    if not archive.exists():
        print(f"Downloading {UCI_URL}")
        tmp = archive.with_suffix(".part")
        with urllib.request.urlopen(UCI_URL, timeout=120) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        tmp.replace(archive)

    digest = _sha256(archive)
    if verify_hash and digest != UCI_SHA256:
        raise RuntimeError(
            f"SHA-256 mismatch for {archive}: got {digest}, expected {UCI_SHA256}. "
            "Delete the file and retry, or pass verify_hash=False if UCI "
            "republished the archive."
        )

    extract_dir = raw_dir / "extracted"
    wanted = _members("train") + _members("test") + [f"{ROOT}/activity_labels.txt"]
    if all((extract_dir / m).exists() for m in wanted):
        return extract_dir

    with zipfile.ZipFile(archive) as outer:
        inner_bytes = outer.read(INNER_ZIP)
    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
        names = set(inner.namelist())
        missing = [m for m in wanted if m not in names]
        if missing:
            raise RuntimeError(f"Inner archive is missing expected files: {missing[:3]}")
        for m in wanted:
            target = extract_dir / m
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(inner.read(m))
    return extract_dir


def _load_split_txt(extract_dir: Path, split: str) -> HARSplit:
    base = extract_dir / ROOT / split
    X = np.stack(
        [np.loadtxt(base / "Inertial Signals" / f"{c}_{split}.txt", dtype=np.float32)
         for c in CHANNELS],
        axis=1,
    )
    y = np.loadtxt(base / f"y_{split}.txt", dtype=np.int64) - 1
    subjects = np.loadtxt(base / f"subject_{split}.txt", dtype=np.int64)
    return HARSplit(X=X, y=y, subjects=subjects)


def load(data_dir: Path, download_if_missing: bool = True) -> tuple[HARSplit, HARSplit]:
    """Return (official_train, official_test), caching a compact .npz."""
    data_dir = Path(data_dir)
    cache = data_dir / "processed" / "uci_har_inertial.npz"
    if cache.exists():
        d = np.load(cache)
        return (HARSplit(d["X_train"], d["y_train"], d["s_train"]),
                HARSplit(d["X_test"], d["y_test"], d["s_test"]))

    extract_dir = data_dir / "raw" / "extracted"
    if not (extract_dir / ROOT).exists():
        if not download_if_missing:
            raise FileNotFoundError(f"No extracted dataset in {extract_dir}")
        extract_dir = download(data_dir)

    train = _load_split_txt(extract_dir, "train")
    test = _load_split_txt(extract_dir, "test")
    validate(train, test)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        X_train=train.X, y_train=train.y, s_train=train.subjects,
        X_test=test.X, y_test=test.y, s_test=test.subjects,
    )
    return train, test


def validate(train: HARSplit, test: HARSplit) -> None:
    """Sanity checks on the official split as documented by the dataset authors."""
    for s in (train, test):
        assert s.X.ndim == 3 and s.X.shape[1:] == (9, 128), s.X.shape
        assert len(s.X) == len(s.y) == len(s.subjects)
        assert set(np.unique(s.y)) == set(range(6))
        assert np.isfinite(s.X).all()
    assert len(train.X) == 7352 and len(test.X) == 2947
    assert not set(train.subjects) & set(test.subjects), "official split overlaps"


def select_channels(X: np.ndarray, condition: str) -> np.ndarray:
    idx = [CHANNELS.index(c) for c in SENSOR_CONDITIONS[condition]]
    return X[:, idx, :]


def subject_split(subjects: np.ndarray, n_val_subjects: int, seed: int
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Split the official training subjects into train/val subject sets.

    Returns boolean masks over windows. No subject appears on both sides.
    """
    uniq = np.unique(subjects)
    rng = np.random.default_rng(seed)
    val_subjects = np.sort(rng.choice(uniq, size=n_val_subjects, replace=False))
    val_mask = np.isin(subjects, val_subjects)
    return ~val_mask, val_mask
