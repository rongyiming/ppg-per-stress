"""Lazy dataset over the VitalDB segment corpus used for pretraining.

The corpus is built by ``dataset/build_vital_ppg.py``: one ``.npy`` per
subject, ``(N, 1250)`` float32, every row a filtered 10 s / 125 Hz segment.

The pretext task regresses each window to ITS OWN ten handcrafted descriptors
(``new_individual`` over that same window), not to a subject-level baseline.
Inputs are z-scored per segment; labels are min-max normalised with statistics
taken from the training subjects only.  Train/validation are split by subject.
"""

import functools
import os

import numpy as np
import torch
from joblib import Parallel, delayed
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data.individual import new_individual

FS = 125
SEG_LEN = FS * 10


def load_corpus(root, min_segments=50):
    """``{subject: (path, n_segments)}`` for subjects with enough segments."""
    corpus = {}
    for name in sorted(os.listdir(root)):
        if not name.endswith(".npy"):
            continue
        path = os.path.join(root, name)
        n = int(np.load(path, mmap_mode="r").shape[0])
        if n < min_segments:
            continue
        corpus[name[:-4]] = (path, n)
    return corpus


def split_subjects(corpus, valid_ratio=0.15, seed=42):
    """Subject-independent train/validation split."""
    subjects = sorted(corpus)
    rng = np.random.RandomState(seed)
    rng.shuffle(subjects)
    n_valid = int(round(len(subjects) * valid_ratio))
    return set(subjects[n_valid:]), set(subjects[:n_valid])


def subject_rows(n_segments, cap):
    """Evenly spaced row indices, at most ``cap`` per subject."""
    if cap is not None and n_segments > cap:
        return np.linspace(0, n_segments - 1, cap).astype(int)
    return np.arange(n_segments)


def normalize_features(feats, mins, maxs):
    """Min-max normalise each dimension with the given (training) statistics."""
    denom = maxs - mins
    out = feats.astype(np.float64).copy()
    mask = denom > 0
    out[:, mask] = (out[:, mask] - mins[mask]) / denom[mask]
    out[:, ~mask] = 0.0
    return out.astype(np.float32)


def _subject_features(path, rows):
    arr = np.load(path, mmap_mode="r")
    return np.stack([
        np.array(new_individual(np.asarray(arr[i], dtype=np.float64), FS),
                 dtype=np.float32)
        for i in rows])


def compute_features(corpus, subjects, cap, cache_dir, n_jobs=16):
    """``{subject: (N, 10) features}``, cached per subject and cap."""
    os.makedirs(cache_dir, exist_ok=True)

    def one(subject):
        path, n = corpus[subject]
        cache = os.path.join(cache_dir, f"{subject}_c{cap}.npy")
        if os.path.exists(cache):
            return subject, np.load(cache)
        feats = _subject_features(path, subject_rows(n, cap))
        np.save(cache, feats)
        return subject, feats

    out = Parallel(n_jobs=n_jobs)(delayed(one)(s)
                                  for s in tqdm(sorted(subjects),
                                                desc="  segment features",
                                                leave=False))
    return dict(out)


@functools.lru_cache(maxsize=64)
def _load_subject(path):
    return np.load(path, mmap_mode="r")


class PretrainDataset(Dataset):
    """``(z-scored segment, normalised descriptors)`` pairs."""

    def __init__(self, paths, rows, features):
        self.paths = paths
        self.rows = np.asarray(rows, dtype=np.int64)
        self.features = features

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        sig = np.asarray(_load_subject(self.paths[idx])[self.rows[idx]],
                         dtype=np.float32)
        mean = float(sig.mean())
        std = float(sig.std())
        sig = (sig - mean) / std if std > 1e-8 else sig - mean
        return torch.from_numpy(sig), torch.from_numpy(self.features[idx])


def build_samples(corpus, subjects, cap):
    """Flatten subjects into ``(paths, rows)`` in subject order."""
    paths, rows = [], []
    for subject in sorted(subjects):
        path, n = corpus[subject]
        for row in subject_rows(n, cap):
            paths.append(path)
            rows.append(row)
    return paths, rows


def make_dataloaders(corpus, train_subjects, valid_subjects, cap,
                     batch_size=512, num_workers=8, n_jobs=16,
                     cache_dir="dataset"):
    """Training and validation loaders for the pretraining stage."""
    train_feats = compute_features(corpus, train_subjects, cap, cache_dir,
                                   n_jobs)
    valid_feats = compute_features(corpus, valid_subjects, cap, cache_dir,
                                   n_jobs)

    mins = np.concatenate([train_feats[s] for s in sorted(train_subjects)]
                          ).min(axis=0)
    maxs = np.concatenate([train_feats[s] for s in sorted(train_subjects)]
                          ).max(axis=0)

    train_paths, train_rows = [], []
    train_labels = []
    for subject in sorted(train_subjects):
        path, n = corpus[subject]
        for row in subject_rows(n, cap):
            train_paths.append(path)
            train_rows.append(row)
        train_labels.append(train_feats[subject])
    train_labels = normalize_features(np.concatenate(train_labels), mins, maxs)

    valid_paths, valid_rows, valid_labels = [], [], []
    for subject in sorted(valid_subjects):
        path, n = corpus[subject]
        for row in subject_rows(n, cap):
            valid_paths.append(path)
            valid_rows.append(row)
        valid_labels.append(valid_feats[subject])
    valid_labels = normalize_features(np.concatenate(valid_labels), mins, maxs)

    train_ds = PretrainDataset(train_paths, train_rows, train_labels)
    valid_ds = PretrainDataset(valid_paths, valid_rows, valid_labels)

    common = dict(batch_size=batch_size, num_workers=num_workers,
                  pin_memory=True)
    if num_workers > 0:
        common["persistent_workers"] = True
        common["prefetch_factor"] = 2
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **common)
    valid_loader = DataLoader(valid_ds, shuffle=False, **common)
    return train_loader, valid_loader, len(train_ds), len(valid_ds)
