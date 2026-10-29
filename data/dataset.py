"""
Dataset and DataLoader utilities for 125 Hz PPG stress classification.

Pieces:
  - NewDataset: tuple-based Dataset wrapper
  - load_dataset_pkl: load a 125 Hz segment file
  - build_segments: sliding-window segment builder (filtering + enrolment)
  - create_finetune_dataloaders: subject-independent split + DataLoader
"""

import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple

from data.preprocessing import preprocess_ppg, calculate_hr_hrv
from data.individual import new_individual


# ═══════════════════════════════════════════════════════════════════════════════
# NewDataset — tuple-based Dataset wrapper
# ═══════════════════════════════════════════════════════════════════════════════

class NewDataset(Dataset):
    """Return one window plus its label and per-subject descriptor.

    Pretrain mode:  (segment, label, dlabel)
    Finetune mode:  (segment, label, dlabel, individual [, subject_id])
    """

    def __init__(self, data, labels, dlabels, finetune=False, individual=None,
                 seg_subject_ids=None):
        self.data = torch.from_numpy(np.asarray(data, dtype=np.float32))
        if finetune:
            self.labels = torch.from_numpy(np.asarray(labels, dtype=np.int64))
            self.individual = torch.from_numpy(
                np.asarray(individual, dtype=np.float32))
        else:
            self.labels = torch.from_numpy(np.asarray(labels, dtype=np.float32))
            _max = self.labels.max(dim=0, keepdim=True)
            _min = self.labels.min(dim=0, keepdim=True)
            self.labels = (self.labels - _min.values) / (_max.values - _min.values)
            self.individual = None
        self.dlabels = torch.from_numpy(np.asarray(dlabels, dtype=np.float32))
        self.seg_subject_ids = seg_subject_ids  # list[str] parallel to data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        dlabel = self.dlabels[idx]
        if self.individual is not None and self.seg_subject_ids is not None:
            return sample, label, dlabel, self.individual[idx], self.seg_subject_ids[idx]
        if self.individual is not None:
            return sample, label, dlabel, self.individual[idx]
        return sample, label, dlabel


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading and window building
# ═══════════════════════════════════════════════════════════════════════════════

def load_dataset_pkl(file_path: str) -> dict:
    """Load a 125 Hz segment file.

    Format: ``{subject: {"data": [class0, class1], "label": [0, 1]}}``.
    """
    with open(file_path, "rb") as f:
        return pickle.load(f)


def build_segments(raw_data: dict, fs: int, window_s: int, overlap: float,
                   filter_order: int = 4, filter_type: str = 'cheby',
                   enrollment_s: int = 120,
                   ) -> dict:
    """Build two window sets per subject + per-subject individual features.

    Both sets are built so that either can be used:
      - adaptive windows: per-class stride chosen to balance the two classes.
      - fixed windows:    stride = window_len * (1 - overlap), label-independent.
    create_finetune_dataloaders() decides which set feeds train/val/test.

    The enrollment segment (tail of the non-stress block, used to compute the
    10-D individual feature) is trimmed off before windowing so it never enters
    any classification window.

    Args:
        raw_data: {subject: {'data': [block0, block1], 'label': [0, 1]}}.
        fs: Sampling rate (Hz).
        window_s: Window duration (seconds).
        overlap: Overlap ratio (0-1).
        filter_order / filter_type: per-window bandpass settings.
        enrollment_s: Enrolment (resting) segment length in seconds.
            Tail of the first block, used for the 10-D individual feature and
            trimmed off before windowing.

    Returns dict:
        adaptive_segments: list of (ppg_1d, label) — class-balanced stride
        adaptive_subjects: list of subject ids, parallel to adaptive_segments
        fixed_segments:    list of (ppg_1d, label) — fixed stride
        fixed_subjects:    list of subject ids, parallel to fixed_segments
        individual_feats:  {subject_id: 10-dim float32 array}
    """
    individual_len = enrollment_s * fs   # enrollment_s seconds baseline

    adaptive_segments, adaptive_subjects = [], []
    fixed_segments, fixed_subjects = [], []
    individual_feats = {}

    for subject_id in sorted(raw_data.keys()):
        b = raw_data[subject_id]
        data_segs = b['data']
        label_segs = list(b['label'])

        # Individual feature: tail of the first (non-stress) block.
        first_seg = np.asarray(data_segs[0], dtype=np.float64)
        n_enroll = min(individual_len, len(first_seg))
        baseline = first_seg[-n_enroll:]
        baseline_clean = preprocess_ppg(baseline, fs, order=0)  # poly-detrend only
        individual_feats[subject_id] = np.array(
            new_individual(baseline_clean, fs), dtype=np.float32)

        # Trim the enrollment segment off the first block so it cannot
        # appear in any classification window (train, val, or test).
        data_trimmed = [first_seg[:len(first_seg) - n_enroll]] + data_segs[1:]

        # Window the trimmed blocks once per method.
        for (segments_out, subjects_out, use_balance) in (
                (adaptive_segments, adaptive_subjects, True),
                (fixed_segments, fixed_subjects, False)):
            for (seg, lbl) in _window_blocks(
                    data_trimmed, label_segs, fs, window_s, overlap,
                    balance_classes=use_balance, filter_order=filter_order,
                    filter_type=filter_type):
                segments_out.append((seg, lbl))
                subjects_out.append(subject_id)

    return {
        'adaptive_segments': adaptive_segments,
        'adaptive_subjects': adaptive_subjects,
        'fixed_segments': fixed_segments,
        'fixed_subjects': fixed_subjects,
        'individual_feats': individual_feats,
    }


def _adaptive_class_steps(data_segs, label_segs, window_len, step):
    """Per-class stride so that stress/non-stress yield ~equal window counts."""
    class_lens = {}
    for seg, lbl in zip(data_segs, label_segs):
        class_lens[lbl] = class_lens.get(lbl, 0) + len(np.asarray(seg, dtype=np.float64))
    if len(class_lens) < 2:
        return {lbl: step for lbl in class_lens}
    total_both = sum(class_lens.values())
    N = max(1, int(1 + (total_both - 2 * window_len) / step))
    class_steps = {}
    for lbl, sig_len in class_lens.items():
        class_steps[lbl] = max(1, int((sig_len - window_len) / max(1, N - 1)))
    return class_steps


def _window_blocks(data_segs, label_segs, fs, window_s, overlap,
                   balance_classes, filter_order, filter_type):
    """Slide windows over a subject's (enrollment-trimmed) blocks.

    balance_classes=True → adaptive per-class stride;
    balance_classes=False → fixed stride.
    Each window is band-pass filtered, detrended and checked for a usable heart
    rate before being kept.
    """
    window_len = window_s * fs
    step = int(window_len * (1 - overlap))
    min_data = window_len + fs + 1

    out = []
    if balance_classes:
        class_steps = _adaptive_class_steps(data_segs, label_segs, window_len, step)
    else:
        class_steps = {}

    for seg, label in zip(data_segs, label_segs):
        raw_signal = np.asarray(seg, dtype=np.float64)
        cls_step = class_steps.get(label, step) if balance_classes else step

        if len(raw_signal) < min_data:
            continue

        for k in range(0, len(raw_signal) - min_data + 1, cls_step):
            segment = raw_signal[k:k + window_len]
            segment_clean = preprocess_ppg(segment, fs, order=filter_order,
                                           filter_type=filter_type)

            # Heart-rate plausibility filter.
            hr = calculate_hr_hrv(segment_clean, fs)
            if hr['n_beats'] is None or hr['hr_bpm'] is None:
                continue
            # 40 / 60 * window_s bpm is the heart rate a single beat would
            # imply; anything at or below it means the detector found noise.
            if hr['hr_bpm'] <= 40 / 60 * window_s:
                continue

            out.append((segment_clean.astype(np.float32), label))

    return out


def create_finetune_dataloaders(bundle, fs, batch_size, seed=42,
                                balance_classes=False, fold_idx=-1,
                                global_ind_norm=False,
                                train_ratio=0.7, val_ratio=0.15,
                                ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders with subject-independent split.

    All splits (train/val/test) use the same window set: adaptive
    (class-balanced) stride when balance_classes=True, else fixed stride.

    Subject split:
      fold_idx >= 0: deterministic 5-fold CV (test=fold, val=(fold+1)%5).
      fold_idx < 0 : random 70/15/15 split seeded by `seed`.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    individual_feats = bundle['individual_feats']
    subjects = sorted(individual_feats.keys())
    n_subj = len(subjects)

    if fold_idx >= 0:
        # ── 5-fold CV: deterministic shuffle, cyclic fold assignment ──
        n_folds = 5
        rng = np.random.RandomState(42)
        shuffled = np.array(subjects)[rng.permutation(n_subj)]
        # Assign subjects to folds cyclically: [0,1,2,3,4,0,1,...]
        fold_of = {s: int(i % n_folds) for i, s in enumerate(shuffled)}

        test_fold = fold_idx % n_folds
        val_fold = (fold_idx + 1) % n_folds
        train_folds = {f for f in range(n_folds) if f not in (test_fold, val_fold)}

        test_subjs = {s for s, f in fold_of.items() if f == test_fold}
        val_subjs = {s for s, f in fold_of.items() if f == val_fold}
        train_subjs = {s for s, f in fold_of.items() if f in train_folds}
    else:
        # ── Random subject split ──
        rng = np.random.RandomState(seed)
        rng.shuffle(subjects)
        n_train = int(n_subj * train_ratio)
        n_val = int(n_subj * val_ratio)
        train_subjs = set(subjects[:n_train])
        val_subjs = set(subjects[n_train:n_train + n_val])
        test_subjs = set(subjects[n_train + n_val:])

    # ── Individual feature normalization ──────────────────────────────────
    if global_ind_norm:
        # Global min-max over ALL subjects, including the held-out folds.
        # It lets the held-out subjects' descriptor ranges shape the scaling,
        # so it is transductive.  Kept only for the earlier runs; the default
        # is the per-fold (training-statistics-only) branch below.
        all_inds = np.array([individual_feats[s] for s in subjects])
        min_vals = all_inds.min(axis=0)
        max_vals = all_inds.max(axis=0)
    else:
        # Per-fold: train-set statistics only.
        train_inds = np.array([individual_feats[s] for s in train_subjs])
        min_vals = train_inds.min(axis=0)
        max_vals = train_inds.max(axis=0)

    def _normalize_individual(ind_vec):
        out = ind_vec.copy()
        denom = max_vals - min_vals
        mask = denom > 0
        out[mask] = (out[mask] - min_vals[mask]) / denom[mask]
        out[~mask] = 0.0
        return out

    normalized_feats = {s: _normalize_individual(np.array(individual_feats[s]))
                        for s in subjects}

    def _build_dataset(segments_list, subjects_list, subj_set):
        seg_data, seg_labels, inds, sids = [], [], [], []
        for (seg, lbl), sid in zip(segments_list, subjects_list):
            if sid in subj_set:
                seg_data.append(seg)
                seg_labels.append(lbl)
                inds.append(normalized_feats[sid])
                sids.append(sid)
        dl = [np.zeros(fs, dtype=np.float32) for _ in seg_data]  # placeholder dlabels
        return NewDataset(seg_data, seg_labels, dl, finetune=True,
                          individual=inds, seg_subject_ids=sids)

    # One window set, shared by all three splits.
    if balance_classes:
        segs, subjs = bundle['adaptive_segments'], bundle['adaptive_subjects']
    else:
        segs, subjs = bundle['fixed_segments'], bundle['fixed_subjects']

    train_ds = _build_dataset(segs, subjs, train_subjs)
    val_ds = _build_dataset(segs, subjs, val_subjs)
    test_ds = _build_dataset(segs, subjs, test_subjs)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    print(f"    {len(train_subjs)} / {len(val_subjs)} / {len(test_subjs)} subjects")
    print(f"    {len(train_ds)} / {len(val_ds)} / {len(test_ds)} segments")

    return train_loader, val_loader, test_loader
