#!/usr/bin/env python
"""Build the VitalDB pretraining corpus: 10 s, 125 Hz, filtered PPG segments.

Input
-----
A directory of ``.vital`` files from VitalDB 1.0.0 (https://vitaldb.net).
Only the ``SNUADC/PLETH`` PPG track is kept; the five Solar8000 numeric tracks
are read to trim each recording to the span they all cover and to drop
segments with no clinical reading.

Output
------
``<out>/<subject>.npy``: float32, shape ``(N, 1250)``, one row per 10 s
segment, in temporal order.

Pipeline (per recording)
------------------------
1. read the tracks: ``SNUADC/PLETH`` at 125 Hz, the numeric tracks at 0.5 Hz;
2. trim to the time span common to every track;
3. take the continuous stretches without NaN that are at least 10 s long;
4. filter each stretch with pyPPG
   ``Preprocess(fL=0.5, fH=12, order=4, sm_wins={'ppg': 50, ...})``;
5. cut non-overlapping 10 s segments; drop those that are all-NaN, that have
   neither HR nor SpO2 recorded, or that are more than 25 % flatline.

Requires Python 3.10 with the optional packages pinned in
requirements-corpus.txt (``pyPPG==1.0.73``, ``vitaldb==1.4.12``, ``dotmap``);
their pins conflict with the training environment, so install them separately.

Usage
-----
    python dataset/build_vital_ppg.py --vital-dir /data/vital/1.0.0/vital_files \\
        --out dataset/VITALDB_10s
"""

import argparse
import os

import numpy as np

NUMERIC_TRACKS = [
    "Solar8000/ART_DBP",
    "Solar8000/ART_MBP",
    "Solar8000/ART_SBP",
    "Solar8000/HR",
    "Solar8000/PLETH_SPO2",
]
PPG_TRACK = "SNUADC/PLETH"
NUMERIC_FS = 0.5

FS = 125
SEG_S = 10
SEG_LEN = FS * SEG_S

FLAT_TIME_S = 2
FLAT_RATIO = 0.25
FLAT_CHANGE = 0.01

_prep = None


def _get_prep():
    """pyPPG filter bank, built once (module-level in the corpus that shipped)."""
    global _prep
    if _prep is None:
        import pyPPG.preproc as PP
        _prep = PP.Preprocess(fL=0.5, fH=12, order=4,
                              sm_wins={"ppg": 50, "vpg": 10,
                                       "apg": 10, "jpg": 10})
    return _prep


def read_tracks(vital_path):
    """Read every track once.  Returns ``{track: (timestamps, values)}``."""
    import vitaldb
    vf = vitaldb.VitalFile(vital_path)
    tracks = {}
    for name in NUMERIC_TRACKS + [PPG_TRACK]:
        interval = 1.0 / NUMERIC_FS if name in NUMERIC_TRACKS else 1.0 / FS
        arr = vf.to_numpy([name], interval, return_timestamp=True)
        ts, vals = arr[:, 0].astype(float), arr[:, 1].astype(float)
        order = np.argsort(ts)
        tracks[name] = (ts[order], vals[order])
    return tracks


def valid_runs(vals):
    """``[(start, end)]`` of consecutive non-NaN stretches of at least 10 s."""
    runs, start = [], None
    for i, ok in enumerate(~np.isnan(vals)):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            if i - start >= SEG_LEN:
                runs.append((start, i))
            start = None
    if start is not None and len(vals) - start >= SEG_LEN:
        runs.append((start, len(vals)))
    return runs


def filter_run(run):
    """pyPPG band-pass (0.5-12 Hz) + smoothing; returns the filtered PPG."""
    from dotmap import DotMap
    signal = DotMap()
    signal.v = run
    signal.fs = FS
    signal.filtering = True
    ppg, _vpg, _apg, _jpg = _get_prep().get_signals(signal)
    return np.asarray(ppg)


def flatline_segments(seg, min_duration, change_threshold):
    """Boundaries of the stretches that change by less than the threshold.

    Same scan as ``biobss.sqatools.detect_flatline_segments``; inlined so the
    corpus build does not need biobss (and its plotting dependencies).
    """
    hi, lo = seg.max(), seg.min()
    starts, ends, in_flat = [], [], False
    for i in range(1, len(seg)):
        change = abs(seg[i] - seg[i - 1])
        if change <= change_threshold and seg[i] != hi and seg[i] != lo:
            if not in_flat:
                starts.append(i - 1)
                in_flat = True
        elif in_flat:
            ends.append(i - 1)
            in_flat = False
    if in_flat:
        ends.append(len(seg) - 1)
    return [(s, e) for s, e in zip(starts, ends) if e - s + 1 >= min_duration]


def is_flat(seg):
    """True if more than 25 % of the segment is detected as flatline."""
    x = np.asarray(seg)
    x = x[np.isfinite(x)]
    scale = float(np.percentile(x, 95) - np.percentile(x, 5)) if x.size >= 2 else 0.0
    segments = flatline_segments(seg, min_duration=FLAT_TIME_S * FS,
                                 change_threshold=FLAT_CHANGE * scale)
    flat = np.sum([end - start for start, end in segments])
    return flat / SEG_LEN > FLAT_RATIO


def numeric_grid(tracks, t0, t1):
    """Resample the numeric tracks onto the common 0.5 Hz grid."""
    import pandas as pd
    step = 1.0 / NUMERIC_FS
    grid = pd.DataFrame(
        {"t": np.arange(np.floor(t0), np.ceil(t1) + step, step)})
    for name in NUMERIC_TRACKS:
        ts, vals = tracks[name]
        keep = (ts >= t0) & (ts <= t1)
        if not keep.any():
            continue
        track = pd.DataFrame({"t": ts[keep], name: vals[keep]}).sort_values("t")
        merged = pd.merge_asof(grid, track, on="t", direction="nearest",
                               tolerance=step / 2)
        grid[name] = merged[name]
    return grid


def build_recording(vital_path):
    """Return the float32 array ``(N, 1250)`` for one recording."""
    tracks = read_tracks(vital_path)
    t0 = max(ts[0] for ts, _ in tracks.values() if len(ts))
    t1 = min(ts[-1] for ts, _ in tracks.values() if len(ts))

    grid = numeric_grid(tracks, t0, t1)
    hr = grid["Solar8000/HR"].to_numpy() if "Solar8000/HR" in grid \
        else np.full(len(grid), np.nan)
    spo2 = grid["Solar8000/PLETH_SPO2"].to_numpy() \
        if "Solar8000/PLETH_SPO2" in grid else np.full(len(grid), np.nan)
    times = grid["t"].to_numpy()

    ts_ppg, vals_ppg = tracks[PPG_TRACK]
    keep = (ts_ppg >= t0) & (ts_ppg <= t1)
    ts_ppg, vals_ppg = ts_ppg[keep], vals_ppg[keep]

    segments = []
    for start, end in valid_runs(vals_ppg):
        run_ts = ts_ppg[start:end]
        ppg = filter_run(vals_ppg[start:end])
        for i in range(len(ppg) // SEG_LEN):
            seg = ppg[i * SEG_LEN:(i + 1) * SEG_LEN]
            if np.all(np.isnan(seg)):
                continue

            seg_t0 = float(run_ts[0]) + i * SEG_S
            window = (times >= seg_t0) & (times < seg_t0 + SEG_S)
            # skip a segment carrying neither a heart rate nor an SpO2 reading
            if np.all(np.isnan(hr[window])) and np.all(np.isnan(spo2[window])):
                continue

            if is_flat(seg):
                continue
            segments.append(np.asarray(seg, dtype=np.float32))

    if not segments:
        return np.zeros((0, SEG_LEN), dtype=np.float32)
    return np.stack(segments)


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vital-dir", required=True,
                   help="directory of .vital files")
    p.add_argument("--out", required=True,
                   help="output directory for <subject>.npy")
    p.add_argument("--skip-existing", action="store_true",
                   help="leave subject files that already exist untouched")
    p.add_argument("--limit", type=int, default=None,
                   help="process at most this many recordings (debugging)")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    files = sorted(f for f in os.listdir(args.vital_dir)
                   if f.endswith(".vital"))
    if args.limit is not None:
        files = files[:args.limit]

    n_total = n_kept = 0
    for k, name in enumerate(files, 1):
        subject = os.path.splitext(name)[0]
        out_path = os.path.join(args.out, f"{subject}.npy")
        if args.skip_existing and os.path.exists(out_path):
            continue
        segs = build_recording(os.path.join(args.vital_dir, name))
        np.save(out_path, segs)
        n_total += 1
        n_kept += len(segs)
        print(f"[{k}/{len(files)}] {subject}: {len(segs)} segments")

    print(f"done: {n_total} recordings, {n_kept} segments -> {args.out}")


if __name__ == "__main__":
    main()
