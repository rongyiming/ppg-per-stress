#!/usr/bin/env python
"""Convert the raw CATSA recordings into the pipeline's 125 Hz segment file.

Input
-----
``<src>/Sub*/{Baseline,Stroop,Logic,Sudoku}/BVP.csv`` -- Empatica E4 BVP at
64 Hz.  Download CATSA from figshare (doi:10.6084/m9.figshare.28690988) and
extract so that ``Sub1/ ... Sub50/`` sit directly under ``<src>``.

Output
------
``CATSA_125Hz.pkl``::

    {subject: {"data": [class0_array, class1_array], "label": [0, 1]}}

Processing
----------
1. Baseline -> non-stress (0); Stroop, Logic and Sudoku -> stress (1).
2. Tasks are concatenated in a fixed order (``TASK_ORDER``).  The order is
   pinned rather than taken from ``os.listdir`` so the segment file is
   identical on every filesystem; it reproduces the concatenation used for the
   reported results.
3. Clip sensor glitches at the 0.1/99.9 global percentiles.
4. Global z-score over all retained samples.
5. Linear resampling 64 Hz -> 125 Hz.

No bandpass filtering happens here; each 10 s window is filtered at training
time (see ``data/preprocessing.py``).

Usage
-----
    python dataset/build_catsa.py --src /data/CATSA --out dataset/CATSA_125Hz.pkl
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd

SRC_FS = 64
TGT_FS = 125

# Fixed task order.  Within each class the files are concatenated in this
# order; keep it stable so the output does not depend on directory listing
# order.
TASK_ORDER = ("Stroop", "Baseline", "Logic", "Sudoku")
TASK_LABELS = {"Baseline": 0, "Stroop": 1, "Sudoku": 1, "Logic": 1}


def build_catsa(src_dir, output_path):
    subjects = sorted(
        [d for d in os.listdir(src_dir) if d.startswith("Sub")],
        key=lambda x: int(x.replace("Sub", "")),
    )
    print(f"Found {len(subjects)} subjects")

    # ── Step 1: collect raw per-class task regions ──────────────────────────
    raw_regions = {}
    for subj_name in subjects:
        subj_dir = os.path.join(src_dir, subj_name)
        regions = {0: [], 1: []}
        for task in TASK_ORDER:
            bvp_path = os.path.join(subj_dir, task, "BVP.csv")
            if not os.path.exists(bvp_path):
                continue
            label = TASK_LABELS[task]
            bvp = pd.read_csv(bvp_path)["BVP"].values.astype(np.float64)
            if len(bvp) > 0:
                regions[label].append(bvp)

        raw_regions[subj_name] = regions
        cls0_s = sum(len(r) for r in regions[0]) / SRC_FS
        cls1_s = sum(len(r) for r in regions[1]) / SRC_FS
        print(f"  {subj_name}: class0={len(regions[0])} tasks ({cls0_s:.0f}s), "
              f"class1={len(regions[1])} tasks ({cls1_s:.0f}s)")

    # ── Step 2: clip sensor glitches ────────────────────────────────────────
    all_raw = np.concatenate([
        np.concatenate([r for cls in [0, 1] for r in regions[cls]])
        for regions in raw_regions.values()
    ])
    lo, hi = float(np.percentile(all_raw, 0.1)), float(np.percentile(all_raw, 99.9))
    print(f"\nSpike clipping: [{lo:.1f}, {hi:.1f}]")
    for regions in raw_regions.values():
        for cls in [0, 1]:
            for i in range(len(regions[cls])):
                regions[cls][i] = np.clip(regions[cls][i], lo, hi)

    # ── Step 3: global z-score ──────────────────────────────────────────────
    all_data = np.concatenate([
        np.concatenate([r for cls in [0, 1] for r in regions[cls]])
        for regions in raw_regions.values()
    ])
    mean = float(np.mean(all_data))
    std = float(np.std(all_data))
    print(f"\nGlobal z-score: mean={mean:.4f}, std={std:.4f}")

    # ── Step 4: z-score -> resample -> concatenate per subject ──────────────
    output = {}
    inv_std = 1.0 / (std + 1e-8)
    for name, regions in raw_regions.items():
        data_list, label_list = [], []
        for cls in [0, 1]:
            zscored = [(r - mean) * inv_std for r in regions[cls]]
            resampled = []
            for x in zscored:
                t_orig = np.arange(len(x)) / SRC_FS
                total_time = t_orig[-1]
                if total_time <= 0:
                    continue
                t_new = np.arange(0, total_time, 1.0 / TGT_FS)
                resampled.append(np.interp(t_new, t_orig, x).astype(np.float32))
            if resampled:
                data_list.append(np.concatenate(resampled))
                label_list.append(cls)
        if data_list:
            output[name] = {"data": data_list, "label": label_list}
            total_s = sum(len(d) for d in data_list) / TGT_FS
            n0_s = (len(data_list[0]) / TGT_FS) if label_list[0] == 0 else 0
            n1_s = (len(data_list[-1]) / TGT_FS) if label_list[-1] == 1 else 0
            print(f"  {name}: {total_s:.0f}s (class0={n0_s:.0f}s, class1={n1_s:.0f}s)")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(output, f, protocol=4)

    total_h = sum(sum(len(d) for d in v["data"]) for v in output.values()) / TGT_FS / 3600
    print(f"\nSaved {output_path} ({os.path.getsize(output_path) / 1e6:.1f} MB)")
    print(f"{len(output)} subjects, {total_h:.1f} h @ {TGT_FS} Hz")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", required=True, help="CATSA root containing Sub1/ ... Sub50/")
    p.add_argument("--out", default="dataset/CATSA_125Hz.pkl")
    args = p.parse_args()
    build_catsa(args.src, args.out)
