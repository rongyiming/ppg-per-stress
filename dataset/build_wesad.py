#!/usr/bin/env python
"""Convert the raw WESAD recordings into the pipeline's 125 Hz segment file.

Input
-----
``<src>/S*/S*.pkl`` -- the original WESAD subject files (BVP at 64 Hz,
labels at 700 Hz).  Download WESAD from the UCI machine learning repository
and extract it so that ``S2/ ... S17/`` sit directly under ``<src>``.

Output
------
``WESAD_125Hz.pkl``::

    {subject: {"data": [class0_array, class1_array], "label": [0, 1]}}

Processing
----------
1. Map the 700 Hz label stream onto the 64 Hz BVP stream and cut contiguous
   regions per label.
2. Label mapping: TSST (2) -> stress (1); baseline/amusement/meditation
   (1/3/4) -> non-stress (0); everything else is discarded.
3. Clip sensor glitches at the 0.1/99.9 global percentiles.
4. Global z-score over all retained samples (one mean/std for the dataset).
5. Linear resampling 64 Hz -> 125 Hz, then concatenate same-label regions in
   the original recording order.

No bandpass filtering happens here; each 10 s window is filtered at training
time (see ``data/preprocessing.py``).

Usage
-----
    python dataset/build_wesad.py --src /data/WESAD --out dataset/WESAD_125Hz.pkl
"""

import argparse
import os
import pickle

import numpy as np

SRC_FS_BVP = 64
SRC_FS_LABEL = 700
TGT_FS = 125

# The 15 analysed subjects are S2-S17 without S12: S1 was the pilot recording
# and S12 has a faulty sensor.  Both are present in some copies of the archive.
EXCLUDED_SUBJECTS = {"S1", "S12"}


def build_wesad(src_dir, output_path):
    folders = sorted(
        [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))],
        key=lambda x: int(x[1:]) if x[1:].isdigit() else 0,
    )
    folders = [d for d in folders
               if d.startswith("S") and d not in EXCLUDED_SUBJECTS]
    print(f"Found {len(folders)} subjects: {folders}")

    # ── Step 1: collect raw per-class regions ────────────────────────────────
    raw_regions = {}  # {subject: {0: [arr, ...], 1: [arr, ...]}}

    for folder in folders:
        name = folder
        pkl_path = os.path.join(src_dir, folder, f"{name}.pkl")
        with open(pkl_path, "rb") as f:
            raw = pickle.load(f, encoding="latin-1")

        bvp_all = raw["signal"]["wrist"]["BVP"]
        labels_all = raw["label"]

        regions = {0: [], 1: []}
        last_j, last_i = 0, 0

        for i in range(len(labels_all)):
            j = i * SRC_FS_BVP // SRC_FS_LABEL
            if labels_all[i] != labels_all[last_i]:
                this_label = labels_all[last_i]
                if this_label == 2:
                    cls = 1
                elif this_label in (1, 3, 4):
                    cls = 0
                else:
                    last_i, last_j = i, j
                    continue

                region = np.array([x[0] for x in bvp_all[last_j:j]], dtype=np.float64)
                if len(region) > 0:
                    regions[cls].append(region)
                last_i, last_j = i, j

        # Handle the final segment.
        this_label = labels_all[last_i]
        if this_label == 2:
            cls = 1
        elif this_label in (1, 3, 4):
            cls = 0
        else:
            cls = None
        if cls is not None:
            region = np.array([x[0] for x in bvp_all[last_j:]], dtype=np.float64)
            if len(region) > 0:
                regions[cls].append(region)

        raw_regions[name] = regions
        total_s = sum(
            sum(len(r) for r in regions[cls]) / SRC_FS_BVP for cls in [0, 1]
        )
        print(f"  {name}: class0={len(regions[0])} regions, "
              f"class1={len(regions[1])} regions, {total_s:.0f}s total")

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
                t_orig = np.arange(len(x)) / SRC_FS_BVP
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
    p.add_argument("--src", required=True, help="WESAD root containing S2/ ... S17/")
    p.add_argument("--out", default="dataset/WESAD_125Hz.pkl")
    args = p.parse_args()
    build_wesad(args.src, args.out)
