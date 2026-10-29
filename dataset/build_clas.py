#!/usr/bin/env python
"""Convert the raw CLAS recordings into the pipeline's 125 Hz segment file.

Input
-----
``<src>/Block_details/Part*_Block_Details.csv`` -- block type, file name and
duration for every block of every participant.
``<src>/Participants/Part*/by_block/*.csv`` -- per-block signals; the ``ppg``
column is the raw Shimmer3 optical-pulse signal at 256 Hz (16-bit ADC counts).

Output
------
``CLAS_125Hz.pkl``::

    {participant: {"data": [class0_array, class1_array], "label": [0, 1]}}

Processing
----------
1. Label mapping: ``Stroop Test`` -> stress (1); ``Baseline`` and ``Neutral``
   -> non-stress (0).  Math/IQ tests, their response blocks, videos and
   pictures are discarded, so only the Stroop blocks carry the stress label.
2. Within each class, blocks are concatenated in the row order of
   ``Block_Details.csv`` (which is a lexicographic block-id order, not the
   chronological recording order).  This order is preserved because it is the
   one used for the reported results.
3. Clip sensor glitches at the 0.1/99.9 global percentiles.
4. Global z-score over all retained samples.
5. Linear resampling 256 Hz -> 125 Hz.

Participants without a ``Block_Details`` file are skipped automatically (CLAS
ships 60 participant directories; ``Part4`` has no block description, leaving
59 usable participants).

No bandpass filtering happens here; each 10 s window is filtered at training
time (see ``data/preprocessing.py``).

Usage
-----
    python dataset/build_clas.py --src /data/CLAS --out dataset/CLAS_125Hz.pkl
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd

SRC_FS = 256
TGT_FS = 125


def find_files_with_prefix(folder_path, prefix):
    """Return files in ``folder_path`` whose name starts with ``prefix``."""
    if not os.path.isdir(folder_path):
        return []
    matched = []
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path) and item.startswith(prefix):
            matched.append(item_path)
    return matched


def build_clas(src_dir, output_path):
    participants = [f"Part{x}" for x in range(1, 61)]

    # ── Step 1: collect raw per-class blocks ────────────────────────────────
    raw_regions = {}
    n_loaded = 0
    for part in participants:
        block_csv = os.path.join(src_dir, "Block_details",
                                 f"{part}_Block_Details.csv")
        if not os.path.exists(block_csv):
            print(f"  {part}: no Block_Details file, skipped")
            continue

        block_details = pd.read_csv(block_csv)
        regions = {0: [], 1: []}

        for _, row in block_details.iterrows():
            block_type = str(row["Block Type"]).strip()

            if block_type == "Stroop Test":
                label = 1
            elif block_type in ("Baseline", "Neutral"):
                label = 0
            else:
                continue

            file_prefix = str(row["EDA&PPG File"]).split(".")[0]
            block_dir = os.path.join(src_dir, "Participants", part, "by_block")
            block_files = find_files_with_prefix(block_dir, prefix=file_prefix)
            if not block_files:
                continue

            ppg = pd.read_csv(block_files[0])["ppg"].values.astype(np.float64)
            if len(ppg) > 0:
                regions[label].append(ppg)

        if regions[0] or regions[1]:
            raw_regions[part] = regions
            n_loaded += 1
            cls0_s = sum(len(r) for r in regions[0]) / SRC_FS
            cls1_s = sum(len(r) for r in regions[1]) / SRC_FS
            print(f"  {part}: class0={len(regions[0])} blocks ({cls0_s:.0f}s), "
                  f"class1={len(regions[1])} blocks ({cls1_s:.0f}s)")

    print(f"Loaded {n_loaded} participants")

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

    # ── Step 4: z-score -> resample -> concatenate per participant ──────────
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
    print(f"{len(output)} participants, {total_h:.1f} h @ {TGT_FS} Hz")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", required=True,
                   help="CLAS root containing Block_details/ and Participants/")
    p.add_argument("--out", default="dataset/CLAS_125Hz.pkl")
    args = p.parse_args()
    build_clas(args.src, args.out)
