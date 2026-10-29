#!/usr/bin/env python
"""Recompute metrics from the per-window predictions dumped during training.

``train.py`` writes, for every dataset, fold and split::

    <preds_dir>/<DATASET>_<tag>_f<fold>_<split>_y_true.npy
    <preds_dir>/<DATASET>_<tag>_f<fold>_<split>_y_pred_proba.npy
    <preds_dir>/<DATASET>_<tag>_f<fold>_<split>_subject.npy

This script reads them back, so metrics can be recomputed -- or confidence
intervals added -- without retraining anything.

Reported quantities
-------------------
per-fold      metric on each fold's test subjects, then mean +/- std over folds
pooled        metric over all windows of all folds concatenated (for reference
              only: windows overlap by 50 %, so pooled counts are not
              independent and this number is optimistic)
bootstrap     subject-level 95 % CI: resample *subjects* with replacement,
              recompute the metric, take the 2.5/97.5 percentiles

Usage
-----
    python evaluate.py runs/preds
    python evaluate.py runs/preds --tag v2 --split test --out runs/metrics.md
"""

import argparse
import glob
import json
import os
import re

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.metrics import recall_score, roc_auc_score

PATTERN = re.compile(
    r"(?P<ds>[^_]+)_(?P<tag>.*)_f(?P<fold>\d+)_(?P<split>test|val)_y_true\.npy$")

METRICS = ("acc", "prec", "rec", "spec", "f1", "auc")


def compute(labels, probs):
    """All reported metrics for one set of windows."""
    preds = (np.asarray(probs) >= 0.5).astype(int)
    out = {
        "acc": accuracy_score(labels, preds),
        "prec": precision_score(labels, preds, pos_label=1, zero_division=0),
        "rec": recall_score(labels, preds, pos_label=1, zero_division=0),
        "spec": recall_score(labels, preds, pos_label=0, zero_division=0),
        "f1": f1_score(labels, preds, pos_label=1, zero_division=0),
    }
    try:
        out["auc"] = roc_auc_score(labels, probs)
    except ValueError:                      # a fold with a single class
        out["auc"] = float("nan")
    return out


def collect(preds_dir, split):
    """Return {(dataset, tag): {fold: (labels, probs, subject_ids)}}."""
    runs = {}
    for path in sorted(glob.glob(os.path.join(preds_dir, "*.npy"))):
        m = PATTERN.search(os.path.basename(path))
        if not m or m.group("split") != split:
            continue
        stem = path[: -len("_y_true.npy")]
        labels = np.load(f"{stem}_y_true.npy")
        probs = np.load(f"{stem}_y_pred_proba.npy")
        subj_path = f"{stem}_subject.npy"
        subjects = (np.load(subj_path, allow_pickle=True) if os.path.exists(subj_path)
                    else np.full(len(labels), m.group("fold")))
        runs.setdefault((m.group("ds"), m.group("tag")), {})[int(m.group("fold"))] = (
            labels, probs, np.asarray(subjects, dtype=object))
    return runs


def bootstrap_f1(subjects, labels, probs, n_boot, seed=0):
    """Subject-level bootstrap CI for F1 (resampling subjects, not windows)."""
    rng = np.random.RandomState(seed)
    uniq = np.unique(subjects)
    idx_by_subject = {s: np.flatnonzero(subjects == s) for s in uniq}
    stats = []
    for _ in range(n_boot):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_subject[s] for s in drawn])
        stats.append(f1_score(labels[idx], (probs[idx] >= 0.5).astype(int),
                              pos_label=1, zero_division=0))
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def evaluate(preds_dir, split="test", n_boot=1000):
    runs = collect(preds_dir, split)
    if not runs:
        raise SystemExit(f"no *_{split}_y_true.npy files under {preds_dir}")

    report = {}
    for (ds, tag), folds in sorted(runs.items()):
        per_fold = {fold: compute(lab, prob) for fold, (lab, prob, _) in folds.items()}
        lab_all = np.concatenate([v[0] for v in folds.values()])
        prob_all = np.concatenate([v[1] for v in folds.values()])
        sub_all = np.concatenate([v[2] for v in folds.values()])

        summary = {
            "n_folds": len(folds),
            "n_windows": int(len(lab_all)),
            "n_subjects": int(len(np.unique(sub_all))),
            "per_fold": {str(k): {m: float(v[m]) for m in METRICS}
                         for k, v in per_fold.items()},
            "mean_std": {m: {"mean": float(np.mean([v[m] for v in per_fold.values()])),
                             "std": float(np.std([v[m] for v in per_fold.values()]))}
                         for m in METRICS},
            "pooled": {m: float(v) for m, v in compute(lab_all, prob_all).items()},
        }
        if len(np.unique(sub_all)) > 1:
            lo, hi = bootstrap_f1(sub_all, lab_all, prob_all, n_boot)
            summary["f1_subject_bootstrap_ci95"] = [lo, hi]
        report[ds if not tag or tag == "run" else f"{ds}_{tag}"] = summary

    return report


def render_markdown(report, split):
    lines = [f"# Metrics ({split} split)", "",
             "`per fold` = mean ± std over folds; `pooled` concatenates all "
             "windows (overlapping, optimistic); `F1 95 % CI` is a "
             "subject-level bootstrap over all folds.", "",
             "| Run | Windows | Subjects | Acc | Precision | Recall | "
             "Specificity | F1 | AUC |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, s in report.items():
        ms = s["mean_std"]
        lines.append(
            "| {} | {} | {} | {:.4f} ± {:.4f} | {:.4f} ± {:.4f} | "
            "{:.4f} ± {:.4f} | {:.4f} ± {:.4f} | **{:.4f} ± {:.4f}** | "
            "{:.4f} ± {:.4f} |".format(
                name, s["n_windows"], s["n_subjects"],
                ms["acc"]["mean"], ms["acc"]["std"],
                ms["prec"]["mean"], ms["prec"]["std"],
                ms["rec"]["mean"], ms["rec"]["std"],
                ms["spec"]["mean"], ms["spec"]["std"],
                ms["f1"]["mean"], ms["f1"]["std"],
                ms["auc"]["mean"], ms["auc"]["std"]))
    lines.append("")
    for name, s in report.items():
        if "f1_subject_bootstrap_ci95" in s:
            lo, hi = s["f1_subject_bootstrap_ci95"]
            lines.append(f"- **{name}** F1 95 % CI (subject bootstrap): "
                         f"[{lo:.4f}, {hi:.4f}]  · pooled F1 = "
                         f"{s['pooled']['f1']:.4f}")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("preds_dir", help="directory holding the dumped .npy files")
    p.add_argument("--tag", default=None,
                   help="only report runs with this tag (default: all)")
    p.add_argument("--split", default="test", choices=["test", "val"])
    p.add_argument("--bootstrap", type=int, default=1000,
                   help="subject-bootstrap resamples (0 disables)")
    p.add_argument("--out", default=None,
                   help="write the Markdown table here as well as printing it")
    args = p.parse_args()

    report = evaluate(args.preds_dir, args.split, args.bootstrap)
    if args.tag is not None:
        report = {k: v for k, v in report.items()
                  if k.split("_", 1)[-1] == args.tag}
        if not report:
            raise SystemExit(f"no runs with tag {args.tag!r}")

    md = render_markdown(report, args.split)
    print(md)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            f.write(md)
        with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
            json.dump(report, f, indent=2)
        print(f"wrote {args.out} and its .json twin")


if __name__ == "__main__":
    main()
