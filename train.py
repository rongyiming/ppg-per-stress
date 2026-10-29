#!/usr/bin/env python
"""Train and evaluate the personalized expert-routing model.

The script has two stages:

``--stage pretrain``
    Regress 10 s PPG windows onto their own 10-dimensional handcrafted
    individual features.  Produces the backbone checkpoint used to initialise
    finetuning.  A checkpoint matching the reported results is shipped in
    ``checkpoints/``, so this stage is optional.

``--stage finetune`` (default)
    Subject-independent 5-fold cross-validation on WESAD, CATSA and CLAS with
    the final architecture.  Writes per-fold predictions and a ``results.json``
    summary.

``--stage both`` runs pretraining and then finetuning.

All defaults reproduce the configuration reported in the paper.  Run
``python train.py --help`` for the full list.

Examples
--------
::

    # Evaluate the final model (5-fold, all three datasets)
    python train.py --stage finetune --datasets WESAD CATSA CLAS

    # Single fold while debugging
    python train.py --stage finetune --datasets WESAD --fold-idx 0 --epochs 5

    # Pretrain the backbone on a VitalDB segment corpus, then finetune
    python train.py --stage both --vital-dir /data/VITALDB_10s
"""

import argparse
import json
import os
import pickle
import random
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from tqdm import tqdm

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data.dataset import (build_segments, create_finetune_dataloaders,  # noqa: E402
                          load_dataset_pkl)
from data.pretrain_dataset import load_corpus, split_subjects          # noqa: E402
from data.pretrain_dataset import make_dataloaders as make_pretrain_loaders  # noqa: E402
from models.finetune_model import FinetuneModel                       # noqa: E402
from models.losses import FocalLoss                                   # noqa: E402
from models.pretrain_model import PretrainModel                       # noqa: E402

# ── Fixed pipeline constants ─────────────────────────────────────────────────
FREQ = 125                 # working sampling rate (Hz)
WINDOW_S = 10              # window length (s)
OVERLAP = 0.5
INDIVIDUAL_S = 120         # enrollment segment length (s)
N_INDIVIDUAL = 10          # handcrafted descriptors predicted at pretraining
ALL_DATASETS = ["WESAD", "CLAS", "CATSA"]

# The segments cache is always built with a Chebyshev type-I order-4 bandpass.
# It is pinned here because filtering happens when the cache is built, not
# during training: a CLI flag could not change it without invalidating the
# cache for every other method.
FILTER_ORDER = 4
FILTER_TYPE = "cheby"

# Defaults that reproduce the final model.
FINAL = {
    "backbone": "inception_time",
    "expert_num": 4,
    "expert_type": "lstm",
    "alpha": 0.3,
    "lambda1": 1.0,
    "lambda2": 0.1,
    "focal_gamma": 2.0,
    "focal_alpha": 0.25,
    "lr": 1e-4,
    "weight_decay": 2e-3,
    "backbone_lr": 5e-5,
    "epochs": 1000,
    "batch_size": 128,
    "patience": 50,
}


# ── Reproducibility ──────────────────────────────────────────────────────────

def set_seed(seed=42):
    """Seed Python, NumPy and PyTorch and request deterministic cuDNN kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dump_predictions(out_dir, ds_name, tag, suffix, split, labs, probs, subjects):
    """Save per-window ground truth, probability and subject id for analysis."""
    preds_dir = os.path.join(out_dir, "preds")
    os.makedirs(preds_dir, exist_ok=True)
    stem = os.path.join(preds_dir, f"{ds_name}_{tag or 'run'}_{suffix}_{split}")
    np.save(f"{stem}_y_true.npy", np.asarray(labs, dtype=np.int64))
    np.save(f"{stem}_y_pred_proba.npy", np.asarray(probs, dtype=np.float64))
    np.save(f"{stem}_subject.npy", np.asarray(subjects, dtype=object),
            allow_pickle=True)


# ═════════════════════════════════════════════════════════════════════════════
# Stage 1 — pretraining
# ═════════════════════════════════════════════════════════════════════════════

def pretrain_loss_fn(outputs, labels, criterion):
    """Sum of MSE over the 10 individual-feature dimensions."""
    return sum(criterion(outputs[:, i], labels[:, i])
               for i in range(N_INDIVIDUAL))


def run_pretraining(vital_dir, out_dir, device, epochs, batch_size,
                    backbone, seed=42, max_subjects=None, cap=192,
                    min_segments=50, valid_ratio=0.15, num_workers=8):
    """Train the backbone to regress each window's own individual features."""
    print(f"\n{'=' * 70}")
    print(f"  STAGE 1: pretraining ({backbone}, {epochs} epochs, {FREQ} Hz)")
    print(f"{'=' * 70}")

    if not os.path.isdir(vital_dir):
        raise FileNotFoundError(
            f"{vital_dir} not found. Provide --vital-dir pointing at the "
            f"segment corpus built by dataset/build_vital_ppg.py, or skip "
            f"pretraining and use the shipped checkpoint.")

    print(f"[1.1] Indexing corpus: {vital_dir}")
    corpus = load_corpus(vital_dir, min_segments)
    if not corpus:
        raise FileNotFoundError(
            f"No subject file with at least {min_segments} segments in "
            f"{vital_dir}.")
    if max_subjects is not None:
        corpus = {s: corpus[s] for s in sorted(corpus)[:max_subjects]}

    train_subjects, valid_subjects = split_subjects(corpus, valid_ratio, seed)
    if not train_subjects or not valid_subjects:
        raise ValueError(
            f"Need at least two subjects for a {1 - valid_ratio:.0%}/"
            f"{valid_ratio:.0%} split; found {len(corpus)}.")
    print(f"  {len(corpus)} subjects (train {len(train_subjects)} / "
          f"valid {len(valid_subjects)}), at most {cap} segments each")

    train_loader, val_loader, n_train, n_val = make_pretrain_loaders(
        corpus, train_subjects, valid_subjects, cap,
        batch_size=batch_size, num_workers=num_workers,
        cache_dir=os.path.join(out_dir, "cache", "segfeat"))
    print(f"[1.2] Train {n_train} / Val {n_val}")

    model = PretrainModel(input_channels=1, resnet_depth=18, hidden_size=64,
                          output_size=N_INDIVIDUAL,
                          backbone=backbone).to(device)
    print(f"[1.3] Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    ckpt_path = os.path.join(out_dir, "checkpoints",
                             f"pretrain_{backbone}_vital.pth")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    best_val = float("inf")
    bad_epochs = 0
    print("[1.4] Training ...")
    for epoch in range(epochs):
        model.train()
        running = 0.0
        t0 = time.time()
        for inputs, labels in tqdm(train_loader,
                                   desc=f"  epoch {epoch + 1}/{epochs}",
                                   leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = pretrain_loss_fn(outputs, labels, criterion)
            loss.backward()
            optimizer.step()
            running += loss.item() * inputs.size(0)
        train_loss = running / n_train

        model.eval()
        val_loss, n_val_seen = 0.0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                val_loss += (pretrain_loss_fn(outputs, labels, criterion).item()
                             * inputs.size(0))
                n_val_seen += inputs.size(0)
        val_loss /= n_val_seen

        improved = val_loss < best_val
        print(f"  epoch {epoch + 1:3d}  train {train_loss:.6f}  "
              f"val {val_loss:.6f}  [{time.time() - t0:.0f}s]"
              f"{'  -> saved' if improved else ''}")
        if improved:
            best_val = val_loss
            bad_epochs = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs > 10:
                print(f"  early stop at epoch {epoch + 1}")
                break

    print(f"[1.5] Best val {best_val:.6f}")
    print(f"      checkpoint: {ckpt_path}")
    return ckpt_path


# ═════════════════════════════════════════════════════════════════════════════
# Stage 2 — finetuning
# ═════════════════════════════════════════════════════════════════════════════

def build_segments_cache(data_dir, out_dir, enrollment_s):
    """Build (or load) the per-window segments used by every method."""
    cache_name = (f"segments_{FREQ}Hz_e{enrollment_s}s_"
                  f"f{FILTER_ORDER}{FILTER_TYPE}.pkl")
    cache_path = os.path.join(out_dir, "cache", cache_name)
    if os.path.exists(cache_path):
        print(f"[2.1] Loading segments cache: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"[2.1] Building segments cache (enrollment={enrollment_s} s) ...")
    cache = {}
    for ds_name in ALL_DATASETS:          # always build all three
        pkl_path = os.path.join(data_dir, f"{ds_name}_{FREQ}Hz.pkl")
        if not os.path.exists(pkl_path):
            print(f"  skip {ds_name}: {pkl_path} not found")
            continue
        t0 = time.time()
        raw = load_dataset_pkl(pkl_path)
        bundle = build_segments(raw, fs=FREQ, window_s=WINDOW_S, overlap=OVERLAP,
                                filter_order=FILTER_ORDER,
                                filter_type=FILTER_TYPE,
                                enrollment_s=enrollment_s)
        print(f"  {ds_name}: {len(bundle['fixed_segments'])} fixed / "
              f"{len(bundle['adaptive_segments'])} adaptive windows, "
              f"{len(bundle['individual_feats'])} subjects "
              f"({time.time() - t0:.0f}s)")
        cache[ds_name] = bundle

    if not cache:
        raise SystemExit(
            f"No segment files found in {data_dir}.\n"
            "Run scripts/prepare_data.sh first (see README.md).")

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f, protocol=4)
    print(f"  saved cache: {cache_path}")
    return cache


def load_backbone_weights(pretrain_ckpt, backbone, scratch):
    """Extract ``backbone.*`` tensors from a pretraining checkpoint."""
    if scratch:
        print("[2.0] Scratch mode: random initialisation")
        return {}
    print(f"[2.0] Loading pretrained backbone: {pretrain_ckpt}")
    pt_model = PretrainModel(input_channels=1, resnet_depth=18, hidden_size=64,
                             output_size=N_INDIVIDUAL, backbone=backbone)
    state = torch.load(pretrain_ckpt, map_location="cpu", weights_only=True)
    pt_model.load_state_dict(state, strict=False)
    weights = {k: v for k, v in pt_model.state_dict().items()
               if k.startswith("backbone.")}
    print(f"     {len(weights)} backbone tensors")
    return weights


def evaluate(model, loader, device):
    """Return predictions, probabilities and subject ids for one loader."""
    model.eval()
    probs, labs, preds, sids = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            inputs = batch[0].to(device)
            labels = batch[1].to(device)
            individual = batch[3].to(device)
            subject_ids = batch[4]
            logits, _ = model(inputs, individual=individual)
            probs.extend(F.softmax(logits, dim=1)[:, 1].cpu().tolist())
            preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
            labs.extend(labels.cpu().tolist())
            sids.extend(list(subject_ids))
    return labs, preds, probs, sids


def run_finetuning(args, device):
    weights = load_backbone_weights(args.pretrain_ckpt, args.backbone, args.scratch)
    segments = build_segments_cache(args.data_dir, args.out_dir,
                                    args.enrollment_s)

    criterion = FocalLoss(gamma=args.focal_gamma, alpha=args.focal_alpha)
    results = {ds: {k: [] for k in
                    ("acc", "prec", "rec", "f1", "auc",
                     "val_acc", "val_f1", "val_auc")}
               for ds in args.datasets}

    fold_indices = [args.fold_idx] if args.fold_idx >= 0 else list(range(5))

    for fold_idx in fold_indices:
        print(f"\n{'=' * 60}\n  FOLD {fold_idx}\n{'=' * 60}")
        set_seed(args.seed)

        loaders = []
        for ds_name in args.datasets:
            if ds_name not in segments:
                print(f"  skip {ds_name}: no segments")
                continue
            train_ldr, val_ldr, test_ldr = create_finetune_dataloaders(
                segments[ds_name], fs=FREQ, batch_size=args.batch_size,
                seed=args.seed, balance_classes=args.balance_classes,
                fold_idx=fold_idx, global_ind_norm=args.global_ind_norm)
            loaders.append((ds_name, train_ldr, val_ldr, test_ldr))

        for ds_name, train_loader, val_loader, test_loader in loaders:
            print(f"\n  --- {ds_name} (fold={fold_idx}) ---")
            t0 = time.time()

            model = FinetuneModel(
                input_channels=1, resnet_depth=18,
                expert_num=args.expert_num,
                lstm_input_size=256, lstm_hidden_size=64, lstm_num_layers=1,
                output_size=2, alpha=args.alpha,
                use_input_norm=args.use_input_norm,
                top_k=args.top_k, backbone=args.backbone,
                single_level=args.single_level,
                feature_gate=args.feature_gate,
                expert_type=args.expert_type,
                use_domain_adapter=args.domain_adapter,
                use_head_skip=args.head_skip,
                use_gate_norm=args.gate_norm,
                use_mixstyle=args.mixstyle,
                use_experts=args.experts,
            ).to(device)
            missing, unexpected = model.load_state_dict(weights, strict=False)
            if missing:
                print(f"    missing keys: {len(missing)} (head/experts — expected)")
            if unexpected:
                print(f"    unexpected keys: {len(unexpected)}")
            model.finetune(partial_unfreeze=not args.unfreeze_all,
                           unfreeze_all=args.unfreeze_all)

            n_trainable = sum(p.numel() for p in model.parameters()
                              if p.requires_grad)
            n_total = sum(p.numel() for p in model.parameters())
            print(f"    params: {n_trainable:,} trainable / {n_total:,} total")

            if args.unfreeze_all:
                backbone_ids = {id(p) for p in model.backbone.parameters()}
                groups = [
                    {"params": [p for p in model.parameters()
                                if id(p) in backbone_ids], "lr": args.backbone_lr},
                    {"params": [p for p in model.parameters()
                                if id(p) not in backbone_ids], "lr": args.lr},
                ]
            else:
                groups = [{"params": [p for p in model.parameters()
                                      if p.requires_grad], "lr": args.lr}]
            optimizer = optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.999),
                                    weight_decay=args.weight_decay)

            ckpt_name = f"Finetune_{ds_name}_K{args.expert_num}_f{fold_idx}_best.pth"
            if args.tag:
                ckpt_name = (f"Finetune_{ds_name}_K{args.expert_num}_{args.tag}"
                             f"_f{fold_idx}_best.pth")
            ckpt_path = os.path.join(args.out_dir, "checkpoints", ckpt_name)
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

            best_val_loss = float("inf")
            best_val = {"acc": float("nan"), "f1": float("nan"),
                        "auc": float("nan")}
            bad_epochs = 0

            for epoch in range(args.epochs):
                model.train()
                tr_loss, tr_ce, tr_div, tr_cons = 0.0, 0.0, 0.0, 0.0
                n_seen = 0
                for batch in train_loader:
                    inputs = batch[0].to(device)
                    labels = batch[1].to(device)
                    individual = batch[3].to(device)

                    optimizer.zero_grad()
                    outputs, (div_loss, cons_loss) = model(
                        inputs, individual=individual)
                    ce_loss = criterion(outputs, labels)
                    loss = ce_loss + args.lambda1 * div_loss + args.lambda2 * cons_loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    n_seen += inputs.size(0)
                    tr_loss += loss.item() * inputs.size(0)
                    tr_ce += ce_loss.item() * inputs.size(0)
                    tr_div += div_loss.item() * inputs.size(0)
                    tr_cons += cons_loss.item() * inputs.size(0)

                tr_loss /= n_seen
                tr_ce /= n_seen
                tr_div /= n_seen
                tr_cons /= n_seen

                # Validation — model selection uses validation loss.
                model.eval()
                val_loss, n_val = 0.0, 0
                with torch.no_grad():
                    v_labs, v_preds, v_probs = [], [], []
                    for batch in val_loader:
                        v_inputs = batch[0].to(device)
                        v_labels = batch[1].to(device)
                        v_individual = batch[3].to(device)
                        v_logits, _ = model(v_inputs, individual=v_individual)
                        val_loss += (criterion(v_logits, v_labels).item()
                                     * v_inputs.size(0))
                        n_val += v_inputs.size(0)
                        v_probs.extend(F.softmax(v_logits, dim=1)[:, 1].cpu().tolist())
                        v_preds.extend(torch.argmax(v_logits, dim=1).cpu().tolist())
                        v_labs.extend(v_labels.cpu().tolist())
                val_loss /= n_val
                val_acc = accuracy_score(v_labs, v_preds)
                val_f1 = f1_score(v_labs, v_preds, pos_label=1, zero_division=0)
                try:
                    val_auc = roc_auc_score(v_labs, v_probs)
                except ValueError:
                    val_auc = float("nan")

                bad_epochs += 1
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val = {"acc": val_acc, "f1": val_f1, "auc": val_auc}
                    bad_epochs = 0
                    torch.save(model.state_dict(), ckpt_path)

                if (epoch + 1) % 10 == 0 or epoch < 5:
                    print(f"  {datetime.now():%m-%d %H:%M:%S} E{epoch + 1:3d}  "
                          f"loss {tr_ce:.4f}  div {tr_div:.4f}  "
                          f"cons {tr_cons:.4f}  val {val_loss:.4f}  "
                          f"[{time.time() - t0:.0f}s]")

                if bad_epochs > args.patience:
                    print(f"    early stop at epoch {epoch + 1}")
                    break

            # Test with the best-validation checkpoint.
            model.load_state_dict(torch.load(ckpt_path, map_location=device,
                                             weights_only=True))
            labs, preds, probs, sids = evaluate(model, test_loader, device)
            acc = accuracy_score(labs, preds)
            prec = precision_score(labs, preds, pos_label=1, zero_division=0)
            rec = recall_score(labs, preds, pos_label=1, zero_division=0)
            f1 = f1_score(labs, preds, pos_label=1, zero_division=0)
            try:
                auc = roc_auc_score(labs, probs)
            except ValueError:          # a test fold containing one class
                auc = float("nan")
            cm = confusion_matrix(labs, preds)

            for key, value in (("acc", acc), ("prec", prec), ("rec", rec),
                               ("f1", f1), ("auc", auc),
                               ("val_acc", best_val["acc"]),
                               ("val_f1", best_val["f1"]),
                               ("val_auc", best_val["auc"])):
                results[ds_name][key].append(float(value))

            dump_predictions(args.out_dir, ds_name, args.tag, f"f{fold_idx}",
                             "test", labs, probs, sids)
            v_labs, _, v_probs, v_sids = evaluate(model, val_loader, device)
            dump_predictions(args.out_dir, ds_name, args.tag, f"f{fold_idx}",
                             "val", v_labs, v_probs, v_sids)

            print(f"  Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} "
                  f"F1={f1:.4f} AUC={auc:.4f}")
            print(f"    confusion matrix: {cm.tolist()}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}\n  FINAL SUMMARY (5-fold CV)\n{'=' * 70}")
    summary = {}
    for ds in args.datasets:
        if not results[ds]["f1"]:
            print(f"\n  {ds}: no folds ran, skipped")
            continue
        row = {k: np.asarray(v, dtype=float) for k, v in results[ds].items()}
        summary[ds] = {k: {"mean": float(np.nanmean(v)),
                           "std": float(np.nanstd(v)),
                           "folds": [float(x) for x in v]}
                       for k, v in row.items()}
        print(f"\n  {ds}:")
        for metric in ("f1", "auc", "acc", "prec", "rec", "val_f1"):
            print(f"    {metric:>6s}: {summary[ds][metric]['mean']:.4f} "
                  f"+/- {summary[ds][metric]['std']:.4f}  "
                  f"{[round(x, 4) for x in summary[ds][metric]['folds']]}")

    out_json = os.path.join(args.out_dir, "results.json")
    with open(out_json, "w") as f:
        json.dump({"config": vars(args), "results": summary}, f, indent=2)
    print(f"\n  wrote {out_json}")


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Personalized expert routing for PPG-based stress detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # I/O
    p.add_argument("--stage", choices=["pretrain", "finetune", "both"],
                   default="finetune")
    p.add_argument("--data-dir", default=os.path.join(_ROOT, "dataset"),
                   help="directory holding {WESAD,CLAS,CATSA}_125Hz.pkl")
    p.add_argument("--out-dir", default=os.path.join(_ROOT, "runs"),
                   help="output directory for caches, checkpoints and predictions")
    p.add_argument("--vital-dir", default=None,
                   help="segment corpus built by dataset/build_vital_ppg.py "
                        "(pretraining only)")
    p.add_argument("--pretrain-ckpt",
                   default=os.path.join(_ROOT, "checkpoints",
                                        "pretrain_inception_time_vital.pth"))
    p.add_argument("--scratch", action="store_true",
                   help="train from random initialisation (no pretrained backbone)")
    p.add_argument("--tag", default="", help="suffix for output file names")

    # Data / splits
    p.add_argument("--datasets", nargs="+", default=ALL_DATASETS,
                   choices=ALL_DATASETS)
    p.add_argument("--enrollment-s", type=int, default=INDIVIDUAL_S)
    p.add_argument("--fold-idx", type=int, default=-1,
                   help="run a single fold (0-4); -1 runs all five")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--balance-classes", action="store_true", default=True)
    p.add_argument("--no-balance-classes", dest="balance_classes",
                   action="store_false")
    p.add_argument("--global-ind-norm", action="store_true", default=False,
                   help="min-max normalise individual features over all "
                        "subjects, held-out folds included (transductive; "
                        "kept to reproduce the earlier runs)")
    p.add_argument("--per-fold-ind-norm", dest="global_ind_norm",
                   action="store_false",
                   help="fit the descriptor scaler on the training folds "
                        "only (default)")

    # Architecture
    p.add_argument("--backbone", default=FINAL["backbone"],
                   choices=["inception_time", "resnet18", "resnet34", "resnet50"])
    p.add_argument("--expert-num", type=int, default=FINAL["expert_num"])
    p.add_argument("--expert-type", default=FINAL["expert_type"],
                   choices=["lstm", "lstm_bidir", "gru", "transformer"])
    p.add_argument("--alpha", type=float, default=FINAL["alpha"],
                   help="weight of the feature gate; the individual gate gets "
                        "1 - alpha")
    p.add_argument("--top-k", type=int, default=None,
                   help="sparse top-k routing; default is full softmax")
    p.add_argument("--feature-gate", default="expert0",
                   choices=["expert0", "per_expert", "backbone_gap",
                            "expert_mean", "expert_concat"])
    p.add_argument("--single-level", action="store_true", default=True,
                   help="all experts read the deepest feature level")
    p.add_argument("--no-single-level", dest="single_level",
                   action="store_false")
    p.add_argument("--use-input-norm", action="store_true", default=True,
                   help="InstanceNorm1d on the input window")
    p.add_argument("--no-input-norm", dest="use_input_norm",
                   action="store_false")
    p.add_argument("--no-domain-adapter", dest="domain_adapter",
                   action="store_false")
    p.add_argument("--no-head-skip", dest="head_skip", action="store_false")
    p.add_argument("--no-gate-norm", dest="gate_norm", action="store_false")
    p.add_argument("--no-mixstyle", dest="mixstyle", action="store_false")
    p.add_argument("--no-experts", dest="experts", action="store_false")

    # Optimisation
    p.add_argument("--epochs", type=int, default=FINAL["epochs"])
    p.add_argument("--batch-size", type=int, default=FINAL["batch_size"])
    p.add_argument("--lr", type=float, default=FINAL["lr"])
    p.add_argument("--weight-decay", type=float, default=FINAL["weight_decay"])
    p.add_argument("--backbone-lr", type=float, default=FINAL["backbone_lr"])
    p.add_argument("--patience", type=int, default=FINAL["patience"])
    p.add_argument("--unfreeze-all", action="store_true", default=True)
    p.add_argument("--freeze-backbone", dest="unfreeze_all", action="store_false")
    p.add_argument("--lambda1", type=float, default=FINAL["lambda1"],
                   help="expert-diversity loss weight")
    p.add_argument("--lambda2", type=float, default=FINAL["lambda2"],
                   help="feature/routing consistency loss weight")
    p.add_argument("--focal-gamma", type=float, default=FINAL["focal_gamma"])
    p.add_argument("--focal-alpha", type=float, default=FINAL["focal_alpha"])

    # Pretraining
    p.add_argument("--pretrain-epochs", type=int, default=100)
    p.add_argument("--pretrain-batch-size", type=int, default=512)
    p.add_argument("--pretrain-cap", type=int, default=192,
                   help="max segments per subject")
    p.add_argument("--pretrain-min-segments", type=int, default=50,
                   help="drop subjects with fewer segments than this")
    p.add_argument("--pretrain-valid-ratio", type=float, default=0.15,
                   help="fraction of subjects held out for validation")
    p.add_argument("--pretrain-workers", type=int, default=8,
                   help="DataLoader workers for pretraining")
    p.add_argument("--pretrain-max-subjects", type=int, default=None)

    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # Optional YAML/JSON config.  It is applied as *defaults* before parsing,
    # so a single file can describe a whole experiment while an explicit
    # command-line flag still wins.
    p.add_argument("--config", default=None,
                   help="YAML/JSON file whose keys set the defaults")
    cfg_path = _prescan_config(sys.argv[1:])
    if cfg_path:
        p.set_defaults(**load_config(p, cfg_path))
        print(f"[0] Loaded config: {cfg_path}")
    return p.parse_args()


def _prescan_config(argv):
    """Return the value of --config in argv without a full parse."""
    for i, tok in enumerate(argv):
        if tok == "--config" and i + 1 < len(argv):
            return argv[i + 1]
        if tok.startswith("--config="):
            return tok.split("=", 1)[1]
    return None


def load_config(parser, path):
    """Read a YAML/JSON config into a dict of argparse defaults."""
    if path.endswith(".json"):
        with open(path) as f:
            cfg = json.load(f)
    else:
        import yaml
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
    known = {a.dest for a in parser._actions}
    out = {}
    for key, value in cfg.items():
        dest = key.lstrip("-").replace("-", "_")
        if dest not in known:
            raise SystemExit(f"unknown config key: {key}")
        out[dest] = value
    return out


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    t_start = time.time()

    ckpt = args.pretrain_ckpt
    if args.stage in ("pretrain", "both"):
        if not args.vital_dir:
            raise SystemExit("--vital-dir is required for --stage "
                             f"{args.stage}")
        ckpt = run_pretraining(
            vital_dir=args.vital_dir, out_dir=args.out_dir, device=device,
            epochs=args.pretrain_epochs, batch_size=args.pretrain_batch_size,
            backbone=args.backbone, seed=args.seed,
            max_subjects=args.pretrain_max_subjects,
            cap=args.pretrain_cap, min_segments=args.pretrain_min_segments,
            valid_ratio=args.pretrain_valid_ratio,
            num_workers=args.pretrain_workers)

    if args.stage in ("finetune", "both"):
        args.pretrain_ckpt = ckpt
        if not args.scratch and not os.path.exists(ckpt):
            raise SystemExit(
                f"Pretrained checkpoint not found: {ckpt}\n"
                f"Pass --pretrain-ckpt, run --stage pretrain first, or use "
                f"--scratch.")
        run_finetuning(args, device)

    print(f"\nDone in {(time.time() - t_start) / 3600:.2f} h.")


if __name__ == "__main__":
    main()
