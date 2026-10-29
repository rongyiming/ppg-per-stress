#!/usr/bin/env bash
# Reproduce the reported run: 5 folds x 3 datasets, ~1-1.5 h on an A800.
#
#   bash scripts/run_final.sh                      # all five folds
#   bash scripts/run_final.sh --fold-idx 0 --tag smoke
#
# Any extra arguments are passed through to train.py and override the config.
set -euo pipefail

cd "$(dirname "$0")/.."

python train.py --config configs/final.yaml "$@"

echo
echo "Metrics from the dumped predictions:"
python evaluate.py runs/preds --out runs/metrics.md
