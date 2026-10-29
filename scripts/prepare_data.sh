#!/usr/bin/env bash
# Build the three 125 Hz segment files from the raw datasets.
#
#   export WESAD_RAW=/path/to/WESAD      # contains S2/ ... S17/
#   export CLAS_RAW=/path/to/CLAS        # contains Block_details/ and Participants/
#   export CATSA_RAW=/path/to/CATSA      # contains Sub1/ ... Sub50/
#   bash scripts/prepare_data.sh
#
# Any of the three can be omitted; the corresponding dataset is then skipped.
set -euo pipefail

cd "$(dirname "$0")/.."

run() {  # run <name> <src env var> <builder>
    local name=$1 var=$2 script=$3
    local src=${!var:-}
    if [[ -z "$src" ]]; then
        echo "== $name: $var not set, skipping"
        return
    fi
    echo "== $name from $src"
    python "$script" --src "$src" --out "dataset/${name}_125Hz.pkl"
}

run WESAD WESAD_RAW dataset/build_wesad.py
run CLAS  CLAS_RAW  dataset/build_clas.py
run CATSA CATSA_RAW dataset/build_catsa.py

echo
ls -lh dataset/*_125Hz.pkl 2>/dev/null || echo "no segment files were produced"
