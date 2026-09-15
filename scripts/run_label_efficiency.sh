#!/usr/bin/env bash
set -euo pipefail

source "$(dirname -- "${BASH_SOURCE[0]}")/run_helpers.sh"
cd "${project_root}"
require_python

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 METHOD SEED [BUDGET ...]" >&2
  exit 2
fi

method=$1
seed=$2
shift 2
if [[ $# -eq 0 ]]; then
  budgets=(0.01 0.05 0.1 0.25 0.5)
else
  budgets=("$@")
fi
case "$method" in
  random|stratified|farthest|kmeans) ;;
  *) echo "Unknown method: $method" >&2; exit 2 ;;
esac

for budget in "${budgets[@]}"; do
  PYTHONPATH=src "${python_bin}" -m phytopathology.train \
    --config configs/dinov3_vitb16_binary_multilayer_conv_cosine.yaml \
    --subset-file "subsets/seed${seed}/${method}_seed${seed}_${budget}.json" \
    --epoch-samples 7916 \
    --seed "$seed"
done
