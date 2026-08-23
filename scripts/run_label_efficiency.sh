#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 METHOD SEED" >&2
  exit 2
fi

method=$1
seed=$2
case "$method" in
  random|stratified|farthest) ;;
  *) echo "Unknown method: $method" >&2; exit 2 ;;
esac

for budget in 0.01 0.05 0.1 0.25 0.5; do
  PYTHONPATH=src .venv/bin/python -m phytopathology.train \
    --config configs/dinov3_vitb16_binary_multilayer_conv_cosine.yaml \
    --subset-file "subsets/seed${seed}/${method}_seed${seed}_${budget}.json" \
    --epoch-samples 7916 \
    --seed "$seed"
done
