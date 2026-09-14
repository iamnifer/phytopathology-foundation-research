#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "${project_root}"

for seed in 42 43 44; do
  .venv/bin/python -m phytopathology.train \
    --config configs/resnet50_binary_linear_probe_native48_cosine.yaml \
    --seed "$seed"
done
