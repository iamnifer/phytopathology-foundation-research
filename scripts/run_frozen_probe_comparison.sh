#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "${project_root}"

for seed in 42 43 44; do
  .venv/bin/python -m phytopathology.train \
    --config configs/resnet50_binary_linear_probe_cosine.yaml \
    --seed "$seed"
done

for seed in 42 43 44; do
  .venv/bin/python -m phytopathology.train \
    --config configs/dinov3_vitb16_binary_linear_probe_cosine.yaml \
    --seed "$seed"
done
