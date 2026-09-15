#!/usr/bin/env bash
set -euo pipefail

source "$(dirname -- "${BASH_SOURCE[0]}")/run_helpers.sh"
cd "${project_root}"
require_python

for seed in 42 43 44; do
  "${python_bin}" -m phytopathology.train \
    --config configs/resnet50_binary_linear_probe_cosine.yaml \
    --seed "$seed"
done

for seed in 42 43 44; do
  "${python_bin}" -m phytopathology.train \
    --config configs/dinov3_vitb16_binary_linear_probe_cosine.yaml \
    --seed "$seed"
done
