#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "${project_root}"

for seed in 42 43 44; do
  .venv/bin/python -m phytopathology.train \
    --config configs/deeplabv3_resnet50_binary_pretrained.yaml \
    --seed "${seed}"
done

.venv/bin/python -m phytopathology.train \
  --config configs/deeplabv3_resnet50_binary_scratch.yaml \
  --seed 42

.venv/bin/python -m phytopathology.train \
  --config configs/dinov3_vitb16_binary_multilayer_conv_cosine_512.yaml \
  --seed 42

.venv/bin/python -m phytopathology.train \
  --config configs/dinov3_vitb16_binary_multilayer_conv_cosine_768.yaml \
  --seed 42
