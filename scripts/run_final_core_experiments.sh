#!/usr/bin/env bash
set -euo pipefail

source "$(dirname -- "${BASH_SOURCE[0]}")/run_helpers.sh"
cd "${project_root}"
require_python

for seed in 42 43 44; do
  "${python_bin}" -m phytopathology.train \
    --config configs/deeplabv3_resnet50_binary_pretrained.yaml \
    --seed "${seed}"
done

"${python_bin}" -m phytopathology.train \
  --config configs/deeplabv3_resnet50_binary_scratch.yaml \
  --seed 42

"${python_bin}" -m phytopathology.train \
  --config configs/dinov3_vitb16_binary_multilayer_conv_cosine_512.yaml \
  --seed 42

"${python_bin}" -m phytopathology.train \
  --config configs/dinov3_vitb16_binary_multilayer_conv_cosine_768.yaml \
  --seed 42
