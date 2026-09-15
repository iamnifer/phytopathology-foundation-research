#!/usr/bin/env bash
set -euo pipefail

source "$(dirname -- "${BASH_SOURCE[0]}")/run_helpers.sh"
cd "${project_root}"
require_python

for seed in 42 43 44; do
  "${python_bin}" -m phytopathology.train \
    --config configs/resnet50_binary_linear_probe_native48_cosine.yaml \
    --seed "$seed"
done
