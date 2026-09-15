#!/usr/bin/env bash
set -euo pipefail

source "$(dirname -- "${BASH_SOURCE[0]}")/run_helpers.sh"
cd "${project_root}"
require_python

mkdir -p artifacts/bootstrap
for seed in 42 43 44; do
  resnet_run=$(find_complete_run resnet50_binary_linear_probe_cosine "${seed}")
  dino_run=$(find_complete_run dinov3_vitb16_binary_linear_probe_cosine "${seed}")
  for run in "${resnet_run}" "${dino_run}"; do
    "${python_bin}" -m phytopathology.analyze_predictions \
      --config "${run}/config.yaml" \
      --checkpoint "${run}/best.pt" \
      --split val \
      --output-dir "${run}/analysis_val_counts"
  done
  "${python_bin}" -m phytopathology.bootstrap_compare \
    --baseline "${resnet_run}/analysis_val_counts/per_image.csv" \
    --candidate "${dino_run}/analysis_val_counts/per_image.csv" \
    --iterations 10000 \
    --seed 2026 \
    --output "artifacts/bootstrap/dino_vs_resnet_linear_probe_seed${seed}.json"

  resnet_native_run=$(find_complete_run resnet50_binary_linear_probe_native48_cosine "${seed}")
  "${python_bin}" -m phytopathology.analyze_predictions \
    --config "${resnet_native_run}/config.yaml" \
    --checkpoint "${resnet_native_run}/best.pt" \
    --split val \
    --output-dir "${resnet_native_run}/analysis_val_counts"
  "${python_bin}" -m phytopathology.bootstrap_compare \
    --baseline "${resnet_native_run}/analysis_val_counts/per_image.csv" \
    --candidate "${resnet_run}/analysis_val_counts/per_image.csv" \
    --iterations 10000 \
    --seed 2026 \
    --output "artifacts/bootstrap/resnet24_vs_resnet48_seed${seed}.json"
done
