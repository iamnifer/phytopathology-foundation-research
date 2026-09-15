#!/usr/bin/env bash
set -euo pipefail

source "$(dirname -- "${BASH_SOURCE[0]}")/run_helpers.sh"
cd "${project_root}"
require_python

evaluate_resized() {
  local run=$1
  local metrics="${run}/confirmatory_test_resized_metrics.json"
  local analysis="${run}/confirmatory_test_resized_counts"
  if [[ ! -f "${metrics}" ]]; then
    "${python_bin}" -m phytopathology.evaluate \
      --config "${run}/config.yaml" \
      --checkpoint "${run}/best.pt" \
      --split test \
      --output "${metrics}"
  fi
  if [[ ! -f "${analysis}/per_image.csv" ]]; then
    "${python_bin}" -m phytopathology.analyze_predictions \
      --config "${run}/config.yaml" \
      --checkpoint "${run}/best.pt" \
      --split test \
      --output-dir "${analysis}"
  fi
}

evaluate_native() {
  local run=$1
  local output="${run}/confirmatory_test_native"
  if [[ ! -f "${output}/summary.json" || ! -f "${output}/per_image.csv" ]]; then
    "${python_bin}" -m phytopathology.evaluate_native \
      --config "${run}/config.yaml" \
      --checkpoint "${run}/best.pt" \
      --split test \
      --output-dir "${output}"
  fi
}

mkdir -p artifacts/bootstrap
for seed in 42 43 44; do
  dino_practical=$(find_complete_run dinov3_vitb16_binary_multilayer_conv_cosine "${seed}")
  cnn_practical=$(find_complete_run deeplabv3_resnet50_binary_pretrained "${seed}")
  dino_probe=$(find_complete_run dinov3_vitb16_binary_linear_probe_cosine "${seed}")
  resnet_probe=$(find_complete_run resnet50_binary_linear_probe_cosine "${seed}")

  evaluate_resized "${dino_practical}"
  evaluate_resized "${cnn_practical}"
  evaluate_native "${dino_practical}"
  evaluate_native "${cnn_practical}"
  "${python_bin}" -m phytopathology.bootstrap_compare \
    --baseline "${dino_practical}/confirmatory_test_native/per_image.csv" \
    --candidate "${cnn_practical}/confirmatory_test_native/per_image.csv" \
    --iterations 10000 \
    --seed 2026 \
    --output "artifacts/bootstrap/test_native_cnn_vs_dino_seed${seed}.json"

  evaluate_resized "${dino_probe}"
  evaluate_resized "${resnet_probe}"
  "${python_bin}" -m phytopathology.bootstrap_compare \
    --baseline "${resnet_probe}/confirmatory_test_resized_counts/per_image.csv" \
    --candidate "${dino_probe}/confirmatory_test_resized_counts/per_image.csv" \
    --iterations 10000 \
    --seed 2026 \
    --output "artifacts/bootstrap/test_dino_vs_resnet_probe_seed${seed}.json"
done

"${python_bin}" -m phytopathology.summarize_confirmatory_test \
  --runs-root runs \
  --bootstrap-root artifacts/bootstrap \
  --output artifacts/confirmatory_test_summary.json
