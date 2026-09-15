#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "${project_root}"

declare -A dino_practical=(
  [42]="runs/20260908T172349Z_dinov3_vitb16_binary_multilayer_conv_cosine"
  [43]="runs/20260914T110651Z_dinov3_vitb16_binary_multilayer_conv_cosine"
  [44]="runs/20260914T112340Z_dinov3_vitb16_binary_multilayer_conv_cosine"
)
declare -A cnn_practical=(
  [42]="runs/20260916T131956Z_deeplabv3_resnet50_binary_pretrained"
  [43]="runs/20260916T135548Z_deeplabv3_resnet50_binary_pretrained"
  [44]="runs/20260916T143136Z_deeplabv3_resnet50_binary_pretrained"
)
declare -A dino_probe=(
  [42]="runs/20260918T114823Z_dinov3_vitb16_binary_linear_probe_cosine"
  [43]="runs/20260918T120507Z_dinov3_vitb16_binary_linear_probe_cosine"
  [44]="runs/20260918T122200Z_dinov3_vitb16_binary_linear_probe_cosine"
)
declare -A resnet_probe=(
  [42]="runs/20260918T105553Z_resnet50_binary_linear_probe_cosine"
  [43]="runs/20260918T111317Z_resnet50_binary_linear_probe_cosine"
  [44]="runs/20260918T113045Z_resnet50_binary_linear_probe_cosine"
)

evaluate_resized() {
  local run=$1
  local metrics="${run}/confirmatory_test_resized_metrics.json"
  local analysis="${run}/confirmatory_test_resized_counts"
  if [[ ! -f "${metrics}" ]]; then
    .venv/bin/python -m phytopathology.evaluate \
      --config "${run}/config.yaml" \
      --checkpoint "${run}/best.pt" \
      --split test \
      --output "${metrics}"
  fi
  if [[ ! -f "${analysis}/per_image.csv" ]]; then
    .venv/bin/python -m phytopathology.analyze_predictions \
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
    .venv/bin/python -m phytopathology.evaluate_native \
      --config "${run}/config.yaml" \
      --checkpoint "${run}/best.pt" \
      --split test \
      --output-dir "${output}"
  fi
}

mkdir -p artifacts/bootstrap
for seed in 42 43 44; do
  evaluate_resized "${dino_practical[$seed]}"
  evaluate_resized "${cnn_practical[$seed]}"
  evaluate_native "${dino_practical[$seed]}"
  evaluate_native "${cnn_practical[$seed]}"
  .venv/bin/python -m phytopathology.bootstrap_compare \
    --baseline "${dino_practical[$seed]}/confirmatory_test_native/per_image.csv" \
    --candidate "${cnn_practical[$seed]}/confirmatory_test_native/per_image.csv" \
    --iterations 10000 \
    --seed 2026 \
    --output "artifacts/bootstrap/test_native_cnn_vs_dino_seed${seed}.json"

  evaluate_resized "${dino_probe[$seed]}"
  evaluate_resized "${resnet_probe[$seed]}"
  .venv/bin/python -m phytopathology.bootstrap_compare \
    --baseline "${resnet_probe[$seed]}/confirmatory_test_resized_counts/per_image.csv" \
    --candidate "${dino_probe[$seed]}/confirmatory_test_resized_counts/per_image.csv" \
    --iterations 10000 \
    --seed 2026 \
    --output "artifacts/bootstrap/test_dino_vs_resnet_probe_seed${seed}.json"
done

.venv/bin/python -m phytopathology.summarize_confirmatory_test \
  --runs-root runs \
  --bootstrap-root artifacts/bootstrap \
  --output artifacts/confirmatory_test_summary.json
