#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "${project_root}"

find_complete_run() {
  local name=$1
  local seed=$2
  local candidate
  local selected=""
  while IFS= read -r candidate; do
    if [[ -f "${candidate}/metrics.jsonl" ]] \
      && [[ $(wc -l < "${candidate}/metrics.jsonl") -eq 30 ]] \
      && grep -q "  seed: ${seed}" "${candidate}/config.yaml"; then
      selected=${candidate}
    fi
  done < <(find runs -maxdepth 1 -type d -name "*_${name}" | sort)
  if [[ -z "${selected}" ]]; then
    echo "No complete run found for ${name}, seed ${seed}" >&2
    return 1
  fi
  printf '%s\n' "${selected}"
}

mkdir -p artifacts/bootstrap
for seed in 42 43 44; do
  resnet_run=$(find_complete_run resnet50_binary_linear_probe_cosine "${seed}")
  dino_run=$(find_complete_run dinov3_vitb16_binary_linear_probe_cosine "${seed}")
  for run in "${resnet_run}" "${dino_run}"; do
    .venv/bin/python -m phytopathology.analyze_predictions \
      --config "${run}/config.yaml" \
      --checkpoint "${run}/best.pt" \
      --split val \
      --output-dir "${run}/analysis_val_counts"
  done
  .venv/bin/python -m phytopathology.bootstrap_compare \
    --baseline "${resnet_run}/analysis_val_counts/per_image.csv" \
    --candidate "${dino_run}/analysis_val_counts/per_image.csv" \
    --iterations 10000 \
    --seed 2026 \
    --output "artifacts/bootstrap/dino_vs_resnet_linear_probe_seed${seed}.json"

  resnet_native_run=$(find_complete_run resnet50_binary_linear_probe_native48_cosine "${seed}")
  .venv/bin/python -m phytopathology.analyze_predictions \
    --config "${resnet_native_run}/config.yaml" \
    --checkpoint "${resnet_native_run}/best.pt" \
    --split val \
    --output-dir "${resnet_native_run}/analysis_val_counts"
  .venv/bin/python -m phytopathology.bootstrap_compare \
    --baseline "${resnet_native_run}/analysis_val_counts/per_image.csv" \
    --candidate "${resnet_run}/analysis_val_counts/per_image.csv" \
    --iterations 10000 \
    --seed 2026 \
    --output "artifacts/bootstrap/resnet24_vs_resnet48_seed${seed}.json"
done
