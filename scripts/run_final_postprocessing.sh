#!/usr/bin/env bash
set -euo pipefail

source "$(dirname -- "${BASH_SOURCE[0]}")/run_helpers.sh"
cd "${project_root}"
require_python

declare -A single_runs
declare -A multilayer_runs
declare -A cnn_runs
for seed in 42 43 44; do
  single_runs[$seed]=$(find_complete_run dinov3_vitb16_binary_conv_cosine "${seed}")
  multilayer_runs[$seed]=$(
    find_complete_run dinov3_vitb16_binary_multilayer_conv_cosine "${seed}"
  )
  cnn_runs[$seed]=$(find_complete_run deeplabv3_resnet50_binary_pretrained "${seed}")
done

for seed in 42 43 44; do
  for run in "${single_runs[$seed]}" "${multilayer_runs[$seed]}"; do
    "${python_bin}" -m phytopathology.analyze_predictions \
      --config "${run}/config.yaml" \
      --checkpoint "${run}/best.pt" \
      --split val \
      --output-dir "${run}/analysis_val_counts"
  done
  "${python_bin}" -m phytopathology.bootstrap_compare \
    --baseline "${single_runs[$seed]}/analysis_val_counts/per_image.csv" \
    --candidate "${multilayer_runs[$seed]}/analysis_val_counts/per_image.csv" \
    --iterations 10000 \
    --seed 2026 \
    --output "artifacts/bootstrap/multilayer_vs_single_seed${seed}.json"
done

cnn_seed42="${cnn_runs[42]}"
dino_seed42="${multilayer_runs[42]}"
scratch_seed42=$(find runs -maxdepth 1 -type d \
  -name "*_deeplabv3_resnet50_binary_scratch" | sort | tail -n 1)

for seed in 42 43 44; do
  cnn_run="${cnn_runs[$seed]}"
  "${python_bin}" -m phytopathology.analyze_predictions \
    --config "${cnn_run}/config.yaml" \
    --checkpoint "${cnn_run}/best.pt" \
    --split val \
    --output-dir "${cnn_run}/analysis_val_counts"
  "${python_bin}" -m phytopathology.bootstrap_compare \
    --baseline "${multilayer_runs[$seed]}/analysis_val_counts/per_image.csv" \
    --candidate "${cnn_run}/analysis_val_counts/per_image.csv" \
    --iterations 10000 \
    --seed 2026 \
    --output "artifacts/bootstrap/cnn_vs_dino_seed${seed}.json"
done

"${python_bin}" -m phytopathology.analyze_predictions \
  --config "${scratch_seed42}/config.yaml" \
  --checkpoint "${scratch_seed42}/best.pt" \
  --split val \
  --output-dir "${scratch_seed42}/analysis_val_counts"
"${python_bin}" -m phytopathology.bootstrap_compare \
  --baseline "${scratch_seed42}/analysis_val_counts/per_image.csv" \
  --candidate "${cnn_seed42}/analysis_val_counts/per_image.csv" \
  --iterations 10000 \
  --seed 2026 \
  --output artifacts/bootstrap/cnn_pretrained_vs_scratch_seed42.json

"${python_bin}" -m phytopathology.render_error_atlas \
  --config "${dino_seed42}/config.yaml" \
  --checkpoint "${dino_seed42}/best.pt" \
  --per-image "${dino_seed42}/analysis_val_counts/per_image.csv" \
  --split val \
  --count 64 \
  --per-page 8 \
  --output-dir artifacts/error_atlas/dino_seed42

"${python_bin}" -m phytopathology.render_dino_sam_panel \
  --dino-config "${dino_seed42}/config.yaml" \
  --dino-checkpoint "${dino_seed42}/best.pt" \
  --sam-model models/sam-vit-base \
  --per-image "${dino_seed42}/analysis_val_counts/per_image.csv" \
  --output artifacts/comparison/dino_sam_cases.png

for model in dino cnn; do
  if [[ "${model}" == "dino" ]]; then
    run="${dino_seed42}"
  else
    run="${cnn_seed42}"
  fi
  "${python_bin}" -m phytopathology.evaluate_healthy \
    --config "${run}/config.yaml" \
    --checkpoint "${run}/best.pt" \
    --manifest data/plantwild_hf/healthy_manifest.json \
    --data-root data/plantwild_hf \
    --split test \
    --output-dir "artifacts/healthy/${model}_seed42"
done

for run in "${dino_seed42}" "${cnn_seed42}"; do
  "${python_bin}" -m phytopathology.evaluate_native \
    --config "${run}/config.yaml" \
    --checkpoint "${run}/best.pt" \
    --split val \
    --output-dir "${run}/native_val"
done

for size in 512 768; do
  run=$(find runs -maxdepth 1 -type d \
    -name "*_dinov3_vitb16_binary_multilayer_conv_cosine_${size}" | sort | tail -n 1)
  "${python_bin}" -m phytopathology.evaluate_native \
    --config "${run}/config.yaml" \
    --checkpoint "${run}/best.pt" \
    --split val \
    --output-dir "${run}/native_val"
done

"${python_bin}" -m phytopathology.bootstrap_compare \
  --baseline "${dino_seed42}/native_val/per_image.csv" \
  --candidate "${cnn_seed42}/native_val/per_image.csv" \
  --iterations 10000 \
  --seed 2026 \
  --output artifacts/bootstrap/cnn_vs_dino_seed42_native.json

for size in 512 768; do
  run=$(find runs -maxdepth 1 -type d \
    -name "*_dinov3_vitb16_binary_multilayer_conv_cosine_${size}" | sort | tail -n 1)
  "${python_bin}" -m phytopathology.bootstrap_compare \
    --baseline "${dino_seed42}/native_val/per_image.csv" \
    --candidate "${run}/native_val/per_image.csv" \
    --iterations 10000 \
    --seed 2026 \
    --output "artifacts/bootstrap/dino_${size}_vs_384_native.json"
done

declare -A benchmark_runs=(
  [cnn_384]="${cnn_seed42}"
  [dino_384]="${dino_seed42}"
)
for size in 512 768; do
  benchmark_runs["dino_${size}"]=$(find runs -maxdepth 1 -type d \
    -name "*_dinov3_vitb16_binary_multilayer_conv_cosine_${size}" | sort | tail -n 1)
done
for label in cnn_384 dino_384 dino_512 dino_768; do
  run="${benchmark_runs[$label]}"
  "${python_bin}" -m phytopathology.benchmark_inference \
    --config "${run}/config.yaml" \
    --checkpoint "${run}/best.pt" \
    --batch-size 1 \
    --warmup 20 \
    --repeats 100 \
    --output "artifacts/benchmarks/${label}.json"
done
