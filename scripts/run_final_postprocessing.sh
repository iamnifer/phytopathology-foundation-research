#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "${project_root}"

declare -A single_runs=(
  [42]="runs/20260914T114031Z_dinov3_vitb16_binary_conv_cosine"
  [43]="runs/20260914T122404Z_dinov3_vitb16_binary_conv_cosine"
  [44]="runs/20260914T124601Z_dinov3_vitb16_binary_conv_cosine"
)
declare -A multilayer_runs=(
  [42]="runs/20260908T172349Z_dinov3_vitb16_binary_multilayer_conv_cosine"
  [43]="runs/20260914T110651Z_dinov3_vitb16_binary_multilayer_conv_cosine"
  [44]="runs/20260914T112340Z_dinov3_vitb16_binary_multilayer_conv_cosine"
)
declare -A cnn_runs=(
  [42]="runs/20260916T131956Z_deeplabv3_resnet50_binary_pretrained"
  [43]="runs/20260916T135548Z_deeplabv3_resnet50_binary_pretrained"
  [44]="runs/20260916T143136Z_deeplabv3_resnet50_binary_pretrained"
)

for seed in 42 43 44; do
  for run in "${single_runs[$seed]}" "${multilayer_runs[$seed]}"; do
    .venv/bin/python -m phytopathology.analyze_predictions \
      --config "${run}/config.yaml" \
      --checkpoint "${run}/best.pt" \
      --split val \
      --output-dir "${run}/analysis_val_counts"
  done
  .venv/bin/python -m phytopathology.bootstrap_compare \
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
  .venv/bin/python -m phytopathology.analyze_predictions \
    --config "${cnn_run}/config.yaml" \
    --checkpoint "${cnn_run}/best.pt" \
    --split val \
    --output-dir "${cnn_run}/analysis_val_counts"
  .venv/bin/python -m phytopathology.bootstrap_compare \
    --baseline "${multilayer_runs[$seed]}/analysis_val_counts/per_image.csv" \
    --candidate "${cnn_run}/analysis_val_counts/per_image.csv" \
    --iterations 10000 \
    --seed 2026 \
    --output "artifacts/bootstrap/cnn_vs_dino_seed${seed}.json"
done

.venv/bin/python -m phytopathology.analyze_predictions \
  --config "${scratch_seed42}/config.yaml" \
  --checkpoint "${scratch_seed42}/best.pt" \
  --split val \
  --output-dir "${scratch_seed42}/analysis_val_counts"
.venv/bin/python -m phytopathology.bootstrap_compare \
  --baseline "${scratch_seed42}/analysis_val_counts/per_image.csv" \
  --candidate "${cnn_seed42}/analysis_val_counts/per_image.csv" \
  --iterations 10000 \
  --seed 2026 \
  --output artifacts/bootstrap/cnn_pretrained_vs_scratch_seed42.json

.venv/bin/python -m phytopathology.render_error_atlas \
  --config "${dino_seed42}/config.yaml" \
  --checkpoint "${dino_seed42}/best.pt" \
  --per-image "${dino_seed42}/analysis_val_counts/per_image.csv" \
  --split val \
  --count 64 \
  --per-page 8 \
  --output-dir artifacts/error_atlas/dino_seed42

.venv/bin/python -m phytopathology.render_dino_sam_panel \
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
  .venv/bin/python -m phytopathology.evaluate_healthy \
    --config "${run}/config.yaml" \
    --checkpoint "${run}/best.pt" \
    --manifest data/plantwild_hf/healthy_manifest.json \
    --data-root data/plantwild_hf \
    --split test \
    --output-dir "artifacts/healthy/${model}_seed42"
done

for run in "${dino_seed42}" "${cnn_seed42}"; do
  .venv/bin/python -m phytopathology.evaluate_native \
    --config "${run}/config.yaml" \
    --checkpoint "${run}/best.pt" \
    --split val \
    --output-dir "${run}/native_val"
done

for size in 512 768; do
  run=$(find runs -maxdepth 1 -type d \
    -name "*_dinov3_vitb16_binary_multilayer_conv_cosine_${size}" | sort | tail -n 1)
  .venv/bin/python -m phytopathology.evaluate_native \
    --config "${run}/config.yaml" \
    --checkpoint "${run}/best.pt" \
    --split val \
    --output-dir "${run}/native_val"
done

.venv/bin/python -m phytopathology.bootstrap_compare \
  --baseline "${dino_seed42}/native_val/per_image.csv" \
  --candidate "${cnn_seed42}/native_val/per_image.csv" \
  --iterations 10000 \
  --seed 2026 \
  --output artifacts/bootstrap/cnn_vs_dino_seed42_native.json

for size in 512 768; do
  run=$(find runs -maxdepth 1 -type d \
    -name "*_dinov3_vitb16_binary_multilayer_conv_cosine_${size}" | sort | tail -n 1)
  .venv/bin/python -m phytopathology.bootstrap_compare \
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
  .venv/bin/python -m phytopathology.benchmark_inference \
    --config "${run}/config.yaml" \
    --checkpoint "${run}/best.pt" \
    --batch-size 1 \
    --warmup 20 \
    --repeats 100 \
    --output "artifacts/benchmarks/${label}.json"
done
