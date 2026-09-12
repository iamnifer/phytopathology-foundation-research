#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "${project_root}"

for seed in 42 43 44; do
  .venv/bin/python -m phytopathology.select_subsets \
    --features features/dinov3_vitb16_train_mean_patches.npz \
    --metadata data/plantseg_v3/Metadatav2.csv \
    --output-dir "subsets/seed${seed}" \
    --methods kmeans \
    --budgets 0.1 0.25 \
    --seed "${seed}"
  .venv/bin/python -m phytopathology.analyze_subsets \
    --metadata data/plantseg_v3/Metadatav2.csv \
    --data-root data/plantseg_v3 \
    --manifests \
      "subsets/seed${seed}/kmeans_seed${seed}_0.1.json" \
      "subsets/seed${seed}/kmeans_seed${seed}_0.25.json" \
    --output-dir "artifacts/subset_analysis_kmeans_seed${seed}" \
    --num-images 16
  scripts/run_label_efficiency.sh kmeans "${seed}" 0.1 0.25
done
