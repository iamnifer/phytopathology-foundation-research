#!/usr/bin/env bash
set -euo pipefail

cd /home/iamnifer/phytopathology-foundation-research

for seed in 42 43 44; do
  .venv/bin/python -m phytopathology.select_subsets \
    --features features/dinov3_vitb16_train_mean_patches.npz \
    --metadata data/plantseg_v3/Metadatav2.csv \
    --output-dir "subsets/seed${seed}" \
    --methods kmeans \
    --budgets 0.1 0.25 \
    --seed "${seed}"
  scripts/run_label_efficiency.sh kmeans "${seed}" 0.1 0.25
done
