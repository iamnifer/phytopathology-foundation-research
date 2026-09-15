# Repository guidance for coding agents

This is a completed research repository for binary plant-lesion segmentation
with DINOv3, DeepLabV3-ResNet50, and SAM. Start with `README.md` and
`docs/README.md`. Read the relevant record under `docs/experiments/` before
changing an experiment or result.

## Research invariants

- Primary task: lesion/disease (all original labels above 0) versus background
  (label 0); `ignore_index=255` is excluded from metrics.
- Primary selection metric: dataset-level lesion IoU on validation.
- Keep the exploratory 116-class experiment separate from binary results.
- Do not tune checkpoints, thresholds, or hyperparameters on test.
- Central claims use seeds 42, 43, and 44 and report mean plus sample standard
  deviation. Preserve null and negative results.
- The practical DeepLabV3 versus DINOv3 comparison uses different adaptation
  regimes. The frozen linear-probe comparison controls the adaptation regime,
  but pretraining data and objectives still differ.
- SAM point and box results use prompts derived from ground truth and are oracle
  diagnostics, not automatic segmentation results.

## Development workflow

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,download]'
ruff check .
pytest -q
```

Run training through `python -m phytopathology.train` and a checked-in YAML
configuration. Each completed experiment should have a concise record under
`docs/experiments/` containing its question, fixed protocol, result, and
interpretation.

Build the report from the repository root with:

```bash
./scripts/build_report.sh
```

## Repository hygiene

- Do not commit datasets, downloaded model weights, checkpoints, complete run
  folders, cached descriptors, or bulk generated artefacts.
- Keep small derived results in `results/` and experiment records in `docs/`.
- Preserve the archived notebook as provenance; it is not the production
  training entry point.
- Avoid absolute paths, credentials, hostnames, and machine-specific setup in
  tracked files.
- Do not change reported values unless the source experiment record and the
  final report are updated consistently.
