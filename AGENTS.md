# Persistent project context

This repository is a coursework/research project on plant-disease semantic
segmentation with DINOv3 and PlantSeg. Read `README.md`, `ROADMAP.md`,
`docs/research_plan.md`, and `docs/colleague_experiments_review.md` before making
research decisions. Record every completed run in `docs/experiments/`.

The official, immutable title supplied by the student is “Сравнение архитектур
CNN и трансформеров для классификации заболеваний растений с применением
трансферного обучения и адаптации доменов”. It is a broad formal umbrella:
semantic segmentation may be treated as pixel classification. Follow the
supervisor's directions—DINO/SAM adaptation quality and data minimization—rather
than adding a separate image-classification track merely to mirror every word.
The final report deadline is 2026-09-18 23:59 Europe/Moscow.

The current research goal is to study the quality/annotation-budget trade-off
when adapting DINOv3 and SAM to binary plant-lesion segmentation. The final
work must connect two lines: (A) DINOv3/SAM adaptation quality and (B) training
image selection with limited pixel masks. See `docs/research_plan.md` for RQs,
budgets, controls, and completion criteria.

## Primary protocol

- Primary task: binary lesion/disease (`all original labels > 0`) vs background
  (`label 0`). `ignore_index=255` remains ignored.
- Primary selection metric: dataset-level foreground/disease IoU on validation.
  Also report mean IoU over the two classes, Dice, precision, recall and pixel AP.
- Do not mix binary results with the exploratory 116-class experiment.
- Tune checkpoints, thresholds and hyperparameters on validation only. The test
  split is locked after the current baselines; evaluate it only for a finalized
  configuration. Never select a model from test numbers.
- Use one controlled change per ablation and preserve null/negative results.
- Before a final comparison, run at least three seeds and report mean ± std.

## Current verified results

All values below use 384×384 stretched inputs unless explicitly noted.

- Frozen DINOv3 ViT-B/16 + linear head: val foreground IoU 0.5944 (epoch 5);
  test foreground IoU 0.6036, mean IoU 0.7431, Dice 0.7528, AP 0.8428.
- Frozen DINOv3 ViT-L/16 + linear head: val foreground IoU 0.6040 (epoch 5);
  test foreground IoU 0.6112, mean IoU 0.7453, Dice 0.7587, AP 0.8432.
- Frozen DINOv3 ViT-B/16 + 3×3 conv head: val foreground IoU 0.6345 (epoch 8);
  test foreground IoU 0.6488, mean IoU 0.7703, Dice 0.7870, AP 0.8763.
- Validation threshold sweep for the conv head: 0.45 gives foreground IoU
  0.6362 vs 0.6345 at 0.50, too small to justify test reuse.
- Frozen ViT-B/16, layers 3/6/9/12 + conv decoder, 30-epoch cosine schedule:
  validation foreground IoU 0.6569 (epoch 25), mean IoU 0.7765, Dice 0.7929,
  precision 0.7713, recall 0.8158, AP 0.8723. Across seeds 42/43/44, foreground
  IoU is 0.6519 ± 0.0062 (sample SD); test not evaluated.
- Matched last-layer convolutional decoder with the same 30-epoch cosine recipe
  gives 0.6438 foreground IoU on seed 42. The matched multi-layer gain is 0.0131,
  but last-layer seeds 43/44 are still required for a statistical comparison.
- Exploratory multiclass ViT-B linear test mIoU: 0.3902. The notebook's saved
  `0.4575` is stale/inconsistent and must not be presented as verified.

The best comparable supervised result found in the colleague repository is
SegNeXt at about 0.701 disease IoU. Our conv probe is already comparable to
their DeepLabV3+/U-Net (~0.646) and the remaining gap is realistic.

## Infrastructure

- Research VM: `ssh -l iamnifer 51.250.85.113` (SSH keys are configured; never
  store credentials in this repository).
- VM project: `/home/iamnifer/phytopathology-foundation-research`.
- Hardware verified 2026-09-08: NVIDIA A100-SXM4-80GB, ~116 GB RAM, ~194 GB
  filesystem with ~169 GB free at setup.
- Python environment: `.venv`; invoke `.venv/bin/python`, `.venv/bin/pytest`,
  and `.venv/bin/ruff` on the VM.
- Data: `data/plantseg_v3`; ModelScope weights:
  `models/dinov3-vitb16-pretrain-lvd1689m` and
  `models/dinov3-vitl16-pretrain-lvd1689m`.
- Run artifacts live in `runs/` and are intentionally gitignored. Each run must
  contain resolved config, environment, JSONL metrics and `best.pt`.
- When syncing, preserve directory structure and never overwrite `data/`,
  `models/`, `runs/`, `.venv/`, or user files. Prefer separate rsync calls such
  as `rsync -az src/ HOST:PROJECT/src/`.

## Current experiment queue

1. Run single-layer cosine controls for seeds 43/44; the multi-layer three-seed
   result and seed-42 matched control are complete and documented in
   `docs/experiments/2026-09-14_decoder_controlled_series.md`.
2. Test one lesion-aware intervention because Q1 small-mask macro IoU remains
   0.3832 vs 0.7687 for Q4: crop/oversampling first, then a loss ablation.
3. Benchmark frozen SAM separately for oracle point/box prompts and automatic
   prompts derived from DINO predictions; never present oracle prompts as an
   automatic baseline. Then train the SAM mask decoder.
4. Extract cached DINOv3 image descriptors and evaluate random, stratified
   random, cluster representatives and farthest-first selection at
   1/5/10/25/50/100% label budgets with fixed optimizer-step rules.
5. Only after the above, test uncertainty+diversity and partial backbone
   unfreezing. Evaluate finalized winners once on test.

## Important findings and pitfalls

- Background dominance makes mean IoU an unsafe sole objective.
- Threshold calibration added only 0.17 percentage points on validation.
- ViT-L scaling helped less than changing the decoder; prioritize readout and
  intermediate features over a still larger frozen backbone.
- The colleague's WSSS/CAM/SAM numbers are not directly comparable to our fully
  supervised protocol. Their reports are synthesized in
  `docs/colleague_experiments_review.md`.
- SAM dense-mask prompting and generic CRF postprocessing can make masks worse;
  both require a defined validation-tuned protocol.
- Original preprocessing stretched every image to a square. New ablations must
  state `resize_mode` and augmentation preset explicitly.
- `report/coursework.tex` has been moved into the repository but still describes
  an earlier stage; synchronize it only after the current experiment series.
- The unchanged legacy notebook remains in the root as provenance, not as the
  production training path. Eventually move it to `notebooks/archive/`.
- The coursework source is autonomous under `report/`: build it with
  `latexmk coursework.tex` from that directory. XeLaTeX/biber are installed on
  the VM. Title-page placeholders live in `report/metadata.tex`.

## Repository hygiene

Do not commit datasets, weights, checkpoints, logs or generated visualization
bulk. Commit source/config/test/doc changes and concise experiment summaries.
The old local workspace material was moved recoverably to
`/home/iamnifer/argus-archive-20260908`; do not delete it without an explicit
request. The filtered all-branch clone of the colleague repository is at
`/home/iamnifer/argus-archive-20260908/reference-repositories/plant-diseases-segmentation`.
Preserve unrelated user changes in both local and VM worktrees.
