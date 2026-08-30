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
when transferring CNN and transformer/foundation models to binary plant-lesion
segmentation. The final work connects (A) a fair DeepLabV3-ResNet50 versus
DINOv3 comparison and controlled DINO adaptation with (B) training-image
selection under limited pixel masks. SAM is now a diagnostic promptability
result, not a promised fine-tuned system. Read `docs/review_response_v1.md`
before changing scope.

## Primary protocol

- Primary task: binary lesion/disease (`all original labels > 0`) vs background
  (`label 0`). `ignore_index=255` remains ignored.
- Primary selection metric: dataset-level foreground/disease IoU on validation.
  Also report mean IoU over the two classes, Dice, precision, recall and pixel AP.
- Do not mix binary results with the exploratory 116-class experiment.
- Tune checkpoints, thresholds and hyperparameters on validation only. Earlier
  exploratory baselines already viewed test, which must be disclosed; do not
  select a model from those numbers or add new test evaluations before the
  final protocol is fixed.
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
  gives 0.6444 ± 0.0007 over three seeds. The paired multi-layer difference is
  0.0075 ± 0.0069 and vanishes on seed 44; describe it as a modest, variable
  mean gain rather than a decisive win.
- Doubling the sampling weight of the smallest-mask train quartile is negative:
  seed-42 disease IoU 0.6498 vs 0.6569, Q1 macro IoU 0.3649 vs 0.3832. Do not
  repeat this image-level oversampling recipe.
- Frozen SAM ViT-B oracle promptability on all 1247 validation images: one
  positive point gives disease IoU 0.3624; exact GT box gives 0.4659. Box recall
  is 0.8835 but precision 0.4964. These are upper-bound diagnostics, not
  automatic segmentation results.
- Seed-42 label-efficiency curves are complete for 1/5/10/25/50% budgets with
  matched optimizer steps. Repeats at 10/25% are complete: stratified minus
  random is only +0.0015/+0.0025, while farthest-first minus random is
  -0.0159/-0.0097 (paired means over three runs). Do not turn the 95% crossing
  into a sharp claim because its margin is much smaller than run-to-run noise.
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
- External healthy false-alarm check: PlantWild v1 manifest at
  `data/plantwild_hf/healthy_manifest.json`; all 1556 official healthy-test
  images are downloaded. Keep this result separate from in-domain PlantSeg.
- Run artifacts live in `runs/` and are intentionally gitignored. Each run must
  contain resolved config, environment, JSONL metrics and `best.pt`.
- When syncing, preserve directory structure and never overwrite `data/`,
  `models/`, `runs/`, `.venv/`, or user files. Prefer separate rsync calls such
  as `rsync -az src/ HOST:PROJECT/src/`.

## Current experiment queue

1. Finish three pretrained DeepLabV3-ResNet50 runs plus one scratch control.
2. Run matched DINOv3 resolution controls at 512 and 768 px.
3. Add 10,000-repeat paired image bootstrap, log-budget AULC excluding the
   shared 100% endpoint, selected-set composition, visual panels and a manual
   taxonomy of 50--100 hard cases. `scripts/run_final_postprocessing.sh` is
   queued after core training and also evaluates all resolutions on each
   annotation's original pixel grid.
4. Record compute cost, rewrite the report according to supervisor review and
   only then decide whether one final test evaluation is defensible.

One additional, bounded data-selection experiment is queued after all mandatory
postprocessing: spherical k-means cluster representatives at 10% and 25% for
seeds 42/43/44. It directly tests whether typical representatives avoid the
outlier bias observed for farthest-first; do not expand this into a broad grid.

Do not spend the deadline window on SAM fine-tuning, DINO→SAM, uncertainty,
partial unfreezing, ViT-L+conv or a broad clustering grid unless all mandatory
review items are finished.

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
  the VM. Title-page metadata in `report/metadata.tex` was copied from the
  submitted KT-1 report; do not replace it with placeholders.

## Repository hygiene

Do not commit datasets, weights, checkpoints, logs or generated visualization
bulk. Commit source/config/test/doc changes and concise experiment summaries.
The old local workspace material was moved recoverably to
`/home/iamnifer/argus-archive-20260908`; do not delete it without an explicit
request. The filtered all-branch clone of the colleague repository is at
`/home/iamnifer/argus-archive-20260908/reference-repositories/plant-diseases-segmentation`.
Preserve unrelated user changes in both local and VM worktrees.
