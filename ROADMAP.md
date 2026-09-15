# Project status and future work

The coursework and its planned experimental programme were completed in
September 2026. The final report, code, configurations, and experiment records
are available in this repository.

## Completed scope

- Replaced the original exploratory notebook with a reusable Python package and
  configuration-driven command-line workflow.
- Established a binary PlantSeg protocol with lesion IoU as the primary metric.
- Compared frozen DINOv3 ViT-B/16 decoders, intermediate features, optimization
  schedules, and input resolutions.
- Trained DeepLabV3-ResNet50 from ImageNet initialization across three seeds and
  ran a fixed-recipe random-initialization control.
- Compared frozen DINOv3 and ResNet-50 encoders using the same linear decoder,
  feature grid, preprocessing, optimizer, and seeds.
- Evaluated fixed central configurations on the test split and computed paired
  image-level bootstrap intervals.
- Evaluated vanilla SAM with oracle point and box prompts.
- Analysed errors by lesion size and manually annotated 64 difficult examples.
- Measured false positives on healthy PlantWild v1 images as an external
  domain-shift diagnostic.
- Built annotation-efficiency curves and compared random, stratified,
  farthest-first, and spherical k-means subset selection.
- Produced a self-contained XeLaTeX report and a concise machine-readable
  result summary.

The exact completion audit against the supervisor review is preserved in
[`docs/final_audit_2026-09-15.md`](docs/final_audit_2026-09-15.md).

## Possible extensions

These are research directions, not missing requirements of the submitted work:

1. Fine-tune the last DINOv3 blocks or use parameter-efficient adapters while
   retaining the matched multi-seed protocol.
2. Test ViT-L/16 with the same convolutional decoder used by ViT-B/16 to
   separate encoder scale from decoder capacity.
3. Combine uncertainty and diversity for sequential annotation after a small
   labelled warm start.
4. Replace global image descriptors with lesion-aware or patch-level selection
   features.
5. Fine-tune SAM's mask decoder and evaluate prompts that do not depend on
   ground-truth masks.
6. Extend the binary task to multiclass recognition of the original disease
   labels and study domain adaptation explicitly.
7. Repeat the external healthy-image evaluation after calibration or domain
   adaptation.

Any extension should keep model selection on validation, report at least three
seeds for central claims, preserve negative results, and avoid further tuning on
the final test split.
