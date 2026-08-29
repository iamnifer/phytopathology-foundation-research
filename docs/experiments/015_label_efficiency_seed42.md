# Label efficiency: cold-start selection, seed 42

## Question

Can frozen DINOv3 image descriptors identify a subset of fully annotated
PlantSeg images that trains a better lesion segmenter than a uniformly random
subset of the same size?

## Fixed protocol

- PlantSeg binary disease/background segmentation;
- frozen DINOv3 ViT-B/16, features from blocks 3/6/9/12, convolutional decoder;
- 384x384 stretched inputs, 30 epochs, AdamW and cosine learning-rate decay;
- budgets are complete pixel masks for 1/5/10/25/50% of the 7916-image train
  pool: 80, 396, 792, 1979 and 3958 images;
- exactly 7916 sampled images per epoch for every budget, so the number of
  optimizer updates is matched;
- nested subsets within each method, selection seed 42;
- checkpoint chosen by validation disease IoU; test was not evaluated.

The compared methods are uniform random sampling, stratified random sampling
using free image-level metadata, and greedy farthest-first selection in the
space of L2-normalized mean DINOv3 patch descriptors.

## Results

| Budget | Images | Random | Stratified random | Farthest-first |
|---:|---:|---:|---:|---:|
| 1% | 80 | 0.4888 | **0.5256** | 0.5124 |
| 5% | 396 | 0.5833 | **0.5850** | 0.5685 |
| 10% | 792 | 0.6047 | **0.6054** | 0.5906 |
| 25% | 1979 | 0.6244 | **0.6246** | 0.6179 |
| 50% | 3958 | **0.6388** | 0.6361 | 0.6316 |
| 100% | 7916 | **0.6569** | **0.6569** | **0.6569** |

Values are best dataset-level validation disease IoU. The 100% endpoint is the
matching seed-42 full-data run. Its three-seed mean is 0.6519 +/- 0.0062, but it
is not mixed into this single-seed controlled curve.

At the selected checkpoints, the 50% runs obtained:

| Method | Best epoch | mIoU | Disease IoU | Dice | Precision | Recall | Pixel AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random | 20 | 0.7654 | **0.6388** | 0.7796 | 0.7709 | 0.7884 | 0.8597 |
| Stratified random | 16 | 0.7631 | 0.6361 | 0.7776 | 0.7628 | 0.7929 | **0.8599** |
| Farthest-first | 18 | 0.7596 | 0.6316 | 0.7742 | 0.7545 | **0.7949** | 0.8526 |

Trapezoidal normalized area under the 1--100% learning curve, after dividing
IoU by the seed-42 full-data score, is 0.9612 for random, 0.9612 for stratified
random and 0.9527 for farthest-first. Random and stratified random first cross
95% of the full-data IoU at 25%; farthest-first crosses it at 50%.

## Interpretation

This first curve provides two useful but still preliminary findings.

1. Half of the masks recover 97.2% of full-data disease IoU with random
   sampling. A quarter already recovers 95.1%. The annotation-efficiency curve
   itself is therefore a stronger result than the ranking of selection methods.
2. Greedy coverage of global mean DINOv3 descriptors does not beat random
   sampling. It is worse by 1.5--1.7 points at 5--10% and remains below both
   random baselines at 25--50%. This is consistent with farthest-first selecting
   visually unusual images or outliers rather than representative lesion
   patterns.

Stratification has a large apparent advantage at 1%, but it is effectively tied
with random at 5--25% and loses by 0.26 points at 50%. Because subset selection
has only one seed, these small differences must not be presented as stable.
The 1% point is particularly high-variance: each selected image is repeated
about 99 times per epoch under the matched-step protocol.

## Next validation

- repeat random, stratified and farthest-first at the decision-relevant 10% and
  25% budgets for seeds 43 and 44;
- add representative selection (k-means centroids or k-medoids), which tests a
  different hypothesis from outlier-seeking farthest-first;
- inspect selected-set disease composition, lesion-size distribution and
  descriptor-to-nearest-selected distances before designing an
  uncertainty-plus-diversity method;
- keep the test split closed until the method and budget are fixed.

## Artifacts

The full histories, resolved configs and checkpoints remain on the research VM
under `runs/20260914T*_dinov3_vitb16_binary_multilayer_conv_cosine_*_seed42_*`.
Only the concise result and protocol are tracked in Git.
