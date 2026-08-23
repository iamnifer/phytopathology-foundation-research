# DINOv3 decoder controlled series — 2026-08-23

## Question

Is the validation gain of the four-layer convolutional decoder reproducible,
and does it remain when training duration and cosine scheduling are controlled?

## Fixed protocol

- PlantSeg binary disease/background segmentation;
- frozen DINOv3 ViT-B/16, 384×384 stretched inputs;
- 30 epochs, AdamW, learning rate 0.001, cosine decay to 1e-5;
- checkpoint selected by validation disease/foreground IoU;
- validation and test ground truth were not used to choose the input subset;
- test was not evaluated in this series.

## Results

| Run | Seed | Features | Best epoch | Disease IoU | mIoU | Pixel AP |
|---|---:|---|---:|---:|---:|---:|
| `20260908T172349Z_dinov3_vitb16_binary_multilayer_conv_cosine` | 42 | blocks 3/6/9/12 | 25 | 0.6569 | 0.7765 | 0.8723 |
| `20260914T110651Z_dinov3_vitb16_binary_multilayer_conv_cosine` | 43 | blocks 3/6/9/12 | 25 | 0.6538 | 0.7762 | 0.8754 |
| `20260914T112340Z_dinov3_vitb16_binary_multilayer_conv_cosine` | 44 | blocks 3/6/9/12 | 23 | 0.6450 | 0.7705 | 0.8714 |
| `20260914T114031Z_dinov3_vitb16_binary_conv_cosine` | 42 | last block | 23 | 0.6438 | 0.7703 | 0.8658 |

For the multi-layer decoder, mean disease IoU over seeds 42/43/44 is
**0.6519 ± 0.0062** (sample standard deviation). On the matched seed 42, the
multi-layer decoder exceeds the last-layer control by 0.0131 absolute IoU.

## Interpretation

The strong multi-layer run was not a one-seed accident: all three runs lie in
the 0.6450–0.6569 range. The matched seed-42 control also indicates that the
gain is not explained only by the longer cosine schedule. However, decoder
architecture has not yet been compared statistically because the single-layer
control has only one seed. Seeds 43 and 44 for that control are needed before
claiming a robust architectural advantage.

## Artifacts

Resolved configs, environment snapshots, metrics histories and decoder
checkpoints are stored under the corresponding names in the gitignored `runs/`
directory. The three new runs are backed up both locally and on the research VM.
