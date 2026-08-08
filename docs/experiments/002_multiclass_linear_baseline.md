# Experiment 002 — multiclass frozen linear baseline

Дата: 2026-08-08. Статус: completed.

## Протокол

- PlantSeg v3: 7916 train / 1247 validation / 2295 test.
- 116 классов: background `0` + disease IDs `1..115`.
- DINOv3 ViT-B/16, frozen; linear 1×1 decoder.
- 384×384, batch 16, AdamW, LR `1e-3`, weight decay `1e-4`, bf16.
- Seed 42, 10 epochs; run ID `20260908T145639Z_dinov3_vitb16_linear`.
- Код на старте запуска: commit `adb93d9`.

## Validation

| Epoch | Train loss | Val mIoU | Pixel accuracy |
|---:|---:|---:|---:|
| 1 | 1.0131 | 0.1896 | 0.8418 |
| 2 | 0.5006 | 0.2639 | 0.8589 |
| 3 | 0.4177 | 0.3043 | 0.8665 |
| 4 | 0.3748 | 0.3228 | 0.8702 |
| 5 | 0.3485 | 0.3401 | 0.8742 |
| 6 | 0.3291 | 0.3372 | 0.8735 |
| 7 | 0.3138 | 0.3491 | 0.8759 |
| 8 | 0.3005 | 0.3622 | 0.8758 |
| 9 | 0.2902 | 0.3657 | 0.8776 |
| 10 | 0.2830 | **0.3694** | 0.8769 |

## Test (однократно для лучшего checkpoint)

| mIoU | Pixel accuracy | Loss |
|---:|---:|---:|
| **0.3902** | 0.8774 | 0.3921 |

Это честный dataset-level mIoU из общей confusion matrix; присутствуют все 116
классов. Результат ниже опубликованного PlantSeg SegNeXt baseline 0.4452, но
получен простой linear head за ~12 минут и задаёт воспроизводимую нижнюю границу.

Следующие осмысленные улучшения: convolutional/multi-layer decoder, crop-based
augmentation вместо искажающего aspect ratio resize, scheduler и частичное
размораживание backbone. Сравнение с целью 0.7 допустимо только после фиксации,
является ли эта цель multiclass или binary.
