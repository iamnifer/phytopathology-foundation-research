# 006 — DINOv3 ViT-B/16 binary convolutional decoder

## Задача

Сравнить linear probe с простой нелинейной головой при неизменном frozen
DINOv3 ViT-B/16 и том же binary-протоколе.

## Конфигурация

- Dataset: PlantSeg v3, train/val/test `7916 / 1247 / 2295`.
- Backbone: DINOv3 ViT-B/16, frozen, веса ModelScope.
- Decoder: `3x3 Conv (768 → 256) → GELU → 1x1 Conv (256 → 2)`.
- Input `384x384`, batch 16, seed 42; AdamW, LR `1e-3`, weight decay `1e-4`.
- 10 эпох, bfloat16 autocast; выбор checkpoint по validation foreground IoU.
- Run: `20260908T155007Z_dinov3_vitb16_binary_conv`.
- Commit: `c708858`; GPU: NVIDIA A100-SXM4-80GB.

## Результат

Лучший validation checkpoint получен на эпохе 8.

| Decoder | Split | Mean IoU | Foreground IoU | Dice | Precision | Recall | Pixel AP |
|---|---|---:|---:|---:|---:|---:|---:|
| Linear | val | 0.7391 | 0.5944 | — | — | — | — |
| Conv | val | 0.7627 | 0.6345 | 0.7764 | 0.7699 | 0.7829 | 0.8597 |
| Linear | test | 0.7431 | 0.6036 | 0.7528 | 0.7871 | 0.7214 | 0.8428 |
| Conv | test | 0.7703 | 0.6488 | 0.7870 | 0.7828 | 0.7912 | 0.8763 |

На test convolutional decoder улучшил foreground IoU на `+0.0452`, Dice на
`+0.0342` и pixel AP на `+0.0334`. Это значительно больше прироста от замены
ViT-B на ViT-L при linear decoder (`+0.0076` foreground IoU).

## Вывод

На этом этапе усложнение decoder эффективнее простого масштабирования frozen
backbone. До тяжёлого fine-tuning разумно проверить multi-layer decoder,
корректный resize/crop и loss/crop sampling для дисбаланса. Результат одного
seed пока является ablation, а не статистически подтверждённым финалом.

Лучшие, медианные и худшие test-примеры сохранены в `errors_test/` каталога
запуска на ВМ.
