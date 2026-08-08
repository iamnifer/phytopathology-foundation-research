# 005 — DINOv3 ViT-L/16 binary linear probe

## Задача

Измерить эффект увеличения только backbone с ViT-B/16 до ViT-L/16. Остальные
условия полностью совпадают с экспериментом 004.

## Конфигурация

- Dataset: PlantSeg v3, train/val/test `7916 / 1247 / 2295`.
- Backbone: DINOv3 ViT-L/16, веса ModelScope, frozen.
- Decoder: `1x1 Conv`, input `384x384`, batch 16, seed 42.
- AdamW, LR `1e-3`, weight decay `1e-4`, 10 эпох, bfloat16 autocast.
- Checkpoint selection: maximum validation foreground IoU.
- Run: `20260908T153621Z_dinov3_vitl16_binary_linear`.
- Commit: `a9f2390ca2d57645b210576401bf0e7e4f53a397`.
- GPU: NVIDIA A100-SXM4-80GB; PyTorch `2.14.0+cu130`.

## Результат

Лучший validation checkpoint получен на эпохе 5.

| Модель | Split | Mean IoU | Foreground IoU | Dice | Precision | Recall | Pixel AP |
|---|---|---:|---:|---:|---:|---:|---:|
| ViT-B/16 | val | 0.7391 | 0.5944 | — | — | — | — |
| ViT-L/16 | val | 0.7432 | 0.6040 | 0.7531 | 0.7586 | 0.7477 | 0.8429 |
| ViT-B/16 | test | 0.7431 | 0.6036 | 0.7528 | 0.7871 | 0.7214 | 0.8428 |
| ViT-L/16 | test | 0.7453 | 0.6112 | 0.7587 | 0.7591 | 0.7583 | 0.8432 |

На test увеличение backbone дало `+0.0076` foreground IoU и `+0.0059` Dice.
Pixel AP практически не изменился (`+0.0004`): ранжирование пикселей сходно,
а различие в IoU связано в том числе с рабочим threshold, заданным argmax.
ViT-L повысил recall на `+0.0369`, но снизил precision на `-0.0280`.

## Вывод

Масштабирование backbone даёт воспроизводимый, но небольшой прирост и само по
себе не приближает foreground IoU к 0.70. Следующие более перспективные
факторы — decoder, сохранение aspect ratio/crop вместо растяжения в квадрат,
loss для дисбаланса и частичное размораживание backbone. Для статистического
вывода лучшую конфигурацию затем нужно повторить минимум на трёх seeds.

Лучшие, медианные и худшие примеры сохранены в `errors_test/` каталога запуска.
