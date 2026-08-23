# Experiment 008 — preprocessing and augmentation ablation

Дата: 2026-08-23. Статус: завершён, validation only.

## Вопрос

Переносятся ли на frozen DINOv3 выводы коллеги о пользе aspect-preserving
`LongestMaxSize + Pad` и умеренного `spatial_color_light` augmentation preset?

Во всех запусках неизменны PlantSeg split, seed 42, ViT-B/16, frozen backbone,
convolutional head, 384 px, AdamW, LR 1e-3 и 10 эпох. Сначала менялся только
stretch → pad, затем к pad добавлялись spatial/color transforms.

## Результаты

| Конфигурация | Лучший epoch | foreground IoU | mean IoU | pixel AP |
|---|---:|---:|---:|---:|
| stretch + horizontal flip (experiment 006) | 8 | **0.6345** | 0.7627 | **0.8763** |
| pad + horizontal flip | 6 | 0.6231 | 0.7732 | 0.8591 |
| pad + spatial/color light | 9 | 0.6248 | **0.7748** | 0.8553 |

Run directories на ВМ:

- `runs/20260908T170127Z_dinov3_vitb16_binary_conv_pad`;
- `runs/20260908T170740Z_dinov3_vitb16_binary_conv_pad_aug`.

## Вывод

Aspect-preserving padding ухудшил главную метрику на 1.13 п.п. Умеренные
аугментации вернули только 0.16 п.п. и остались на 0.97 п.п. ниже baseline.
Рост mean IoU при падении disease IoU вызван добавлением большого количества
лёгких background pixels в padding и ещё раз показывает недостаточность mean
IoU как единственной метрики. Для следующих DINOv3 запусков сохраняется stretch;
рецепт коллеги не переносится механически между архитектурами.
