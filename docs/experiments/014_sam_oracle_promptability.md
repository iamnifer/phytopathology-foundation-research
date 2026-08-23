# Experiment 014 — vanilla SAM oracle promptability

Дата: 2026-08-23. Статус: завершён.

## Протокол

- SAM ViT-B (`facebook/sam-vit-base`), веса загружены с ModelScope;
- все 1247 validation-изображений PlantSeg;
- одна положительная точка внутри GT либо точный GT bounding box;
- из трёх масок выбирается вариант с максимальным predicted IoU самой SAM;
- параметры SAM не обучаются; test не используется.

Это диагностическая верхняя оценка при известном расположении заболевания, а
не автоматическая disease segmentation.

## Результаты

| Oracle prompt | Disease IoU | mIoU | Dice | Pixel AP | Macro image IoU |
|---|---:|---:|---:|---:|---:|
| Positive point | 0.3624 | 0.5934 | 0.5320 | 0.4689 | 0.3096 |
| Exact box | 0.4659 | 0.6255 | 0.6357 | 0.6168 | 0.4887 |

Box даёт существенно более сильный сигнал, но остаётся заметно хуже frozen
DINOv3 decoder (`0.6519 ± 0.0062`). Высокий recall box-режима (`0.8835`) при
низкой precision (`0.4964`) показывает систематическое включение здоровой части
листа или фона внутрь маски. Fine-tuning mask decoder содержательно оправдан;
oracle point не следует использовать как сильный baseline.
