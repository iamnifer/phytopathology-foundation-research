# Experiment 003 — binary lesion/background smoke test

Дата: 2026-08-08. Статус: pipeline validation.

- Все disease IDs `1..115` отображены в foreground `1`, background остаётся `0`.
- 32 train / 16 validation изображений, 1 эпоха, два optimizer steps.
- Frozen DINOv3 ViT-B/16, linear head, 384×384, batch 16.
- Train loss `0.6201`, validation loss `0.5812`.
- Mean IoU двух классов `0.4243`; background IoU `0.7996`, lesion IoU `0.0491`.

Результат не измеряет качество из-за размера запуска; он подтверждает, что
binary mapping и отдельный конфиг выполняются end-to-end.
