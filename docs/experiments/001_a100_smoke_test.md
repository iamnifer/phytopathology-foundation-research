# Experiment 001 — A100 smoke test

Дата: 2026-08-03. Статус: pipeline validation, не quality experiment.

## Цель

Проверить полный путь `PlantSeg → preprocessing → DINOv3 → loss → backward →
mIoU → checkpoint` на подготовленной ВМ. Маленькая подвыборка не предназначена
для сравнения качества.

## Среда

| Параметр | Значение |
|---|---|
| GPU | NVIDIA A100-SXM4-80GB, 81920 MiB |
| NVIDIA driver | 595.71.05 |
| PyTorch | 2.14.0+cu130 |
| CUDA runtime | 13.0 |
| Python | 3.12.3 |
| Dataset | PlantSeg v3, MD5 verified |
| Weights | DINOv3 ViT-B/16 via ModelScope |

## Конфигурация и результат

- Run ID: `20260908T145357Z_dinov3_vitb16_linear`.
- 32 train / 16 validation images, 1 epoch, batch size 16, 384×384.
- Frozen backbone, linear 1×1 decoder, AdamW, bf16 autocast.
- Train loss: `4.7613`; validation loss: `4.6255`.
- mIoU: `0.0039`; pixel accuracy: `0.0536`.
- Создан decoder checkpoint размером 351 KiB.

Низкая метрика ожидаема после двух optimizer steps и не является baseline.
Значимый результат — pipeline полностью выполняется на CUDA, модули проходят
lint, а unit-тесты метрики проходят (`3 passed`).
