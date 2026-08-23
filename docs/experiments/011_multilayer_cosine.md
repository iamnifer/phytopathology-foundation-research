# Experiment 011 — multi-layer decoder with cosine schedule

Дата: 2026-08-23. Статус: завершён, validation only.

## Гипотеза

Короткий multi-layer run из experiment 009 достиг максимума на последней
эпохе. Проверяется, раскрывается ли decoder из слоёв 3/6/9/12 при 30 эпохах и
плавном снижении LR с `1e-3` до `1e-5`. Другие параметры не менялись: frozen
ViT-B/16, stretch 384, horizontal flip, AdamW, seed 42.

## Результат

Лучший checkpoint выбран по validation foreground IoU на epoch 25 при
`LR = 1.045e-4`.

| Конфигурация | Эпох | foreground IoU | mean IoU | Dice | Precision | Recall | AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| single-layer conv | 10 | 0.6345 | 0.7627 | 0.7764 | 0.7699 | 0.7829 | **0.8763** |
| multi-layer, constant LR | 10 | 0.6357 | 0.7648 | 0.7773 | 0.7864 | 0.7684 | 0.8678 |
| **multi-layer, cosine** | **30** | **0.6569** | **0.7765** | **0.7929** | **0.7713** | **0.8158** | 0.8723 |

Run: `runs/20260908T172349Z_dinov3_vitb16_binary_multilayer_conv_cosine`.

Прирост к прежнему лучшему validation baseline составляет `+0.0224` absolute
foreground IoU. Само объединение промежуточных слоёв на коротком рецепте почти
не помогло; результат появился только вместе с достаточной длительностью и LR
decay. Поэтому архитектуру decoder и optimization schedule следует считать
взаимодействующими факторами.

Validation threshold sweep выбрал 0.48: foreground IoU `0.6571` против `0.6569`
при 0.50 (+0.0002), поэтому стандартный порог сохраняется.

## Изменение профиля ошибок

| Метрика | Single-layer conv | Multi-layer cosine | Изменение |
|---|---:|---:|---:|
| Macro per-image IoU | 0.5352 | **0.5766** | +0.0413 |
| Q1 small-lesion IoU | 0.3247 | **0.3832** | +0.0586 |
| Q2 IoU | 0.4744 | **0.5230** | +0.0486 |
| Q3 IoU | 0.5952 | **0.6316** | +0.0363 |
| Q4 large-lesion IoU | 0.7469 | **0.7687** | +0.0218 |
| Zero-IoU images | 2.25% | **0.80%** | −1.45 п.п. |
| Undersegmented images | 31.28% | **20.45%** | −10.83 п.п. |
| Oversegmented images | **23.42%** | 26.86% | +3.44 п.п. |

Выигрыш максимален на малых поражениях, хотя именно они остаются главным
failure mode. Модель стала полнее захватывать поражение (recall вырос), ценой
умеренного роста oversegmentation.

## Решение

Test не запускался. Следующий обязательный шаг — повторить эту конфигурацию на
двух дополнительных seeds. Если mean/std подтверждают прирост, после этого
проверять lesion-aware sampling/loss и только затем однократно оценивать
зафиксированного победителя на test.
