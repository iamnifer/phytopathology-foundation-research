# Experiment 009 — four-layer DINOv3 convolutional decoder

Дата: 2026-08-23. Статус: завершён, validation only; follow-up — experiment 011.

## Гипотеза

Промежуточные DINOv3 features содержат пространственную информацию, которую
теряет decoder только по последнему слою. Использованы outputs блоков 3, 6, 9
и 12 ViT-B, каждый пропущен через финальный frozen LayerNorm; patch features
конкатенированы (`4 × 768` channels) перед прежним `3×3 conv → GELU → 1×1 conv`.

Остальные параметры совпадают с experiment 006: stretch 384, horizontal flip,
frozen backbone, AdamW LR 1e-3, seed 42, 10 эпох.

## Результат

| Decoder input | Лучший epoch | foreground IoU | mean IoU | pixel AP |
|---|---:|---:|---:|---:|
| последний слой | 8 | 0.6345 | 0.7627 | **0.8763** |
| слои 3/6/9/12 | 10 | **0.6357** | **0.7648** | 0.8678 |

Run: `runs/20260908T171539Z_dinov3_vitb16_binary_multilayer_conv`.

Прирост foreground IoU `+0.0012` слишком мал для вывода о превосходстве.
Однако лучший checkpoint пришёлся на последнюю эпоху и train loss продолжал
снижаться. Уточняющий 30-эпоховый запуск с cosine decay в experiment 011 поднял
foreground IoU до `0.6569`. Test до завершения выбора конфигурации не используется.
