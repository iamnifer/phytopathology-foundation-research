# Experiment 007 — binary threshold calibration

Дата: 2026-08-23. Статус: завершён, validation only.

## Вопрос

Скрывает ли высокий pixel AP (`0.8763`) заметный запас качества, который можно
получить без переобучения простой заменой порога foreground probability?

## Протокол

- checkpoint: `runs/20260908T155007Z_dinov3_vitb16_binary_conv/best.pt`;
- split: validation, 1247 изображений;
- 91 порог от 0.05 до 0.95;
- выбор только по dataset-level foreground IoU;
- scores накапливались потоково в 2048 bins;
- test не использовался.

Команда:

```bash
python -m phytopathology.tune_threshold \
  --config configs/dinov3_vitb16_binary_conv.yaml \
  --checkpoint runs/20260908T155007Z_dinov3_vitb16_binary_conv/best.pt
```

## Результат

| Порог | foreground IoU | mean IoU | precision | recall |
|---:|---:|---:|---:|---:|
| 0.50 | 0.6345 | **0.7627** | 0.7699 | 0.7829 |
| **0.45** | **0.6362** | 0.7620 | 0.7504 | **0.8070** |

Прирост foreground IoU равен лишь `+0.0017`, а mean IoU немного ухудшается.
Калибровка не объясняет разрыв до disease IoU около 0.70. Порог не переносился
на test; дальнейший приоритет — признаки, decoder и обучение.
