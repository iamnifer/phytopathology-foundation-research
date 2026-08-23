# Experiment 010 — validation error analysis

Дата: 2026-08-23. Статус: завершён.

## Протокол

Лучший ViT-B conv baseline из experiment 006 оценён per image на validation.
Для каждого изображения сохранены foreground IoU, precision/recall, доля
foreground в GT и prediction, растение и заболевание. Артефакты на ВМ:
`runs/20260908T155007Z_dinov3_vitb16_binary_conv/analysis_val/`.

## Общая диагностика

- 1247 изображений;
- macro foreground IoU: mean `0.5352`, median `0.5732`;
- нулевой foreground IoU: `2.25%` изображений;
- средняя GT/predicted foreground fraction: `0.2033 / 0.2068`;
- predicted/GT area ratio > 1.25: `23.42%` изображений;
- predicted/GT area ratio < 0.75: `31.28%` изображений.

Dataset-level foreground IoU (`0.6345`) выше macro per-image IoU, поскольку
крупные маски имеют больший пиксельный вес.

## Зависимость от размера поражения

| Квартиль GT foreground fraction | Изображений | Mean IoU | Median IoU |
|---|---:|---:|---:|
| Q1, самые маленькие | 312 | **0.3247** | 0.3013 |
| Q2 | 312 | 0.4744 | 0.4901 |
| Q3 | 311 | 0.5952 | 0.6114 |
| Q4, самые большие | 312 | **0.7469** | 0.7887 |

Разрыв Q1→Q4 равен 42.2 п.п. и является главным обнаруженным failure mode.
Среди групп минимум с пятью примерами особенно трудны lettuce mosaic virus
(mean 0.107), ginger sheath blight (0.193), carrot cavity spot (0.254), bell
pepper bacterial spot (0.271) и banana bunchy top (0.271).

## Вывод

Средняя площадь prediction хорошо откалибрована лишь на уровне всего набора;
ошибки конкретных изображений разнонаправленны. Следующая обоснованная линия —
lesion-aware crop sampling, foreground-aware loss или multi-scale decoder для
малых поражений. Увеличение backbone не адресует наблюдаемую проблему напрямую.

Follow-up experiment 011 улучшил macro IoU до `0.5766`, Q1 до `0.3832` и
снизил долю zero-IoU изображений до `0.80%`. Улучшение сильнее всего именно на
малых масках, но разрыв Q1→Q4 всё ещё составляет 38.6 п.п.
