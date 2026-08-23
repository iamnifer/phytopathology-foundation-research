# План работы

Обновлено: 2026-09-14. Полная мотивация и протокол находятся в
[`docs/research_plan.md`](docs/research_plan.md).

## Сейчас

- [x] Проверить вычислительную ВМ, GPU и диск.
- [x] Выделить reusable training code из исходного ноутбука.
- [x] Исправить число классов, загрузку данных и расчёт mIoU.
- [x] Скачать PlantSeg v2/v3 на ВМ и проверить split/metadata.
- [x] Проверить диапазон пиксельных меток PlantSeg v3 (`0..115`).
- [x] Выполнить smoke test на A100 и сохранить результат как experiment 001.
- [x] Воспроизвести полноценный frozen DINOv3 ViT-B/16 linear baseline.
- [x] Уточнить основной протокол: binary lesion/background.
- [x] Запустить полный binary ViT-B/16 linear baseline на 10 эпох.
- [x] Оценить лучший binary checkpoint на test и отрисовать ошибки.
- [x] Сравнить ViT-B/16 с ViT-L/16 при неизменном linear decoder.

## Затем: качество

- [x] Сравнить linear probe и convolutional head при одном протоколе.
- [x] Проверить multi-layer decoder (4 промежуточных слоя DINOv3).
- [x] Проверить validation-only threshold calibration.
- [x] Сравнить stretch, aspect-preserving pad и умеренные аугментации.
- [x] Завершить 30-эпоховый multi-layer run с cosine schedule (`val fg IoU 0.6569`).
- [ ] Повторить multi-layer + cosine на seeds 43 и 44.
- [ ] Запустить single-layer conv с тем же 30-эпоховым cosine schedule как
      обязательный контроль.
- [ ] Проверить lesion-aware crop sampling / foreground-aware loss для малых масок.
- [ ] Проверить частичный fine-tuning последних 1–2 DINOv3 blocks.
- [ ] Выполнить минимум 3 запуска лучших конфигураций с разными seeds.
- [x] Добавить foreground IoU, Dice, precision/recall и pixel AP.
- [x] Добавить выбор и отрисовку лучших/типичных/худших binary predictions.
- [x] Зафиксировать точное название текущей метрики: foreground pixel AP, не
      COCO mask AP.
- [ ] Зафиксировать mean/std; ориентир 0.70 не использовать для выбора по test.

## Затем: эффективность разметки

- [x] Зафиксировать основную единицу бюджета: полностью размеченное изображение.
- [ ] Добавить фиксированный optimizer-step budget для сравнения подмножеств.
- [ ] Извлечь и закешировать DINOv3 image descriptors из patch embeddings.
- [ ] Random sampling baseline.
- [ ] Stratified random baseline по бесплатным image-level metadata.
- [ ] k-means representatives / k-medoids.
- [ ] Farthest-first / greedy k-center.
- [ ] Гибрид uncertainty + diversity после cold start.
- [ ] Кривые качества для 1/5/10/25/50/100% train, mean ± std.

## Отдельные ветки исследования

- [ ] Vanilla SAM: oracle point/box promptability benchmark.
- [ ] Автоматические DINO→SAM prompts без GT.
- [ ] SAM mask-decoder fine-tuning; adapter/LoRA только как второй этап.
- [x] Визуализация ошибок и анализ качества по размеру поражения/болезни.
- [x] Перестроить промежуточный отчёт вокруг общей исследовательской цели.
- [x] Добавить автономную XeLaTeX/biber сборку отчёта.

## Definition of done для эксперимента

Эксперимент считается зафиксированным, если сохранены commit hash, resolved
config, версии Python/PyTorch/CUDA, seed, train/validation loss, mIoU,
per-class IoU, лучший checkpoint и короткий вывод в `docs/experiments/`.

## Текущий исследовательский вывод

После обзора всех веток репозитория коллеги и validation-диагностики основной
приоритет — качество малых поражений. Нижний квартиль размера GT-маски имеет
macro foreground IoU `0.3247`, верхний — `0.7469`. Простое масштабирование
backbone, padding и threshold tuning эту проблему не адресуют. Следующая серия
после cosine-контроля должна проверять sampling/loss и spatial decoder, сохраняя
test закрытым до выбора финальной конфигурации.
