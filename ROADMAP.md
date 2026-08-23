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
- [x] Повторить multi-layer + cosine на seeds 43 и 44.
- [x] Запустить single-layer conv с тем же 30-эпоховым cosine schedule как
      обязательный контроль.
- [x] Проверить image-level lesion-aware oversampling; результат отрицательный.
- [ ] Проверить foreground-centred crop / foreground-aware loss только после
      основной серии эффективности данных.
- [ ] Проверить частичный fine-tuning последних 1–2 DINOv3 blocks.
- [x] Выполнить минимум 3 запуска DINO decoder-конфигураций с разными seeds.
- [x] Добавить foreground IoU, Dice, precision/recall и pixel AP.
- [x] Добавить выбор и отрисовку лучших/типичных/худших binary predictions.
- [x] Зафиксировать точное название текущей метрики: foreground pixel AP, не
      COCO mask AP.
- [x] Зафиксировать mean/std; ориентир 0.70 не использовать для выбора по test.

## Затем: эффективность разметки

- [x] Зафиксировать основную единицу бюджета: полностью размеченное изображение.
- [x] Добавить фиксированный optimizer-step budget для сравнения подмножеств.
- [x] Извлечь и закешировать DINOv3 image descriptors из patch embeddings.
- [x] Подготовить nested random sampling subsets.
- [x] Подготовить nested stratified random subsets по image-level metadata.
- [ ] k-means representatives / k-medoids.
- [x] Подготовить nested farthest-first / greedy k-center subsets.
- [ ] Гибрид uncertainty + diversity после cold start.
- [ ] Кривые качества для 1/5/10/25/50/100% train, mean ± std.

## Отдельные ветки исследования

- [x] Vanilla SAM: oracle point/box promptability benchmark.
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

Multi-layer decoder имеет небольшой, но нестабильный средний выигрыш над
single-layer (`+0.0075 ± 0.0069` в парном сравнении). Image-level oversampling
малых масок не помог даже Q1, а oracle box для frozen SAM дал только `0.4659`
disease IoU. Основной приоритет теперь — запустить сопоставимые кривые качества
random/stratified/farthest-first при фиксированном числе шагов, сохраняя test
закрытым до выбора финальной конфигурации.
