# План работы

Обновлено: 2026-09-08.

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
- [ ] Проверить lesion-aware crop sampling / foreground-aware loss для малых масок.
- [ ] Настроить AdamW и class weights.
- [ ] Проверить частичный fine-tuning последних блоков и LoRA/adapters.
- [ ] Выполнить минимум 3 запуска лучших конфигураций с разными seeds.
- [x] Добавить foreground IoU, Dice, precision/recall и pixel AP.
- [x] Добавить выбор и отрисовку лучших/типичных/худших binary predictions.
- [ ] Уточнить у руководителя, означает ли `mAP` pixel AP или COCO mask AP.
- [ ] Зафиксировать mean/std; целевая binary-метрика — около 0.70.

## Затем: few-shot

- [ ] Зафиксировать протокол: число изображений на класс и seeds выборки.
- [ ] Random sampling baseline.
- [ ] k-means / k-medoids по patch embeddings.
- [ ] Farthest Point Sampling.
- [ ] Кривые качество–число размеченных изображений.

## Отдельные ветки исследования

- [ ] Сравнение с SAM/SAM3 при одинаковом split и метрике.
- [x] Визуализация ошибок и анализ качества по размеру поражения/болезни.
- [ ] Синхронизировать таблицы экспериментов с текстом курсовой.

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
