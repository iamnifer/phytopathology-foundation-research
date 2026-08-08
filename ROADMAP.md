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
- [ ] Оценить лучший binary checkpoint на test и отрисовать ошибки.
- [ ] Сравнить ViT-B/16 с ViT-L/16 при неизменном linear decoder.

## Затем: качество

- [ ] Сравнить linear probe и convolutional head при одном протоколе.
- [ ] Добавить multi-layer decoder (4 промежуточных слоя DINOv3).
- [ ] Настроить AdamW, learning-rate schedule, class weights и crop sampling.
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
- [ ] Визуализация ошибок и анализ редких классов.
- [ ] Синхронизировать таблицы экспериментов с текстом курсовой.

## Definition of done для эксперимента

Эксперимент считается зафиксированным, если сохранены commit hash, resolved
config, версии Python/PyTorch/CUDA, seed, train/validation loss, mIoU,
per-class IoU, лучший checkpoint и короткий вывод в `docs/experiments/`.
