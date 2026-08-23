# Исследовательская программа

Обновлено 2026-09-14 после анализа предварительных экспериментов, требований к
курсовому проекту и отчётов по смежной работе.

## Цель

Исследовать компромисс между качеством и объёмом пиксельной разметки при
адаптации visual foundation models DINOv3 и SAM к бинарной сегментации
поражённых областей растений на PlantSeg.

Итог работы должен отвечать не на вопрос «какой запуск дал самое большое
число», а на три проверяемых исследовательских вопроса:

1. Какие части адаптации DINOv3 дают устойчивый прирост и как меняют качество
   малых поражений?
2. Что именно умеет vanilla SAM при честно заданных prompts и помогает ли его
   адаптация либо использование после DINOv3?
3. Какие изображения следует размечать первыми и сколько размеченных
   изображений нужно, чтобы приблизиться к full-data качеству?

## Единый протокол

- Задача: disease/lesion (`all source labels > 0`) против background (`0`),
  `ignore_index=255` не участвует в метриках.
- Единица бюджета разметки: полностью размеченное изображение. Region-level
  budget остаётся возможным расширением, но не смешивается с основной кривой.
- Главная метрика: dataset-level foreground/disease IoU на validation.
- Дополнительные метрики: background/foreground mean IoU, Dice, precision,
  recall, foreground pixel AP, macro per-image IoU и IoU по квартилям площади
  поражения.
- Checkpoint, threshold и гиперпараметры выбираются только по validation. Test
  применяется один раз к зафиксированным финальным моделям.
- Ключевые результаты: минимум три seed, mean ± std. Отрицательные результаты
  сохраняются наравне с положительными.
- Для data-budget экспериментов фиксируются число optimizer steps и правило
  выбора checkpoint: малые подмножества не должны получать меньше обновлений
  только из-за меньшего числа изображений.

## Линия A — качество адаптации

### A1. Завершить причинную абляцию DINOv3

Текущий лучший результат смешивает два изменения: multi-layer features и
30-эпоховый cosine schedule. Поэтому обязательны:

1. ~~повторы multi-layer + cosine для seed 43 и 44~~ — выполнено, итог по трём
   seed: $0.6519\pm0.0062$ disease IoU;
2. ~~single-layer convolutional decoder + тот же 30-эпоховый cosine schedule~~
   — выполнено: $0.6444\pm0.0007$ против $0.6519\pm0.0062$ у multi-layer;
3. ~~один lesion-aware метод~~ — image-level oversampling дал отрицательный
   результат; не повторять, возможный follow-up только foreground crop/loss;
4. частичное размораживание последних 1–2 ViT blocks с отдельным малым LR;
5. перенос только финального рецепта на ViT-L/16.

Нельзя интерпретировать ViT-B против ViT-L как полное сравнение размеров: пока
они сопоставлены только с linear head.

### A2. SAM

SAM является promptable, а не disease-aware моделью. Сравниваются три разных
режима, которые должны находиться в разных строках таблицы:

1. **Oracle promptability:** positive point и bounding box из GT выполнены;
   получено 0.3624 и 0.4659 disease IoU. Это верхняя диагностическая оценка, а
   не автоматическая сегментация.
2. **Automatic refinement:** box/points извлекаются из coarse DINOv3 mask без
   использования validation/test GT.
3. **Adaptation:** обучение mask decoder; затем, только если оправдано,
   LoRA/adapter для image encoder. Объём train-разметки совпадает с DINOv3.

Automatic Mask Generator без семантического правила выбора маски не считается
решением disease segmentation.

## Линия B — эффективность разметки

Frozen DINOv3 вычисляет patch embeddings всего train pool. Для основной серии
они агрегируются в L2-нормированный image descriptor; выбранные индексы
сохраняются до обучения.

### Cold-start методы

1. uniform random — обязательный baseline;
2. stratified random по бесплатным image-level metadata — отдельный сильный
   baseline, не смешиваемый с полностью unsupervised выбором;
3. k-means representatives либо k-medoids — типичные представители;
4. farthest-first / greedy k-center — разнообразие и покрытие feature space.

### Последовательный метод

После небольшого cold-start набора обучается probe. Для оставшегося pool
считается pixel uncertainty, после чего выбираются неопределённые, но не
дублирующие друг друга изображения. Это гибрид uncertainty и diversity и
единственный кандидат на собственную модификацию метода в основной работе.

### Бюджеты и итоговые величины

Основная сетка: 1, 5, 10, 25, 50 и 100% train. Если 1% оказывается слишком
нестабильным, он остаётся диагностической точкой, а статистические выводы
начинаются с 5%.

Для каждого метода строятся:

- validation disease IoU против числа размеченных изображений;
- normalized area under learning curve;
- минимальная доля данных, достигающая 95% full-data disease IoU;
- стоимость самого отбора и распределение выбранных примеров по заболеванию и
  площади поражения (последнее используется только для анализа, не для
  unsupervised выбора).

## Что должно оказаться в финале

Минимально достаточный содержательный финал:

1. статистически подтверждённый DINOv3 baseline и контролируемая абляция
   decoder/schedule/adaptation;
2. SAM promptability upper bound, автоматический DINO→SAM pipeline и хотя бы
   один корректный режим fine-tuning;
3. label-efficiency curves random против representation-based selection;
4. анализ малых поражений и типичных FP/FN;
5. выводы в форме рекомендаций, а не только таблица рекордов.

Успех не определяется заранее числом 0.70. Весомым является и отрицательный
результат, если протокол показывает, например, что SAM не исправляет coarse
masks или DINOv3 clustering не превосходит random. Числовые ориентиры нужны
как sanity check, но не как условие подгонки экспериментов.

## Порядок выполнения

1. ~~Seeds 43/44 и single-layer cosine control.~~
2. ~~Один lesion-aware DINOv3 experiment.~~
3. ~~Vanilla SAM oracle prompt benchmark~~; далее DINO-generated prompts.
4. Извлечение DINOv3 descriptors и cold-start selection curves.
5. Один SAM mask-decoder fine-tuning режим.
6. Uncertainty + diversity только после устойчивого cold-start baseline.
7. Один финальный test для выбранных DINOv3 и SAM конфигураций.
