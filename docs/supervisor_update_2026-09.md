# Предварительное обновление для руководителя

## Коротко

Основную постановку зафиксировал как binary segmentation поражение/фон на
PlantSeg. Старый ноутбук перенесён в воспроизводимый training pipeline; метрики
считаются по общей confusion matrix, каждый run сохраняет config, environment,
seed, историю и checkpoint. Главная метрика — disease/foreground IoU, потому
что mean IoU заметно искажается доминирующим фоном.

Лучший текущий результат на validation: **foreground IoU 0.6569**, mean IoU
0.7765, Dice 0.7929. Это frozen DINOv3 ViT-B/16, convolutional decoder по
промежуточным слоям 3/6/9/12, 30 эпох, AdamW и cosine LR decay. По seed
42/43/44 получено **0.6519 ± 0.0062** foreground IoU. Test для нового кандидата
не использовался.

## Что проверено

| Эксперимент | Validation foreground IoU |
|---|---:|
| ViT-B frozen + linear | 0.5944 |
| ViT-L frozen + linear | 0.6040 |
| ViT-B frozen + conv head | 0.6345 |
| + aspect-preserving padding | 0.6231 |
| + padding и moderate spatial/color aug | 0.6248 |
| 4 DINO layers + conv, 10 epochs | 0.6357 |
| **4 DINO layers + conv, 30 epochs + cosine** | **0.6519 ± 0.0062** |
| Last DINO layer + conv, 30 epochs + cosine | 0.6444 ± 0.0007 |
| + small-lesion image oversampling (seed 42) | 0.6498 |

Для первых трёх заранее зафиксированных baseline есть test: ViT-B linear
0.6036, ViT-L linear 0.6112, ViT-B conv 0.6488 foreground IoU. Последующие
настройки выбирались только по validation, чтобы не продолжать адаптацию к test.

Подбор probability threshold почти не помог: 0.48 даёт 0.6571 против 0.6569
при стандартном 0.50. Значит улучшение связано с обучением spatial readout, а
не с постобработкой.

## Сопоставление с работой коллеги

Изучил все ветки `dfbakin/plant-diseases-segmentation`, включая architecture,
augmentation и scheduler ablations, CAM/WeakCLIP/SAM/SPDNet и поздний общий
research report. Его fully supervised ориентиры по disease IoU: около 0.646
для DeepLabV3+/U-Net, 0.682 для SegFormer и 0.701 для исправленного pretrained
SegNeXt. Наш текущий DINO decoder уже выше первых двух простых baseline и
находится в реалистичном диапазоне до 0.70. WSSS/CAM числа коллеги напрямую с
нашими не сравниваю, поскольку там pixel masks не используются для основного
обучения.

Ключевой переносимый вывод коллеги подтвердился: качество readout из
промежуточных features важнее classification quality и простого увеличения
backbone. Его preprocessing/augmentation recipe у нас, напротив, не перенёсся:
padding увеличил mean IoU за счёт лёгкого фона, но снизил disease IoU.

## Что показывают ошибки

Для прежнего conv baseline macro per-image IoU был 0.325 на четверти самых
маленьких масок и 0.747 на четверти самых больших. Новый multi-layer cosine
decoder поднял эти значения до 0.383 и 0.769 соответственно. Общий macro IoU
вырос 0.535→0.577, доля полностью пропущенных поражений снизилась 2.25%→0.80%,
а undersegmentation — 31.3%→20.4%. Цена — рост oversegmentation 23.4%→26.9%.

Таким образом, основной оставшийся failure mode — малые поражения, а не общая
нехватка ёмкости frozen backbone.

Простое удвоение частоты train-изображений из нижнего квартиля не помогло: Q1
IoU снизился 0.383→0.365, а общий disease IoU 0.6569→0.6498. Повторять этот
вариант нецелесообразно.

## Vanilla SAM

На всех 1247 validation-изображениях frozen SAM ViT-B с одной oracle-точкой
дал disease IoU 0.3624, а с точным GT box — 0.4659. Даже box имеет высокий
recall 0.8835, но precision лишь 0.4964. Это показывает, что умение SAM
очертить подсказанный объект не заменяет адаптацию к семантике поражения.

## Предлагаемый следующий шаг

1. Построить label-efficiency curves: random, stratified random и
   representation-based selection по frozen DINOv3 embeddings.
2. Проверить автоматические DINO→SAM prompts отдельно от oracle режима.
3. Обучить SAM mask decoder в одном зафиксированном режиме.
4. Рассматривать partial DINO fine-tuning только после основных кривых данных.
5. Один раз оценить финальных победителей на test и обновить текст курсовой.

После стабилизации supervised baseline можно переходить к второй линии —
минимизации разметки: random subset, k-means/k-medoids по embeddings и farthest
point sampling с кривыми качества от объёма данных. SAM разумно держать отдельным
baseline с фиксированным prompt protocol; dense mask prompting в экспериментах
коллеги работал плохо.
