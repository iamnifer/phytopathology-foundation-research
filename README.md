# Перенос CNN и трансформеров на сегментацию заболеваний растений

Код, отчёт и журнал экспериментов курсовой работы о переносе DeepLabV3,
DINOv3 и SAM на семантическую сегментацию заболеваний растений при ограниченном
бюджете пиксельной разметки.

Исследование объединяет две линии: повышение качества адаптации foundation-
моделей и выбор наиболее полезных изображений для разметки. Формальная цель,
исследовательские вопросы и критерии завершения находятся в
[docs/research_plan.md](docs/research_plan.md).

Последнее [ревью руководителя](docs/reviews/supervisor_review_v1.md) и
[матрица внесённых правок](docs/review_response_v1.md) сохранены рядом с
исследовательским планом.

## Текущее состояние

- В исходном ноутбуке проведён один exploratory-запуск; записанное значение
  `val mIoU = 0.4575` считается **непроверенным**: mIoU усреднялся
  по отсутствующим классам, а число выходов было 115 вместо 116.
- Код ноутбука перенесён в пакет с CLI, конфигурацией, потоковым Dataset,
  корректным накоплением confusion matrix и артефактами каждого запуска.
- Основной протокол уточнён как binary `поражение / фон`. Frozen ViT-B/16
  linear probe после 10 эпох дал `val mean IoU = 0.7391` и
  `val foreground IoU = 0.5944`.
- Контролируемая замена backbone на frozen ViT-L/16 дала на test
  `foreground IoU = 0.6112` против `0.6036` у ViT-B/16.
- Простая convolutional head на ViT-B/16 оказалась заметно сильнее:
  `test foreground IoU = 0.6488`, Dice `0.7870`, pixel AP `0.8763`.
- Validation-only абляции показали, что aspect-preserving padding и умеренные
  spatial/color аугментации не переносятся автоматически: лучший foreground
  IoU снизился с `0.6345` до `0.6248`.
- Четыре промежуточных слоя сами по себе дали лишь `0.6357`, но 30 эпох с
  cosine decay подняли validation foreground IoU до **`0.6569`**. Результат
  подтверждён на трёх seeds: validation disease IoU $0.6519\pm0.0062$;
  новый test не запускался.
- Главный обнаруженный failure mode — малые поражения: macro IoU меняется от
  `0.3247` в нижнем квартиле размера маски до `0.7469` в верхнем.
- Повторы эффективности разметки на 10% и 25% завершены. Stratified отличается
  от random лишь на +0.0015/+0.0025 IoU, а farthest-first хуже на
  -0.0159/-0.0097. Анализ состава показывает, что farthest-first выбирает
  выбросы и сильнее искажает распределения растений и болезней.
- Три запуска полностью дообучаемого DeepLabV3-ResNet50 с ImageNet-инициализацией
  дали validation disease IoU `0.6690 ± 0.0008` против `0.6519 ± 0.0062` у
  DINOv3 с замороженным энкодером. Согласованный seed-42 контроль со случайной
  инициализацией дал `0.5442` против `0.6681` с ImageNet-весами.
- DINOv3 при 512 px дал `0.6644` на масштабированной сетке против `0.6569` при
  384 px; 768 px выполняется, а итог будет сделан по общей исходной сетке.
  Затем автоматически
  запускаются парный bootstrap, оценка на исходной сетке аннотаций, атлас 64
  ошибок и проверка
  ложных тревог на 1556 здоровых изображениях PlantWild v1. Последней идёт
  ограниченная серия сферического k-means при 10/25% разметки.
- Вычислительная ВМ подготовлена: A100 80 GB, 116 GB RAM; веса ViT-B/16 и
  ViT-L/16 получены через ModelScope.
- Финальный отчёт перестроен вокруг общей исследовательской задачи и
  автономно собирается через XeLaTeX/biber из каталога `report/`.

Актуальные задачи и критерии готовности находятся в [ROADMAP.md](ROADMAP.md),
а исходный эксперимент описан в
[docs/experiments/000_legacy_notebook.md](docs/experiments/000_legacy_notebook.md).
Параметры подготовленной ВМ и расположение данных записаны в
[docs/infrastructure.md](docs/infrastructure.md).
Два направления работы и порядок экспериментов собраны в
[docs/research_plan.md](docs/research_plan.md).
Результаты масштабирования backbone записаны в
[docs/experiments/005_dinov3_vitl16_binary_linear.md](docs/experiments/005_dinov3_vitl16_binary_linear.md).
Сравнение decoder heads записано в
[docs/experiments/006_dinov3_vitb16_binary_conv.md](docs/experiments/006_dinov3_vitb16_binary_conv.md).
Эксперименты коллеги и их применимость разобраны в
[docs/colleague_experiments_review.md](docs/colleague_experiments_review.md).
Последние validation-абляции находятся в
[docs/experiments/007_binary_threshold_calibration.md](docs/experiments/007_binary_threshold_calibration.md),
[docs/experiments/008_preprocessing_augmentation_ablation.md](docs/experiments/008_preprocessing_augmentation_ablation.md),
[docs/experiments/009_dinov3_multilayer_conv.md](docs/experiments/009_dinov3_multilayer_conv.md)
и [docs/experiments/010_validation_error_analysis.md](docs/experiments/010_validation_error_analysis.md).
Лучший новый validation run описан в
[docs/experiments/011_multilayer_cosine.md](docs/experiments/011_multilayer_cosine.md),
а историческая сводка для обсуждения до ревью v1 — в
[docs/supervisor_update_2026-09.md](docs/supervisor_update_2026-09.md).
Первая кривая эффективности данных зафиксирована в
[docs/experiments/015_label_efficiency_seed42.md](docs/experiments/015_label_efficiency_seed42.md).
Повторы DeepLabV3 и их вычислительная стоимость записаны в
[docs/experiments/018_deeplabv3_baseline.md](docs/experiments/018_deeplabv3_baseline.md).
Контроль разрешения DINOv3 фиксируется в
[docs/experiments/019_resolution_control.md](docs/experiments/019_resolution_control.md).

## Быстрый старт

Требуется Python 3.10+ и установленный PyTorch с поддержкой CUDA.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,download]'
mkdir -p models
modelscope download --model facebook/dinov3-vitb16-pretrain-lvd1689m \
  --local_dir models/dinov3-vitb16-pretrain-lvd1689m
python -m phytopathology.train --config configs/dinov3_vitb16_linear.yaml \
  --data-root /path/to/plantsegv3 --smoke-test
pytest
```

Полный запуск отличается только отсутствием `--smoke-test`:

```bash
python -m phytopathology.train --config configs/dinov3_vitb16_linear.yaml \
  --data-root /path/to/plantsegv3
```

Начальное значение генератора из YAML-конфигурации можно переопределить без её
копирования:

```bash
python -m phytopathology.train \
  --config configs/dinov3_vitb16_binary_multilayer_conv_cosine.yaml --seed 43
```

Для экспериментов с бюджетом разметки сначала кэшируются frozen-признаки и
создаются точные manifests подмножеств:

```bash
python -m phytopathology.extract_descriptors \
  --config configs/dinov3_vitb16_binary_multilayer_conv_cosine.yaml \
  --output features/train.npz
python -m phytopathology.select_subsets --features features/train.npz \
  --metadata data/plantseg_v3/Metadatav2.csv --output-dir subsets/seed42
python -m phytopathology.train \
  --config configs/dinov3_vitb16_binary_multilayer_conv_cosine.yaml \
  --subset-file subsets/seed42/farthest_seed42_0.1.json --epoch-samples 7916
```

`--epoch-samples` сохраняет одинаковое число optimizer steps на эпоху для
полной выборки и её подмножеств.

Внешняя проверка ложных тревог использует только официальную test-часть
здоровых классов PlantWild v1. Manifest и нужные изображения воспроизводятся
отдельно от PlantSeg:

```bash
python -m phytopathology.prepare_plantwild_healthy \
  --dataset-root data/plantwild_hf --download-split test
```

Каждый запуск создаёт отдельный каталог в `runs/`: resolved-конфиг,
информацию об окружении, метрики JSONL и лучший checkpoint. `runs/` не хранится
в Git; итоговые числа и выводы переносятся в `docs/experiments/`.

Ожидаемая структура PlantSeg:

```text
plantsegv3/
├── Metadatav2.csv
├── images/{train,val,test}/...
└── annotations/{train,val,test}/...
```

Имена файлов берутся из колонок `Name` и `Label file`, а разбиение — из
колонки `Split`. Для совместимости поддерживаются как `val`, так и
`validation` в именах директорий.

## Структура репозитория

```text
configs/                 параметры воспроизводимых запусков
docs/experiments/        зафиксированные результаты и выводы
notebooks/               только exploration и визуальный анализ
report/                  LaTeX-исходники курсовой работы
src/phytopathology/      dataset, модель, метрики и training CLI
tests/                   быстрые тесты без скачивания весов и датасета
ROADMAP.md               текущая очередь работ
```

Исходный ноутбук сохранён как
`notebooks/archive/legacy_dinov3_experiment.ipynb` для происхождения первого
эксперимента. Он не используется как основной код обучения.

## Основные соглашения

- PlantSeg — 116 классов (`0..115`), фон включён; `ignore_index=255`.
- Постановки разделены явно: основной multiclass-конфиг сохраняет `0..115`,
  а `configs/dinov3_vitb16_binary_linear.yaml` отображает все болезни в `1`.
- В binary-протоколе главная метрика поражения — foreground IoU; отдельно
  сохраняются mean IoU двух классов, Dice, precision, recall и pixel AP.
- Изображения и маски читаются по требованию, а не целиком в RAM.
- Размер входа должен делиться на patch size DINOv3 (16).
- Случайность фиксируется seed; mixed precision включается только на CUDA.

## Источники

- [DINOv3](https://github.com/facebookresearch/dinov3)
- [PlantSeg](https://github.com/tqwei05/PlantSeg)
