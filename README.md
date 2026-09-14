# Перенос CNN и трансформеров на сегментацию заболеваний растений

Код, отчёт и журнал экспериментов курсовой работы о переносе DeepLabV3,
DINOv3 и SAM на семантическую сегментацию заболеваний растений при ограниченном
бюджете пиксельной разметки.

Собранная версия курсовой доступна в
[report/coursework.pdf](report/coursework.pdf); исходники и инструкция по
сборке находятся в том же каталоге.

Исследование объединяет две линии: повышение качества адаптации foundation-
моделей и выбор наиболее полезных изображений для разметки. Формальная цель,
исследовательские вопросы и критерии завершения находятся в
[docs/research_plan.md](docs/research_plan.md).

Последнее [ревью руководителя](docs/reviews/supervisor_review_v1.md) и
[матрица внесённых правок](docs/review_response_v1.md) сохранены рядом с
исследовательским планом. Итоговая проверка каждого замечания, чисел и PDF
зафиксирована в [финальном аудите](docs/final_audit_2026-09-16.md).

## Текущее состояние

- В исходном ноутбуке проведён один разведочный запуск; записанное значение
  `val mIoU = 0.4575` считается **непроверенным**: mIoU усреднялся
  по отсутствующим классам, а число выходов было 115 вместо 116.
- Код ноутбука перенесён в пакет с интерфейсом командной строки,
  конфигурациями, потоковым набором данных, корректным накоплением матрицы
  ошибок и артефактами каждого запуска.
- Основной протокол уточнён как бинарный `поражение / фон`. Замороженный
  ViT-B/16 с линейным декодером после 10 эпох дал валидационное среднее IoU
  `0.7391` и IoU поражения `0.5944`.
- Контролируемая замена энкодера на замороженный ViT-L/16 дала на тесте IoU
  поражения `0.6112` против `0.6036` у ViT-B/16.
- Небольшой свёрточный декодер на ViT-B/16 оказался заметно сильнее: на тесте
  IoU поражения `0.6488`, Dice `0.7870`, пиксельная AP `0.8763`.
- Абляции только по валидации показали, что сохранение пропорций с дополнением
  и умеренные пространственные/цветовые аугментации не переносятся
  автоматически: лучший IoU поражения снизился с `0.6345` до `0.6248`.
- Четыре промежуточных слоя сами по себе дали лишь `0.6357`, но 30 эпох с
  косинусным затуханием подняли валидационное IoU поражения до **`0.6569`**.
  По трём начальным значениям получено $0.6519\pm0.0062$; новый тест не
  запускался.
- Главный обнаруженный режим ошибки — малые поражения: среднее IoU по
  изображениям меняется от
  `0.3247` в нижнем квартиле размера маски до `0.7469` в верхнем.
- Повторы эффективности разметки на 10% и 25% завершены. Стратифицированный
  отбор отличается от случайного лишь на +0.0015/+0.0025 IoU, а жадный отбор
  наиболее удалённых объектов хуже на -0.0159/-0.0097. Анализ состава
  показывает, что последний выбирает выбросы и сильнее искажает распределения
  растений и болезней.
- Сферический k-means с представителями кластеров дал `0.6051 ± 0.0064` при
  10% и `0.6236 ± 0.0037` при 25% разметки. Он лучше жадного отбора крайних
  точек на +0.0194/+0.0115, но практически совпадает со случайным и
  стратифицированным отбором; устойчивого выигрыша над простыми базами нет.
- Три запуска полностью дообучаемого DeepLabV3-ResNet50 с ImageNet-инициализацией
  дали валидационное IoU поражения `0.6690 ± 0.0008` против `0.6519 ± 0.0062`
  у DINOv3 с замороженным энкодером. Согласованный контроль со случайной
  инициализацией дал `0.5442` против `0.6681` с ImageNet-весами.
- В дополнительном честном линейном зондировании оба энкодера заморожены и
  используют одинаковую голову и сетку признаков: DINOv3 получил
  `0.5990 ± 0.0011`, ResNet-50 — `0.3860 ± 0.0044`, парная разность
  `+0.2129 ± 0.0034`. Все три bootstrap-интервала исключают ноль. Контроль
  ResNet на исходной сетке 48×48 дал лишь `0.3703 ± 0.0041`, поэтому разрыв не
  объясняется приведением признаков к 24×24.
- DINOv3 при 384/512/768 px дал `0.6569`/`0.6644`/`0.6671` на
  соответствующей масштабированной сетке. Итог будет сделан по общей
  исходной сетке: `0.6194`/`0.6276`/`0.6323`; интервал разности 768
  против 384 px включает ноль. Парный бутстреп, атлас и ручная таксономия
  64 ошибок, панель DINO/SAM, оценка скорости и проверка на 1556 здоровых
  изображениях PlantWild v1 завершены. Ограниченная серия сферического
  k-means при 10/25% разметки также завершена.
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
Результаты увеличения энкодера записаны в
[docs/experiments/005_dinov3_vitl16_binary_linear.md](docs/experiments/005_dinov3_vitl16_binary_linear.md).
Сравнение декодеров записано в
[docs/experiments/006_dinov3_vitb16_binary_conv.md](docs/experiments/006_dinov3_vitb16_binary_conv.md).
Эксперименты коллеги и их применимость разобраны в
[docs/colleague_experiments_review.md](docs/colleague_experiments_review.md).
Последние валидационные абляции находятся в
[docs/experiments/007_binary_threshold_calibration.md](docs/experiments/007_binary_threshold_calibration.md),
[docs/experiments/008_preprocessing_augmentation_ablation.md](docs/experiments/008_preprocessing_augmentation_ablation.md),
[docs/experiments/009_dinov3_multilayer_conv.md](docs/experiments/009_dinov3_multilayer_conv.md)
и [docs/experiments/010_validation_error_analysis.md](docs/experiments/010_validation_error_analysis.md).
Лучший новый валидационный запуск описан в
[docs/experiments/011_multilayer_cosine.md](docs/experiments/011_multilayer_cosine.md),
а историческая сводка для обсуждения до ревью v1 — в
[docs/supervisor_update_2026-09.md](docs/supervisor_update_2026-09.md).
Первая кривая эффективности данных зафиксирована в
[docs/experiments/015_label_efficiency_seed42.md](docs/experiments/015_label_efficiency_seed42.md).
Повторы DeepLabV3 и их вычислительная стоимость записаны в
[docs/experiments/018_deeplabv3_baseline.md](docs/experiments/018_deeplabv3_baseline.md).
Контроль разрешения DINOv3 фиксируется в
[docs/experiments/019_resolution_control.md](docs/experiments/019_resolution_control.md).
Итоговое сравнение представителей кластеров записано в
[docs/experiments/021_kmeans_selection.md](docs/experiments/021_kmeans_selection.md).
Согласованное линейное зондирование CNN и трансформера записано в
[docs/experiments/022_frozen_linear_probe_comparison.md](docs/experiments/022_frozen_linear_probe_comparison.md).

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

Для экспериментов с бюджетом разметки сначала кэшируются замороженные признаки
и создаются точные манифесты подмножеств:

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

`--epoch-samples` сохраняет одинаковое число шагов оптимизатора на эпоху для
полной выборки и её подмножеств.

Внешняя проверка ложных тревог использует только официальную тестовую часть
здоровых классов PlantWild v1. Манифест и нужные изображения воспроизводятся
отдельно от PlantSeg:

```bash
python -m phytopathology.prepare_plantwild_healthy \
  --dataset-root data/plantwild_hf --download-split test
```

Каждый запуск создаёт отдельный каталог в `runs/`: фактическую конфигурацию,
информацию об окружении, метрики JSONL и лучшую контрольную точку. `runs/` не хранится
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
notebooks/               только разведочные запуски и визуальный анализ
report/                  LaTeX-исходники курсовой работы
src/phytopathology/      набор данных, модель, метрики и обучение
tests/                   быстрые тесты без скачивания весов и датасета
ROADMAP.md               текущая очередь работ
```

Исходный ноутбук сохранён как
`notebooks/archive/legacy_dinov3_experiment.ipynb` для происхождения первого
эксперимента. Он не используется как основной код обучения.

## Основные соглашения

- PlantSeg — 116 классов (`0..115`), фон включён; `ignore_index=255`.
- Постановки разделены явно: основной многоклассовый конфиг сохраняет `0..115`,
  а `configs/dinov3_vitb16_binary_linear.yaml` отображает все болезни в `1`.
- В бинарном протоколе главная метрика — IoU поражения; отдельно сохраняются
  среднее IoU двух классов, Dice, точность, полнота и пиксельная AP.
- Изображения и маски читаются по требованию, а не целиком в RAM.
- Размер входа должен делиться на размер фрагмента DINOv3 (16).
- Начальное значение генератора фиксируется; смешанная точность включается
  только на CUDA.

## Источники

- [DINOv3](https://github.com/facebookresearch/dinov3)
- [PlantSeg](https://github.com/tqwei05/PlantSeg)
