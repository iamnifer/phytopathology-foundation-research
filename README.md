# Сегментация заболеваний растений с DINOv3

Код и журнал экспериментов курсовой работы по семантической сегментации
заболеваний растений на PlantSeg с визуальным энкодером DINOv3.

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
- Вычислительная ВМ подготовлена: A100 80 GB, 116 GB RAM; веса ViT-B/16 и
  ViT-L/16 получены через ModelScope.

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
src/phytopathology/      dataset, модель, метрики и training CLI
tests/                   быстрые тесты без скачивания весов и датасета
ROADMAP.md               текущая очередь работ
```

Файл `notebook8f60e8b9e5.ipynb` временно оставлен в корне как неизменённый
источник первого эксперимента. После воспроизведения CLI-запуска его следует
перенести в `notebooks/archive/`, не использовать как основной training-код.

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
