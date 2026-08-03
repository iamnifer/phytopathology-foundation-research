# Сегментация заболеваний растений с DINOv3

Код и журнал экспериментов курсовой работы по семантической сегментации
заболеваний растений на PlantSeg с визуальным энкодером DINOv3.

## Текущее состояние

- В исходном ноутбуке проведён один exploratory-запуск: frozen DINOv3 ViT-B/16,
  простая convolutional head, 1 эпоха, записано `val mIoU = 0.4575`.
- Старое значение пока считается **непроверенным**: в ноутбуке mIoU усреднялся
  по отсутствующим классам, а число выходов было 115 вместо 116.
- Код ноутбука перенесён в пакет с CLI, конфигурацией, потоковым Dataset,
  корректным накоплением confusion matrix и артефактами каждого запуска.
- Вычислительная ВМ подготовлена: A100 80 GB, 116 GB RAM, 190 GB свободного
  места на момент настройки.

Актуальные задачи и критерии готовности находятся в [ROADMAP.md](ROADMAP.md),
а исходный эксперимент описан в
[docs/experiments/000_legacy_notebook.md](docs/experiments/000_legacy_notebook.md).

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
- Основная метрика — dataset-level mIoU, рассчитанная из общей confusion
  matrix. Дополнительно сохраняются pixel accuracy и IoU каждого класса.
- Изображения и маски читаются по требованию, а не целиком в RAM.
- Размер входа должен делиться на patch size DINOv3 (16).
- Случайность фиксируется seed; mixed precision включается только на CUDA.

## Источники

- [DINOv3](https://github.com/facebookresearch/dinov3)
- [PlantSeg](https://github.com/tqwei05/PlantSeg)
