# Вычислительное окружение

Проверено 2026-09-08 на `compute-iamnifer` (Ubuntu 24.04, kernel
6.8.0-139-generic).

| Ресурс | Значение |
|---|---|
| GPU | NVIDIA A100-SXM4-80GB, 81920 MiB |
| Driver | 595.71.05 server-open |
| Driver CUDA capability | 13.2 |
| CPU | 28 logical cores |
| RAM | 116 GiB |
| Root filesystem | 194 GiB total |
| Свободно после setup/data | 171 GiB |

На чистой ВМ GPU присутствовала в PCI, но драйвера и `nvidia-smi` не было.
Были установлены рекомендованные Ubuntu compute packages:

```bash
sudo apt-get update
sudo apt-get install -y ubuntu-drivers-common
sudo ubuntu-drivers install --gpgpu
sudo apt-get install -y nvidia-utils-595-server python3.12-venv unzip aria2
sudo modprobe nvidia
```

Python-окружение расположено в `.venv`. Проверка PyTorch:

```text
torch 2.14.0+cu130
CUDA runtime 13.0
NVIDIA A100-SXM4-80GB
```

## Локальные данные и веса на ВМ

```text
data/
├── raw/
│   ├── plantseg_v2.zip  # MD5 c321381894575e5dca83686d125fe2cd
│   └── plantseg_v3.zip  # MD5 9458f4fb61d026df1580ce437df0b63a
├── plantseg_v2/         # legacy: train/test, старая metadata с дефектом
├── plantseg_v3/         # основной: train/val/test и Metadatav2.csv
└── plantwild_hf/        # metadata и healthy test для внешней проверки
models/
├── dinov3-vitb16-pretrain-lvd1689m/
├── dinov3-vitl16-pretrain-lvd1689m/
└── sam-vit-base/
```

PlantSeg v3 содержит 7916/1247/2295 изображений в train/val/test. В каждой
проверенной PNG-маске встречаются фон `0` и один disease class; по датасету
используются IDs `0..115`.

Из PlantWild v1 подготовлен manifest 7840 здоровых изображений 30 видов
растений: 5510/774/1556 в train/val/test. На ВМ загружены только 1556 test-
изображений (около 383 MB), используемых для внешней оценки ложных тревог.
Manifest и загрузка воспроизводятся командой
`python -m phytopathology.prepare_plantwild_healthy`.

Веса Hugging Face gated и без пользовательского токена возвращают HTTP 401.
На ВМ использована публичная копия ModelScope:

```bash
modelscope download --model facebook/dinov3-vitb16-pretrain-lvd1689m \
  --local_dir models/dinov3-vitb16-pretrain-lvd1689m
```

Старые установщики и черновые рендеры вынесены из репозитория ВМ в
`/home/iamnifer/phytopathology-workspace-archive-20260916`. Карантин ошибочной
синхронизации сохранён отдельно в
`/home/iamnifer/sync-mistake-quarantine-20260916`; оба каталога можно удалить
после сдачи, если артефакты точно не нужны.

Завершённые ключевые запуски копируются с ВМ в локальный игнорируемый Git
каталог `runs/`, включая лучшую контрольную точку, фактическую конфигурацию,
окружение и полную историю метрик. Числа и выводы, необходимые без доступа к
тяжёлым файлам, дополнительно фиксируются в `docs/experiments/`.

## Сборка отчёта

С 2026-09-14 на ВМ установлены `latexmk`, XeLaTeX, biber, кириллические и
дополнительные LaTeX-пакеты, а также Liberation fonts. Отчёт собирается так:

```bash
cd /home/iamnifer/phytopathology-foundation-research/report
latexmk coursework.tex
```

Результат создаётся в `report/build/coursework.pdf`; каталог `build/`
игнорируется Git.
