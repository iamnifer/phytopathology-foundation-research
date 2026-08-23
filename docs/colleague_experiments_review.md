# Разбор экспериментов коллеги

Источник: репозиторий
[`dfbakin/plant-diseases-segmentation`](https://github.com/dfbakin/plant-diseases-segmentation).
Разбор выполнен 23 августа 2026 года по всем удалённым веткам и по отчётам,
включая `RESEARCH_CONTEXT.md` из наиболее полной ветки
`wsss-weakclip-pipeline`.

## Что именно исследовал коллега

Основная линия коллеги — weakly supervised semantic segmentation (WSSS):
обучение локализации по image-level меткам с использованием PlantVillage и
оценка по пиксельной разметке PlantSeg. Полностью supervised модели на масках
PlantSeg использовались как верхняя граница. Наша текущая линия отличается:
DINOv3 frozen probe обучается непосредственно на масках PlantSeg, поэтому её
следует сравнивать с fully supervised baselines, а не с WSSS CAM-результатами.

Просмотренные ветки:

- `master` — подготовка данных и начальный benchmark;
- `arch-benchmark-exps` — сравнение supervised архитектур;
- `augmentation-ablation` — девять augmentation presets;
- `multiclass-segmentation` — 116-классовая постановка;
- `scheduler-test` — 12 вариантов learning-rate schedule;
- `disease-classifier` — классификаторы и CAM;
- `wsss-weakclip-pipeline` — PSA, WeakCLIP, SAM, SPDNet, auxiliary losses,
  segmentation probes и итоговый исследовательский контекст.

## Сопоставимые fully supervised результаты

Поздняя сводка коллеги использует dataset-level confusion matrix, вход 384 px,
30 эпох и AdamW. Это наиболее близкий внешний ориентир для нашего binary
протокола.

| Модель | mean IoU, % | disease IoU, % | background IoU, % |
|---|---:|---:|---:|
| DeepLabV3+ / ResNet-50 | 78.9 | 64.6 | 92.9 |
| U-Net / ResNet-34 | 78.7 | 64.6 | 92.8 |
| SegFormer / MiT-B3 | 80.9 | 68.2 | 93.6 |
| SegNeXt / MSCAN-T, исправленный pretrained run | **82.1** | **70.1** | **94.0** |
| Наш frozen DINOv3 ViT-B + conv head | 77.0 test | 64.9 test | — |

Следовательно, disease IoU около 0.70 достижим на этом датасете. Наш текущий
результат уже находится на уровне DeepLabV3+/U-Net, но отстаёт примерно на
3.3 п.п. от SegFormer и на 5.2 п.п. от сильнейшего SegNeXt. Это реалистичный,
а не порядковый разрыв.

В раннем architecture report SegNeXt без корректного pretraining получил
существенно худший результат. Позднюю цифру 82.1% нельзя смешивать с тем
запуском: история репозитория показывает, что доступность и корректная загрузка
pretrained weights были существенной переменной.

## Наиболее полезные абляции

### Предобработка и аугментации

Коллега не растягивал изображения в квадрат: использовались
`LongestMaxSize(384)` и padding до `384×384`. В augmentation ablation было 13
запусков. Для SegFormer лучший test mean IoU около 81.0% дал умеренный preset
`spatial_color_light`; чрезмерно тяжёлые искажения не были систематически лучше.
У SegNeXt полный preset дал около 81.5%, но и baseline был сильным — около
80.2%. Практический вывод: сначала отдельно проверить удаление геометрического
искажения, затем умеренные spatial + natural-color transforms.

### Learning-rate schedule

В multiclass SegNeXt experiment на 40 эпохах constant LR дал test mIoU 0.4085,
cosine — 0.4483, polynomial — 0.4411, step — 0.4218. Агрессивный one-cycle и
несколько cyclic вариантов были хуже. Это не переносится численно на frozen
DINO head, но делает cosine decay при более длинном обучении разумным следующим
контролируемым фактором. В работе коллеги test использовался много раз для
сравнения вариантов; у нас эту практику повторять нельзя — настройки выбираются
только по validation.

### Порог и выбор метрики

Для WSSS выбор порога по mean IoU приводил к более консервативной маске и
ухудшал disease IoU: в одном из отчётов threshold 0.59 давал disease IoU 29.98%,
тогда как оптимизация mean IoU выбирала 0.73 и давала 27.40%. Большой фон может
маскировать ухудшение поражения. Поэтому главная метрика нашей binary-задачи —
foreground/disease IoU; mean IoU, Dice, precision, recall и pixel AP остаются
диагностическими.

Наш validation sweep для ViT-B conv head показал лишь малый резерв: threshold
0.45 дал foreground IoU 0.6362 против 0.6345 при 0.50 (+0.0017), причём mean
IoU немного снизился. Значит текущий разрыв нельзя закрыть одной калибровкой.

## Выводы из WSSS, CAM, SAM и SPDNet

- Высокая classification accuracy/mAP не гарантирует локализацию. У разных
  классификаторов CAM IoU оставался примерно 15–21% при accuracy 68–74%.
- Binary classifier легко выучивал наличие заболевания, но строил диффузные
  CAM. Multiclass-to-binary агрегация заставляла различать болезни и улучшала
  disease IoU примерно с 20–26% до 30–33%.
- SAM с dense mask prompt провалился (<1% disease IoU), часто выбирая весь лист.
  Bbox + положительные/отрицательные точки из CAM дал около 34%, но был скорее
  precision filter. Протокол prompts важнее самого факта использования SAM.
- В SPDNet spatial cross-attention научился ненулевому gate, но не улучшил
  локализацию без пространственной supervision. Equivariance и contrastive
  auxiliary losses имели тривиальные uniform/collapse решения и дали null result.
- Простая двухслойная segmentation probe на замороженных промежуточных признаках
  дала до 46.2% disease IoU против production CAM около 32%. После совместного
  fine-tuning probe достигала около 59.9%; from-scratch supervised ceiling этой
  архитектуры — raw 64.9% (CRF ухудшал до 61.8%). Это прямое свидетельство, что
  качество readout и промежуточные multi-scale признаки важнее усложнения CAM.
- CRF нужно перенастраивать для каждого распределения seed masks; параметры для
  слабых CAM ухудшали сильные маски примерно на 3 п.п. Sweep на 50 изображениях
  переобучался на 15–20 п.п.; для него требовалось не менее 200 validation images.
- Простое увеличение разрешения с 448 до 896 ухудшало результат из-за изменения
  effective scale/LR и коллапса attention. Разрешение нельзя повышать без
  отдельного контролируемого протокола.

## Что перенять в наш экспериментальный процесс

1. Главным числом сделать disease IoU, всегда показывая оба class IoU.
2. Выбирать checkpoint, threshold и все hyperparameters только по validation;
   test запускать один раз для зафиксированной конфигурации.
3. Сохранять per-image метрики и анализировать распределение disease IoU,
   foreground fraction, over/under-segmentation и зависимость ошибки от размера
   поражения. Среднее число само по себе скрывает главный тип ошибки.
4. В ближайшей серии по одному фактору проверить aspect-preserving resize,
   умеренные аугментации, cosine schedule и multi-layer DINO decoder.
5. После определения лучшего рецепта сделать минимум три seed runs и сообщать
   mean ± std. Только затем переходить к partial fine-tuning.
6. SAM оставить отдельной baseline-веткой с заранее определённым prompt protocol,
   а не использовать как постобработку по умолчанию.

## Приоритет следующих исследований

Непосредственно сейчас наиболее информативна цепочка:

1. ViT-B conv, stretch → aspect-preserving pad при прочих равных;
2. к победителю добавить `spatial_color_light`;
3. multi-layer decoder из четырёх промежуточных слоёв DINOv3;
4. для лучшей архитектуры — 20–30 эпох с cosine decay;
5. три seed runs, validation error analysis и единственная финальная test-оценка.

Если multi-layer decoder не улучшает результат, следующий содержательный шаг —
частично разморозить последние 1–2 блока ViT-B. Масштабирование ViT-B → ViT-L
уже дало только +0.76 п.п. test disease IoU с linear head, поэтому просто брать
ещё более крупную frozen модель менее перспективно, чем улучшать spatial readout.
