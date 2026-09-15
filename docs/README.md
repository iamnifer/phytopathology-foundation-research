# Указатель документации

Работа завершена. Здесь отделены итоговые материалы от исторических планов и
замечаний.

## С чего начать

- [Итоговый отчёт](../report/coursework.pdf) — текст курсовой работы.
- [Исследовательская программа](research_plan.md) — вопросы, протокол и
  обоснование.
- [Сводка результатов в JSON](../results/final_metrics.json) — основные
  численные результаты.
- [Финальный аудит](final_audit_2026-09-15.md) — проверка согласованности и
  выполнения замечаний руководителя.

## Журнал экспериментов

Каждая запись содержит вопрос, зафиксированный протокол, результат и вывод.
Полные каталоги запусков и контрольные точки в Git не хранятся.

### Постановка задачи и первые модели

- [000](experiments/000_legacy_notebook.md) — аудит исходного разведочного ноутбука.
- [001](experiments/001_a100_smoke_test.md) — сквозная проверка рабочего процесса.
- [002](experiments/002_multiclass_linear_baseline.md) — разведочная
  116-классовая модель DINOv3.
- [003](experiments/003_binary_smoke_test.md) — проверка бинарного протокола.
- [004](experiments/004_dinov3_vitb16_binary_linear.md) — линейный декодер ViT-B/16.
- [005](experiments/005_dinov3_vitl16_binary_linear.md) — контроль размера ViT-L/16.
- [006](experiments/006_dinov3_vitb16_binary_conv.md) — свёрточный декодер.

### Абляции DINOv3 и анализ ошибок

- [007](experiments/007_binary_threshold_calibration.md) — подбор порога только
  по валидации.
- [008](experiments/008_preprocessing_augmentation_ablation.md) — изменение
  размера входа и аугментации.
- [009](experiments/009_dinov3_multilayer_conv.md) — декодер промежуточных слоёв.
- [010](experiments/010_validation_error_analysis.md) — ошибки и размер поражения.
- [011](experiments/011_multilayer_cosine.md) — более длительное обучение с
  косинусным расписанием.
- [012](experiments/012_decoder_controlled_series.md) — согласованное сравнение
  декодеров по трём запускам.
- [013](experiments/013_small_lesion_oversampling.md) — отрицательный результат
  дополнительной выборки малых поражений.

### Диагностика SAM

- [014](experiments/014_sam_oracle_promptability.md) — SAM с точкой и рамкой,
  полученными из эталонной маски.

### Эффективность разметки

- [015](experiments/015_label_efficiency_seed42.md) — кривая качества для
  случайного, стратифицированного и жадного отбора.
- [016](experiments/016_label_efficiency_repeats.md) — повторы по трём начальным
  значениям при бюджете 10% и 25%.
- [017](experiments/017_subset_composition.md) — состав выбранных подмножеств
  и анализ выбросов.
- [021](experiments/021_kmeans_selection.md) — представители кластеров
  сферического k-means.

### Сравнение моделей и итоговые диагностики

- [018](experiments/018_deeplabv3_baseline.md) — DeepLabV3 с предобучением и
  со случайной инициализацией.
- [019](experiments/019_resolution_control.md) — DINOv3 при 384, 512 и 768 px.
- [020](experiments/020_final_diagnostics.md) — вычислительная стоимость,
  ошибки и оценка на внешнем наборе здоровых растений.
- [022](experiments/022_frozen_linear_probe_comparison.md) — согласованные
  линейные декодеры для замороженных DINOv3 и ResNet-50.
- [023](experiments/023_confirmatory_test_evaluation.md) — подтверждающая
  оценка заранее зафиксированных моделей на тестовой выборке.

## Ревью и справочные материалы

- [Исходное ревью руководителя](reviews/supervisor_review_v1.md) — архив замечаний.
- [План реакции на ревью](review_response_v1.md) — историческая матрица до
  подтверждающей тестовой оценки.
- [Разбор работы коллеги](colleague_experiments_review.md) — анализ близкого
  проекта на PlantSeg.
- [Предварительное обновление для руководителя](supervisor_update_2026-08.md) —
  историческая сводка.
- [Первичный аудит курсовой](coursework_review.md) — ранняя проверка текста,
  сохранённая для истории.

Основные ограничения и итоговые выводы изложены в отчёте и протоколах 022–023.
