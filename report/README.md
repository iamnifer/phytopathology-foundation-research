# Отчёт по курсовой работе

Отчёт написан на русском языке и собирается с помощью XeLaTeX, biber и latexmk.
Итоговая версия хранится в [coursework.pdf](coursework.pdf), основной исходный
файл — [coursework.tex](coursework.tex).

## Сборка из корня репозитория

```bash
./scripts/build_report.sh
```

Скрипт находит TinyTeX, если он установлен, запускает latexmk в каталоге
`report/` и копирует `build/coursework.pdf` в `coursework.pdf`.

## Системные пакеты в Ubuntu

```bash
sudo apt-get install latexmk texlive-xetex texlive-lang-cyrillic \
  texlive-latex-extra biber fonts-liberation
```

Равнозначная ручная сборка:

```bash
cd report
latexmk coursework.tex
cp build/coursework.pdf coursework.pdf
```

Поля титульного листа заданы в `metadata.tex`, библиография — в `refs.bib`,
иллюстрации — в `figures/`. Файлы сборки и полученные отдельно правила
оформления намеренно исключены из Git.
