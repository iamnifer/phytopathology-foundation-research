# Текст курсовой работы

Основной файл — `coursework.tex`. Документ собирается автономно через XeLaTeX
и biber. Поля титульного листа находятся в `metadata.tex` и заполнены по
сданному отчёту КТ-1.

На Ubuntu необходимы пакеты:

```bash
sudo apt-get install latexmk texlive-xetex texlive-lang-cyrillic \
  texlive-latex-extra biber fonts-liberation
```

Сборка из каталога `report/`:

```bash
latexmk coursework.tex
```

PDF и вспомогательные файлы создаются в `report/build/`. Справочные PDF и
примеры в этом каталоге не являются частью собираемого документа.
