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
cp build/coursework.pdf coursework.pdf
```

Свежая сборка сначала создаётся как `report/build/coursework.pdf`, а второй
командой копируется в ожидаемый для просмотра путь `report/coursework.pdf`.
Финальная копия `report/coursework.pdf` хранится в Git для удобного просмотра,
а каталог `report/build/`, справочные PDF и примеры игнорируются и не являются
частью собираемого документа.
