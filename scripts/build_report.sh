#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tinytex_bin="${HOME}/.TinyTeX/bin/x86_64-linux"

if ! command -v latexmk >/dev/null 2>&1 && [ -x "${tinytex_bin}/latexmk" ]; then
  export PATH="${tinytex_bin}:${PATH}"
fi

if ! command -v latexmk >/dev/null 2>&1; then
  echo "latexmk not found; install TinyTeX or TeX Live" >&2
  exit 1
fi

cd "${repo_root}/report"
latexmk coursework.tex
cp build/coursework.pdf coursework.pdf
echo "Built ${repo_root}/report/coursework.pdf"
