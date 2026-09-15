#!/usr/bin/env bash

project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${PYTHON_BIN:-python}

require_python() {
  if ! command -v "${python_bin}" >/dev/null 2>&1; then
    echo "Python executable not found: ${python_bin}" >&2
    echo "Activate the project environment or set PYTHON_BIN explicitly." >&2
    return 1
  fi
}

find_complete_run() {
  local name=$1
  local seed=$2
  local candidate
  local expected_epochs
  local metric_rows
  local selected=""

  while IFS= read -r candidate; do
    if [[ ! -f "${candidate}/config.yaml" ]] \
      || [[ ! -f "${candidate}/metrics.jsonl" ]] \
      || [[ ! -f "${candidate}/best.pt" ]]; then
      continue
    fi
    expected_epochs=$(sed -nE 's/^[[:space:]]+epochs:[[:space:]]+([0-9]+).*$/\1/p' \
      "${candidate}/config.yaml" | head -n 1)
    metric_rows=$(wc -l < "${candidate}/metrics.jsonl")
    if [[ -n "${expected_epochs}" ]] \
      && [[ "${metric_rows}" -eq "${expected_epochs}" ]] \
      && grep -Eq "^[[:space:]]*seed:[[:space:]]*${seed}[[:space:]]*$" \
        "${candidate}/config.yaml"; then
      selected=${candidate}
    fi
  done < <(find runs -maxdepth 1 -type d -name "*_${name}" | sort)

  if [[ -z "${selected}" ]]; then
    echo "No complete run found for ${name}, seed ${seed}" >&2
    return 1
  fi
  printf '%s\n' "${selected}"
}
