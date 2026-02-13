#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/src"
BIN_DIR="${SCRIPT_DIR}/bin"

mkdir -p "${BIN_DIR}"

module list >/dev/null 2>&1 || true

if ! command -v mpiicpx >/dev/null 2>&1; then
  echo "mpiicpx not found. Load Intel oneAPI modules first (e.g. intel + impi)." >&2
  exit 1
fi

mpiicpx -O2 -axSAPPHIRERAPIDS,CORE-AVX512 -o "${BIN_DIR}/get_counts_json" "${SRC_DIR}/get_counts_json.cpp"
mpiicpx -O3 -axSAPPHIRERAPIDS,CORE-AVX512 -o "${BIN_DIR}/get_counts_hist" "${SRC_DIR}/get_counts_hist.cpp"

echo "Built binaries:"
ls -l "${BIN_DIR}/get_counts_json" "${BIN_DIR}/get_counts_hist"
