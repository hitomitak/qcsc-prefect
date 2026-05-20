#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

VENV_DIR="${BITCOUNT_TUTORIAL_VENV:-${REPO_ROOT}/.venv}"
PACKAGE_SPEC="${BITCOUNT_TUTORIAL_PACKAGE:-qcsc-prefect[qiskit]}"
PYTHON_BIN="${PYTHON:-python3}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  if [[ "${PYTHON_BIN}" == "python3" ]] && command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "Error: Python command not found: ${PYTHON_BIN}" >&2
    echo "Set PYTHON=/path/to/python and rerun this script." >&2
    exit 1
  fi
fi

echo "BitCount tutorial environment bootstrap"
echo "Repository: ${REPO_ROOT}"
echo "Virtual environment: ${VENV_DIR}"
echo "Package: ${PACKAGE_SPEC}"
echo

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Creating virtual environment..."
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
else
  echo "Using existing virtual environment."
fi

if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  echo "Error: virtual environment activate script not found: ${VENV_DIR}/bin/activate" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip
python -m pip install "${PACKAGE_SPEC}"

cat <<EOF

Bootstrap complete.

Next steps:
  1. Activate the environment:
       source "${VENV_DIR}/bin/activate"

  2. Select the Prefect profile for your target environment.
       Miyabi: prefect profile use mdx
       Fugaku: prefect profile use cloud-fugaku

  3. Continue the BitCount tutorial:
       - build the MPI executable on Miyabi or Fugaku
       - create the BitCount blocks with examples/prefect_bitcount_demo/create_blocks.py
       - run examples/prefect_bitcount_demo/flow_optimized.py with --quantum-source random

This helper does not configure IBM Quantum credentials, start Prefect services,
or submit HPC jobs.
EOF
