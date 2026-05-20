#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Create or update a Python virtual environment for qcsc-prefect tutorials.

Usage:
  ./scripts/bootstrap_tutorial_env.sh [options]

Options:
  --package SPEC    Package spec to install (default: qcsc-prefect)
  --venv PATH       Virtual environment path (default: .venv)
  --python PATH     Python executable to use (default: python3)
  -h, --help        Show this help

Environment overrides:
  TUTORIAL_PACKAGE  Same as --package
  TUTORIAL_VENV     Same as --venv
  PYTHON            Same as --python

This helper creates a virtual environment if needed, upgrades pip, and installs
the requested package. It does not configure credentials, start services, or
submit HPC jobs.
EOF
}

need_value() {
  local option="$1"
  local value="${2:-}"
  if [[ -z "$value" || "$value" == --* ]]; then
    printf 'Missing value for %s\n\n' "$option" >&2
    usage >&2
    exit 2
  fi
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

VENV_DIR="${TUTORIAL_VENV:-${REPO_ROOT}/.venv}"
PACKAGE_SPEC="${TUTORIAL_PACKAGE:-qcsc-prefect}"
PYTHON_BIN="${PYTHON:-python3}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --package)
      need_value "$1" "${2:-}"
      PACKAGE_SPEC="$2"
      shift 2
      ;;
    --venv)
      need_value "$1" "${2:-}"
      VENV_DIR="$2"
      shift 2
      ;;
    --python)
      need_value "$1" "${2:-}"
      PYTHON_BIN="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown option: %s\n\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  if [[ "${PYTHON_BIN}" == "python3" ]] && command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "Error: Python command not found: ${PYTHON_BIN}" >&2
    echo "Set PYTHON=/path/to/python or pass --python /path/to/python." >&2
    exit 1
  fi
fi

echo "qcsc-prefect tutorial environment bootstrap"
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

Next step:
  source "${VENV_DIR}/bin/activate"

This helper did not configure credentials, start services, or submit jobs.
EOF
