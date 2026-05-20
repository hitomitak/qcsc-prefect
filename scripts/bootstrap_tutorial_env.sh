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
  --uv PATH         uv executable to use when Python is too old (default: uv)
  -h, --help        Show this help

Environment overrides:
  TUTORIAL_PACKAGE  Same as --package
  TUTORIAL_VENV     Same as --venv
  PYTHON            Same as --python
  UV                Same as --uv

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

python_is_supported() {
  local python_bin="$1"
  "${python_bin}" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' \
    >/dev/null 2>&1
}

python_version_text() {
  local python_bin="$1"
  "${python_bin}" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))' 2>/dev/null \
    || printf 'unknown'
}

print_uv_install_hint() {
  cat >&2 <<'EOF'
Install uv first, then rerun this script:

  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  ./scripts/bootstrap_tutorial_env.sh

See docs/howto/howto_setup_python_env.md for the MDX workflow client setup.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

VENV_DIR="${TUTORIAL_VENV:-${REPO_ROOT}/.venv}"
PACKAGE_SPEC="${TUTORIAL_PACKAGE:-qcsc-prefect}"
PYTHON_BIN="${PYTHON:-python3}"
UV_BIN="${UV:-uv}"

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
    --uv)
      need_value "$1" "${2:-}"
      UV_BIN="$2"
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

PYTHON_AVAILABLE=0
if command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_AVAILABLE=1
elif [[ "${PYTHON_BIN}" == "python3" ]] && command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
  PYTHON_AVAILABLE=1
fi

echo "qcsc-prefect tutorial environment bootstrap"
echo "Repository: ${REPO_ROOT}"
echo "Virtual environment: ${VENV_DIR}"
echo "Package: ${PACKAGE_SPEC}"
echo

if [[ ! -d "${VENV_DIR}" ]]; then
  if [[ "$PYTHON_AVAILABLE" -eq 1 ]] && python_is_supported "${PYTHON_BIN}"; then
    echo "Creating virtual environment with ${PYTHON_BIN}..."
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  elif command -v "${UV_BIN}" >/dev/null 2>&1; then
    if [[ "$PYTHON_AVAILABLE" -eq 1 ]]; then
      echo "Python $(python_version_text "${PYTHON_BIN}") is too old; creating ${VENV_DIR} with uv and Python 3.12..."
    else
      echo "Python command not found; creating ${VENV_DIR} with uv and Python 3.12..."
    fi
    "${UV_BIN}" venv "${VENV_DIR}" -p 3.12
  else
    if [[ "$PYTHON_AVAILABLE" -eq 1 ]]; then
      echo "Error: Python $(python_version_text "${PYTHON_BIN}") is too old; tutorials require Python >= 3.11." >&2
    else
      echo "Error: Python command not found: ${PYTHON_BIN}" >&2
    fi
    print_uv_install_hint
    exit 1
  fi
else
  echo "Using existing virtual environment."
fi

if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  echo "Error: virtual environment activate script not found: ${VENV_DIR}/bin/activate" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

if ! python_is_supported python; then
  cat >&2 <<EOF
Error: ${VENV_DIR} uses Python $(python_version_text python), but tutorials require Python >= 3.11.

Remove the existing virtual environment and rerun this script after installing
uv if the system Python is too old:

  rm -rf "${VENV_DIR}"
  ./scripts/bootstrap_tutorial_env.sh
EOF
  exit 1
fi

if ! python -m pip --version >/dev/null 2>&1; then
  echo "pip is not available in ${VENV_DIR}; bootstrapping pip with ensurepip..."
  if ! python -m ensurepip --upgrade; then
    cat >&2 <<EOF
Error: could not bootstrap pip in ${VENV_DIR}.

Your Python installation may not include ensurepip. Use a Python build with
venv/ensurepip support, or remove ${VENV_DIR} and recreate it after pip is
available.
EOF
    exit 1
  fi
fi

python -m pip install --upgrade pip
python -m pip install "${PACKAGE_SPEC}"

cat <<EOF

Bootstrap complete.

Next step:
  source "${VENV_DIR}/bin/activate"

This helper did not configure credentials, start services, or submit jobs.
EOF
