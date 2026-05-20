#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
GENERIC_BOOTSTRAP="${SCRIPT_DIR}/bootstrap_tutorial_env.sh"

if [[ ! -x "${GENERIC_BOOTSTRAP}" ]]; then
  echo "Error: generic tutorial bootstrap helper not found: ${GENERIC_BOOTSTRAP}" >&2
  exit 1
fi

for arg in "$@"; do
  if [[ "$arg" == "-h" || "$arg" == "--help" ]]; then
    "${GENERIC_BOOTSTRAP}" "$@"
    exit 0
  fi
done

"${GENERIC_BOOTSTRAP}" \
  --package "${BITCOUNT_TUTORIAL_PACKAGE:-qcsc-prefect[qiskit]}" \
  --venv "${BITCOUNT_TUTORIAL_VENV:-${REPO_ROOT}/.venv}" \
  "$@"

cat <<EOF

BitCount tutorial next steps:
  1. Select the Prefect profile for your target environment.
       Miyabi: prefect profile use mdx
       Fugaku: prefect profile use cloud-fugaku

  2. Continue the BitCount tutorial:
       - build the MPI executable on Miyabi or Fugaku
       - create the BitCount blocks with examples/prefect_bitcount_demo/create_blocks.py
       - run examples/prefect_bitcount_demo/flow_optimized.py with --quantum-source random

The bootstrap helper does not configure IBM Quantum credentials, start Prefect
services, or submit HPC jobs.
EOF
