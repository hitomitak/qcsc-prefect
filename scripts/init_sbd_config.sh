#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Initialize an SBD block configuration TOML file from the checked-in example.

Usage:
  ./scripts/init_sbd_config.sh --target miyabi --project PROJECT --work-dir WORK_DIR --sbd-executable PATH [options]
  ./scripts/init_sbd_config.sh --target fugaku --group GROUP --work-dir WORK_DIR --sbd-executable PATH [options]

Required:
  --target miyabi|fugaku       HPC target for the SBD tutorial
  --work-dir WORK_DIR          Base work directory for SBD jobs
  --sbd-executable PATH        Absolute path to the built SBD diag executable

Target-specific:
  --project PROJECT            Miyabi project name, such as gz00
  --group GROUP                Fugaku group name, such as ra000000

Options:
  --queue QUEUE                Miyabi queue or Fugaku rscgrp; keeps example value if omitted
  --output PATH                Output TOML path
  --solver-mode cpu|gpu        Override solver_mode if present in the example
  --num-nodes N                Override num_nodes if present in the example
  --mpiprocs N                 Override mpiprocs and tutorial mpi_options if present
  --force                      Overwrite an existing output file
  -h, --help                   Show this help

Defaults:
  Miyabi output: algorithms/sbd/sbd_blocks.toml
  Fugaku output: algorithms/sbd/sbd_blocks.fugaku.toml

This script only writes a baseline config file. It does not submit HPC jobs,
create Prefect blocks, configure IBM Quantum credentials, or start services.
EOF
}

error() {
  printf 'Error: %s\n' "$*" >&2
}

warn() {
  printf 'Warning: %s\n' "$*" >&2
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

toml_string() {
  local value="$1"
  value="${value//\\/\\\\}"
  value="${value//\"/\\\"}"
  printf '"%s"' "$value"
}

is_positive_int() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

TARGET=""
PROJECT=""
GROUP=""
QUEUE=""
WORK_DIR=""
SBD_EXECUTABLE=""
OUTPUT=""
SOLVER_MODE=""
NUM_NODES=""
MPIPROCS=""
FORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)
      need_value "$1" "${2:-}"
      TARGET="$2"
      shift 2
      ;;
    --project)
      need_value "$1" "${2:-}"
      PROJECT="$2"
      shift 2
      ;;
    --group)
      need_value "$1" "${2:-}"
      GROUP="$2"
      shift 2
      ;;
    --queue)
      need_value "$1" "${2:-}"
      QUEUE="$2"
      shift 2
      ;;
    --work-dir)
      need_value "$1" "${2:-}"
      WORK_DIR="$2"
      shift 2
      ;;
    --sbd-executable)
      need_value "$1" "${2:-}"
      SBD_EXECUTABLE="$2"
      shift 2
      ;;
    --output)
      need_value "$1" "${2:-}"
      OUTPUT="$2"
      shift 2
      ;;
    --solver-mode)
      need_value "$1" "${2:-}"
      SOLVER_MODE="$2"
      shift 2
      ;;
    --num-nodes)
      need_value "$1" "${2:-}"
      NUM_NODES="$2"
      shift 2
      ;;
    --mpiprocs)
      need_value "$1" "${2:-}"
      MPIPROCS="$2"
      shift 2
      ;;
    --force)
      FORCE=1
      shift
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

if [[ ! -f "mkdocs.yml" || ! -d "algorithms/sbd" ]]; then
  error "run this script from the qcsc-prefect repository root"
  exit 1
fi

case "$TARGET" in
  miyabi)
    EXAMPLE="algorithms/sbd/sbd_blocks.example.toml"
    DEFAULT_OUTPUT="algorithms/sbd/sbd_blocks.toml"
    if [[ -z "$PROJECT" ]]; then
      error "--project is required when --target miyabi"
      exit 2
    fi
    ;;
  fugaku)
    EXAMPLE="algorithms/sbd/sbd_blocks.fugaku.example.toml"
    DEFAULT_OUTPUT="algorithms/sbd/sbd_blocks.fugaku.toml"
    if [[ -z "$GROUP" ]]; then
      error "--group is required when --target fugaku"
      exit 2
    fi
    ;;
  "")
    error "--target is required"
    exit 2
    ;;
  *)
    error "--target must be either 'miyabi' or 'fugaku'"
    exit 2
    ;;
esac

if [[ -z "$WORK_DIR" ]]; then
  error "--work-dir is required"
  exit 2
fi

if [[ -z "$SBD_EXECUTABLE" ]]; then
  error "--sbd-executable is required"
  exit 2
fi

if [[ -n "$SOLVER_MODE" && "$SOLVER_MODE" != "cpu" && "$SOLVER_MODE" != "gpu" ]]; then
  error "--solver-mode must be either 'cpu' or 'gpu'"
  exit 2
fi

if [[ -n "$NUM_NODES" ]] && ! is_positive_int "$NUM_NODES"; then
  error "--num-nodes must be a positive integer"
  exit 2
fi

if [[ -n "$MPIPROCS" ]] && ! is_positive_int "$MPIPROCS"; then
  error "--mpiprocs must be a positive integer"
  exit 2
fi

if [[ ! -f "$EXAMPLE" ]]; then
  error "example config was not found: $EXAMPLE"
  exit 1
fi

if [[ -z "$OUTPUT" ]]; then
  OUTPUT="$DEFAULT_OUTPUT"
fi

if [[ -e "$OUTPUT" && "$FORCE" -ne 1 ]]; then
  error "$OUTPUT already exists; rerun with --force to overwrite it"
  exit 1
fi

if [[ -z "$QUEUE" ]]; then
  warn "--queue was not provided; keeping the queue value from $EXAMPLE"
fi

if [[ "$SOLVER_MODE" == "gpu" ]]; then
  warn "--solver-mode gpu only updates solver_mode; review queue, block names, modules, and executable"
fi

OUTPUT_DIR="$(dirname "$OUTPUT")"
if [[ ! -d "$OUTPUT_DIR" ]]; then
  error "output directory does not exist: $OUTPUT_DIR"
  exit 1
fi

TEMP_PATH="${OUTPUT}.tmp"

{
  printf '# Generated by scripts/init_sbd_config.sh from %s\n' "$EXAMPLE"
  printf '# Review solver and site-specific parameters before creating Prefect blocks.\n\n'
  while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ "$line" == "# Copy this file to algorithms/sbd/sbd_blocks.toml and adjust the required paths/group." ]]; then
      printf '# Rerun scripts/init_sbd_config.sh with --force to update this file.\n'
      continue
    fi
    if [[ "$line" =~ ^[[:space:]]*([A-Za-z0-9_]+)[[:space:]]*= ]]; then
      key="${BASH_REMATCH[1]}"
      case "$key" in
        hpc_target)
          printf 'hpc_target = %s\n' "$(toml_string "$TARGET")"
          ;;
        project)
          if [[ "$TARGET" == "miyabi" ]]; then
            printf 'project = %s\n' "$(toml_string "$PROJECT")"
          else
            printf '%s\n' "$line"
          fi
          ;;
        group)
          if [[ "$TARGET" == "fugaku" ]]; then
            printf 'group = %s\n' "$(toml_string "$GROUP")"
          else
            printf '%s\n' "$line"
          fi
          ;;
        queue)
          if [[ -n "$QUEUE" ]]; then
            printf 'queue = %s\n' "$(toml_string "$QUEUE")"
          else
            printf '%s\n' "$line"
          fi
          ;;
        work_dir)
          printf 'work_dir = %s\n' "$(toml_string "$WORK_DIR")"
          ;;
        sbd_executable)
          printf 'sbd_executable = %s\n' "$(toml_string "$SBD_EXECUTABLE")"
          ;;
        solver_mode)
          if [[ -n "$SOLVER_MODE" ]]; then
            printf 'solver_mode = %s\n' "$(toml_string "$SOLVER_MODE")"
          else
            printf '%s\n' "$line"
          fi
          ;;
        num_nodes)
          if [[ -n "$NUM_NODES" ]]; then
            printf 'num_nodes = %s\n' "$NUM_NODES"
          else
            printf '%s\n' "$line"
          fi
          ;;
        mpiprocs)
          if [[ -n "$MPIPROCS" ]]; then
            printf 'mpiprocs = %s\n' "$MPIPROCS"
          else
            printf '%s\n' "$line"
          fi
          ;;
        mpi_options)
          if [[ -n "$MPIPROCS" && "$TARGET" == "miyabi" ]]; then
            printf 'mpi_options = ["-np", "%s"]\n' "$MPIPROCS"
          elif [[ -n "$MPIPROCS" && "$TARGET" == "fugaku" ]]; then
            printf 'mpi_options = ["-n", "%s"]\n' "$MPIPROCS"
          else
            printf '%s\n' "$line"
          fi
          ;;
        *)
          printf '%s\n' "$line"
          ;;
      esac
    else
      printf '%s\n' "$line"
    fi
  done <"$EXAMPLE"
} >"$TEMP_PATH"

mv "$TEMP_PATH" "$OUTPUT"

printf 'Wrote %s\n\n' "$OUTPUT"
printf 'Next command:\n'
printf '  python algorithms/sbd/create_blocks.py \\\n'
printf '    --config %s \\\n' "$OUTPUT"
printf '    --hpc-target %s\n' "$TARGET"
