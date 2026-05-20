#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Check whether the Miyabi BitCount tutorial prerequisites appear ready.

Usage:
  ./scripts/check_miyabi_tutorial_prereqs.sh [options]

Options:
  --profile NAME       Prefect profile to check (default: mdx)
  --project NAME       Miyabi project/group name, such as gz00
  --work-dir PATH      Writable tutorial work directory to check or create
  --mdx-host HOST      Optional SSH config host for the MDX workflow client
  --miyabi-host HOST   Optional SSH config host for Miyabi-C
  -h, --help           Show this help

Example:
  ./scripts/check_miyabi_tutorial_prereqs.sh \
    --profile mdx \
    --project gz00 \
    --work-dir /work/gz00/z12345/miyabi_tutorial

This script does not submit HPC jobs, create Prefect blocks, configure IBM
Quantum credentials, or start Prefect services.
EOF
}

PROFILE="mdx"
PROJECT=""
WORK_DIR=""
MDX_HOST=""
MIYABI_HOST=""
fail_count=0
warn_count=0

pass() {
  printf 'PASS: %s\n' "$*"
}

warn() {
  warn_count=$((warn_count + 1))
  printf 'WARN: %s\n' "$*"
}

fail() {
  fail_count=$((fail_count + 1))
  printf 'FAIL: %s\n' "$*"
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

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      need_value "$1" "${2:-}"
      PROFILE="$2"
      shift 2
      ;;
    --project)
      need_value "$1" "${2:-}"
      PROJECT="$2"
      shift 2
      ;;
    --work-dir)
      need_value "$1" "${2:-}"
      WORK_DIR="$2"
      shift 2
      ;;
    --mdx-host)
      need_value "$1" "${2:-}"
      MDX_HOST="$2"
      shift 2
      ;;
    --miyabi-host)
      need_value "$1" "${2:-}"
      MIYABI_HOST="$2"
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

printf 'Miyabi BitCount tutorial preflight\n'
printf 'Profile: %s\n' "$PROFILE"
if [[ -n "$PROJECT" ]]; then
  printf 'Project: %s\n' "$PROJECT"
fi
if [[ -n "$WORK_DIR" ]]; then
  printf 'Work directory: %s\n' "$WORK_DIR"
fi
printf '\n'

if [[ -f "mkdocs.yml" && -d "docs/tutorials" && -d "examples/prefect_bitcount_demo" ]]; then
  pass "running from the qcsc-prefect repository root"
else
  fail "run this script from the qcsc-prefect repository root"
fi

if command -v python3 >/dev/null 2>&1; then
  pass "python3 is available ($(python3 --version 2>&1))"
else
  fail "python3 is not available; install Python 3 before continuing"
fi

if command -v git >/dev/null 2>&1; then
  pass "git is available"
else
  fail "git is not available; install git before continuing"
fi

if [[ -d ".venv" ]]; then
  pass ".venv exists"
else
  warn ".venv does not exist; run ./scripts/bootstrap_bitcount_tutorial_env.sh before executing the flow"
fi

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  if command -v prefect >/dev/null 2>&1; then
    pass "prefect CLI is available in the active virtual environment"
  else
    warn "a virtual environment is active, but prefect CLI is not available in PATH"
  fi
else
  warn "no virtual environment is active; activate .venv before running Prefect commands"
fi

prefect_available=0
if command -v prefect >/dev/null 2>&1; then
  prefect_available=1
fi

if [[ "$prefect_available" -eq 1 ]]; then
  if profile_output="$(prefect profile ls 2>/dev/null)"; then
    if grep -Eq "(^|[[:space:]])${PROFILE}($|[[:space:]])" <<<"$profile_output"; then
      pass "Prefect profile '${PROFILE}' exists"
    else
      warn "Prefect profile '${PROFILE}' was not found; create/select it before block creation and flow runs"
    fi
  else
    warn "could not list Prefect profiles with 'prefect profile ls'"
  fi

  if [[ -n "${PREFECT_API_URL:-}" ]]; then
    pass "PREFECT_API_URL is set in the environment"
  elif config_output="$(PREFECT_PROFILE="$PROFILE" prefect config view --show-sources 2>/dev/null)"; then
    if grep -q "PREFECT_API_URL" <<<"$config_output"; then
      pass "PREFECT_API_URL is configured for profile '${PROFILE}'"
    else
      warn "PREFECT_API_URL is not configured for profile '${PROFILE}'"
    fi
  else
    warn "could not inspect Prefect config for profile '${PROFILE}'"
  fi
else
  warn "prefect CLI is not available; activate .venv after running the bootstrap helper"
fi

if [[ "$PROFILE" == "mdx" || "$PROFILE" == *"mdx"* ]]; then
  if command -v prefect-auth >/dev/null 2>&1; then
    pass "prefect-auth is available for the MDX on-prem profile"
  else
    warn "prefect-auth is not available; if your MDX token expires, follow the MDX Prefect setup guidance"
  fi
fi

if [[ -f "scripts/prefect_sync_env_to_config.sh" ]]; then
  pass "scripts/prefect_sync_env_to_config.sh exists"
else
  fail "scripts/prefect_sync_env_to_config.sh is missing"
fi

if [[ -n "$WORK_DIR" ]]; then
  if [[ -d "$WORK_DIR" ]]; then
    if [[ -w "$WORK_DIR" ]]; then
      pass "work directory exists and is writable: $WORK_DIR"
    else
      fail "work directory exists but is not writable: $WORK_DIR"
    fi
  elif mkdir -p "$WORK_DIR" 2>/dev/null; then
    pass "work directory can be created: $WORK_DIR"
  else
    fail "work directory cannot be created: $WORK_DIR"
  fi

  if [[ -n "$PROJECT" && "$WORK_DIR" != *"/${PROJECT}/"* && "$WORK_DIR" != *"/${PROJECT}" ]]; then
    warn "work directory does not appear to include project '${PROJECT}'; confirm the path matches your Miyabi allocation"
  fi
else
  warn "no --work-dir provided; pass a writable Miyabi work directory to check it"
fi

if [[ -f "examples/prefect_bitcount_demo/bitcount_blocks.example.toml" ]]; then
  pass "BitCount block config example exists"
else
  fail "examples/prefect_bitcount_demo/bitcount_blocks.example.toml is missing"
fi

if [[ -f "examples/prefect_bitcount_demo/build_on_miyabi.sh" ]]; then
  pass "Miyabi build script exists"
else
  fail "examples/prefect_bitcount_demo/build_on_miyabi.sh is missing"
fi

check_ssh_host() {
  local label="$1"
  local host="$2"
  if [[ -z "$host" ]]; then
    warn "no ${label} SSH host provided; pass --${label}-host to check ~/.ssh/config"
    return
  fi

  if [[ ! -f "${HOME}/.ssh/config" ]]; then
    warn "~/.ssh/config was not found; cannot check SSH host '${host}'"
    return
  fi

  if awk -v target="$host" '
    /^[[:space:]]*[Hh][Oo][Ss][Tt][[:space:]]+/ {
      for (i = 2; i <= NF; i++) {
        if ($i == target) {
          found = 1
        }
      }
    }
    END { exit found ? 0 : 1 }
  ' "${HOME}/.ssh/config"; then
    pass "SSH config contains ${label} host '${host}'"
  else
    warn "SSH config does not appear to contain ${label} host '${host}'"
  fi
}

check_ssh_host "mdx" "$MDX_HOST"
check_ssh_host "miyabi" "$MIYABI_HOST"

printf '\nSummary: %d failure(s), %d warning(s)\n' "$fail_count" "$warn_count"

if [[ "$fail_count" -gt 0 ]]; then
  printf 'Preflight failed. Fix the FAIL items before continuing the random-source tutorial path.\n'
  exit 1
fi

printf 'Preflight completed. WARN items may still need attention before running the full tutorial.\n'
