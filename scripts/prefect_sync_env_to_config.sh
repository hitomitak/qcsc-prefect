#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Sync Prefect settings from a auth-file file into Prefect config (profile-scoped).

Usage:
  scripts/prefect_sync_env_to_config.sh [options]

Options:
  -e, --env-file PATH        Path to .env file (default: .env)
  -p, --profile NAME         Prefect profile name to switch to before setting values
      --all-prefect-vars     Sync all PREFECT_* variables found in .env
      --dry-run              Show what would be set without changing config
  -h, --help                 Show this help

Default synced keys:
  - PREFECT_API_URL
  - PREFECT_CLIENT_CUSTOM_HEADERS

Examples:
  scripts/prefect_sync_env_to_config.sh -e auth-file -p mdx
  scripts/prefect_sync_env_to_config.sh -e /path/to/auth-file --all-prefect-vars
EOF
}

run_cmd() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] $*"
  else
    "$@"
  fi
}

ENV_FILE="auth-file"
PROFILE=""
SYNC_ALL=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -e|--env-file)
      ENV_FILE="${2:-}"
      shift 2
      ;;
    -p|--profile)
      PROFILE="${2:-}"
      shift 2
      ;;
    --all-prefect-vars)
      SYNC_ALL=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if ! command -v prefect >/dev/null 2>&1; then
  echo "Error: 'prefect' command not found in PATH." >&2
  exit 1
fi

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Error: env file not found: $ENV_FILE" >&2
  exit 1
fi

if [[ -n "$PROFILE" ]]; then
  run_cmd prefect profile use "$PROFILE" >/dev/null
  echo "Using profile: $PROFILE"
fi

# shellcheck disable=SC1090
set -a
source "$ENV_FILE"
set +a

declare -a KEYS=()
if [[ "$SYNC_ALL" -eq 1 ]]; then
  while IFS= read -r key; do
    KEYS+=("$key")
  done < <(
    grep -E '^[[:space:]]*(export[[:space:]]+)?PREFECT_[A-Za-z0-9_]+=' "$ENV_FILE" \
      | sed -E 's/^[[:space:]]*(export[[:space:]]+)?(PREFECT_[A-Za-z0-9_]+)=.*/\2/' \
      | awk '!seen[$0]++'
  )
else
  KEYS=("PREFECT_API_URL" "PREFECT_CLIENT_CUSTOM_HEADERS")
fi

if [[ ${#KEYS[@]} -eq 0 ]]; then
  echo "No PREFECT_* variables found in $ENV_FILE."
  exit 0
fi

set_count=0
skip_count=0

for key in "${KEYS[@]}"; do
  if [[ -z "${!key+x}" ]]; then
    echo "Skip: $key is not set in $ENV_FILE"
    skip_count=$((skip_count + 1))
    continue
  fi

  value="${!key}"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] would set $key (value hidden)"
  else
    # Keep value hidden from logs since it may include auth tokens.
    prefect config set "${key}=${value}" >/dev/null
    echo "Set: $key"
  fi
  set_count=$((set_count + 1))
done

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Dry-run complete. set=$set_count skip=$skip_count"
else
  echo "Sync complete. set=$set_count skip=$skip_count"
fi
