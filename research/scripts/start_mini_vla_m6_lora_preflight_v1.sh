#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/project_env.sh"
cd "$PROJECT_ROOT"
[[ -n "${CODEX_THREAD_ID:-}" ]] || { echo 'Originating thread required' >&2; exit 2; }
exec /usr/bin/python3 -u -B "$PROJECT_ROOT/scripts/mini_vla_lora_preflight_supervisor_v1.py" "$@"
