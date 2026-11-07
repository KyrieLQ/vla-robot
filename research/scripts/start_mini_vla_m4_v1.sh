#!/usr/bin/env bash
set -Eeuo pipefail
main() {
  source "$(dirname "$0")/project_env.sh"
  cd "$PROJECT_ROOT"
  [[ -n "${CODEX_THREAD_ID:-}" ]] || { echo 'Originating thread is required' >&2; return 2; }
  export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONFAULTHANDLER=1
  exec /usr/bin/python3 -u -B "$PROJECT_ROOT/scripts/mini_vla_supervisor_m4_v1.py" "$@"
}
main "$@"
