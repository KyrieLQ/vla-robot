#!/usr/bin/env bash
main() {
    set -Eeuo pipefail
    source "$(dirname "$0")/project_env.sh"
    cd "$PROJECT_ROOT"
    [[ -n "${CODEX_THREAD_ID:-}" ]] || return 2
    exec /usr/bin/python3 -u -B "$PROJECT_ROOT/scripts/mini_vla_pi05_formal_supervisor_v3.py" "$@"
}
main "$@"
