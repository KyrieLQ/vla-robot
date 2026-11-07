#!/usr/bin/env bash
set -Eeuo pipefail
main() {
  source "$(dirname "$0")/project_env.sh"
  cd "$PROJECT_ROOT"
  [[ -n "${CODEX_THREAD_ID:-}" ]] || { echo 'Originating thread required' >&2; return 2; }
  export CUDA_CACHE_PATH="$PROJECT_ROOT/cache/cuda"
  export TORCHINDUCTOR_CACHE_DIR="$PROJECT_ROOT/cache/torchinductor"
  export PYTHONFAULTHANDLER=1
  exec /usr/bin/python3 -u -B "$PROJECT_ROOT/scripts/mini_vla_final_preflight_supervisor_v1.py" "$@"
}
main "$@"
