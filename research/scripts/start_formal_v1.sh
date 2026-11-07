#!/usr/bin/env bash
set -Eeuo pipefail
main() {
  source "$(dirname "$0")/project_env.sh"
  [[ -n "${CODEX_THREAD_ID:-}" ]] || { echo 'Originating CODEX_THREAD_ID is required for completion follow-through' >&2; return 2; }
  export TRAIN_GPUS=2 NCCL_DEBUG=WARN PYTHONFAULTHANDLER=1
  exec /usr/bin/python3 -u -B "$PROJECT_ROOT/scripts/formal_stage_supervisor.py" \
    --name formal_pipeline_20261013_v1 --timeout 86400 -- \
    "$PROJECT_ROOT/envs/openpi/bin/python" -u -B "$PROJECT_ROOT/scripts/formal_pipeline_v1.py" \
    --run-dir "$PROJECT_ROOT/results/formal_pipeline_20261013_v1" \
    --experiment formal_spatial_20261013_v1
}
main "$@"
