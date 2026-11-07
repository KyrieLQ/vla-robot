#!/usr/bin/env bash
set -Eeuo pipefail
main() {
  source "$(dirname "$0")/project_env.sh"
  export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
  [[ $# -ge 1 ]] || { echo 'Usage: evaluate.sh CHECKPOINT_DIRECTORY [EPISODES] [TASK_ID]' >&2; return 2; }
  "$PROJECT_ROOT/envs/openpi/bin/python" "$PROJECT_ROOT/scripts/gpu_run.py" --gpu "${GPU:-auto}" -- \
    "$PROJECT_ROOT/envs/openpi/bin/python" "$PROJECT_ROOT/scripts/closed_loop.py" --checkpoint "$1" --episodes "${2:-2}" --task-id "${3:-0}" --config "${4:-pi05_libero_lora_3090_smoke}"
}
main "$@"
