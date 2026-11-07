#!/usr/bin/env bash
set -Eeuo pipefail
main() {
  source "$(dirname "$0")/project_env.sh"
  export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
  [[ $# -ge 2 ]] || { echo 'Usage: serve_policy.sh CHECKPOINT_DIRECTORY FREE_LOCAL_PORT [CONFIG_NAME]' >&2; return 2; }
  "$PROJECT_ROOT/envs/openpi/bin/python" "$PROJECT_ROOT/scripts/gpu_run.py" --gpu "${GPU:-auto}" -- \
    "$PROJECT_ROOT/envs/openpi/bin/python" "$PROJECT_ROOT/repos/openpi/scripts/serve_policy.py" \
      --host 127.0.0.1 --port "$2" policy:checkpoint --policy.config "${3:-pi05_libero_lora_3090_smoke}" --policy.dir "$1"
}
main "$@"
