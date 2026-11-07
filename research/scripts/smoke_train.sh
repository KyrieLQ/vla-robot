#!/usr/bin/env bash
set -Eeuo pipefail
main() {
  source "$(dirname "$0")/project_env.sh"
  export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
  local exp="${1:-smoke_$(date +%Y%m%d_%H%M%S)}"
  local count="${TRAIN_GPUS:-1}"
  local reason="${VLA_MULTI_GPU_REASON:-Explicit user-authorized parallel validation}"
  "$PROJECT_ROOT/envs/openpi/bin/python" "$PROJECT_ROOT/scripts/gpu_run.py" --gpu "${GPU:-auto}" --gpus "$count" --multi-card-reason "$reason" -- \
    "$PROJECT_ROOT/envs/openpi/bin/python" "$PROJECT_ROOT/scripts/train_verified.py" --exp-name "$exp" --steps 20 --fsdp-devices "$count"
}
main "$@"
