#!/usr/bin/env bash
set -Eeuo pipefail
main() {
  source "$(dirname "$0")/project_env.sh"
  export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
  [[ $# -eq 2 ]] || { echo 'Usage: train.sh EXPLICIT_TOTAL_STEPS UNIQUE_EXPERIMENT_NAME' >&2; return 2; }
  [[ -f "$PROJECT_ROOT/reports/spatial_full_manifest.json" ]] || { echo 'Prepare full spatial data and its separate norm stats first' >&2; return 2; }
  local count="${TRAIN_GPUS:-1}"
  local reason="${VLA_MULTI_GPU_REASON:-Explicit user-authorized parallel training}"
  "$PROJECT_ROOT/envs/openpi/bin/python" "$PROJECT_ROOT/scripts/gpu_run.py" --gpu "${GPU:-auto}" --gpus "$count" --multi-card-reason "$reason" -- \
    "$PROJECT_ROOT/envs/openpi/bin/python" "$PROJECT_ROOT/scripts/train_verified.py" --mode train --steps "$1" --exp-name "$2" --fsdp-devices "$count"
}
main "$@"
