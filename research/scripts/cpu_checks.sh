#!/usr/bin/env bash
set -Eeuo pipefail
main() {
  source "$(dirname "$0")/project_env.sh"
  export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
  local python="$PROJECT_ROOT/envs/openpi/bin/python"
  "$python" "$PROJECT_ROOT/scripts/parameter_audit.py"
  "$python" "$PROJECT_ROOT/scripts/data_audit.py"
  bash "$PROJECT_ROOT/scripts/compute_norm_stats.sh" smoke
  "$python" "$PROJECT_ROOT/scripts/batch_audit.py"
  "$python" "$PROJECT_ROOT/scripts/base_weight_audit.py"
}
main "$@"
