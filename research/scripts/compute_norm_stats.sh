#!/usr/bin/env bash
set -Eeuo pipefail
main() {
  source "$(dirname "$0")/project_env.sh"
  export CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
  local mode="${1:-smoke}"
  [[ "$mode" == smoke || "$mode" == train ]] || { echo 'mode must be smoke or train' >&2; return 2; }
  cd "$PROJECT_ROOT/repos/openpi"
  "$PROJECT_ROOT/envs/openpi/bin/python" scripts/compute_norm_stats.py --config-name "pi05_libero_lora_3090_${mode}"
}
main "$@"
