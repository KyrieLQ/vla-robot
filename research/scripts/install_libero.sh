#!/usr/bin/env bash
set -Eeuo pipefail
main() {
  source "$(dirname "$0")/project_env.sh"
  export CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu GIT_LFS_SKIP_SMUDGE=1
  cd "$PROJECT_ROOT/repos/openpi"
  if [[ ! -x "$PROJECT_ROOT/envs/libero/bin/python" ]]; then
    uv venv --python 3.8.20 "$PROJECT_ROOT/envs/libero"
  fi
  uv pip install --python "$PROJECT_ROOT/envs/libero/bin/python" \
    -r examples/libero/requirements.txt -r third_party/libero/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy unsafe-best-match
  uv pip install --python "$PROJECT_ROOT/envs/libero/bin/python" -e packages/openpi-client -e third_party/libero
  uv pip check --python "$PROJECT_ROOT/envs/libero/bin/python"
}
main "$@"
