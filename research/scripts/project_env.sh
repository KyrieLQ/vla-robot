#!/usr/bin/env bash
# Source only in project subprocesses; never changes HOME or global shell files.
export PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export UV_CACHE_DIR="$PROJECT_ROOT/cache/uv"
export UV_PYTHON_INSTALL_DIR="$PROJECT_ROOT/envs/python"
export UV_TOOL_DIR="$PROJECT_ROOT/tools/uv-tools"
export UV_TOOL_BIN_DIR="$PROJECT_ROOT/tools/bin"
export UV_PROJECT_ENVIRONMENT="$PROJECT_ROOT/envs/openpi"
export PIP_CACHE_DIR="$PROJECT_ROOT/cache/pip"
export HF_HOME="$PROJECT_ROOT/cache/huggingface"
export HF_LEROBOT_HOME="$PROJECT_ROOT/datasets/lerobot"
unset LEROBOT_HOME  # This pinned LeRobot explicitly rejects the deprecated variable.
export OPENPI_DATA_HOME="$PROJECT_ROOT/cache/openpi"
export JAX_COMPILATION_CACHE_DIR="$PROJECT_ROOT/cache/jax"
export XDG_CACHE_HOME="$PROJECT_ROOT/cache/xdg"
export XDG_CONFIG_HOME="$PROJECT_ROOT/configs/xdg"
export XDG_DATA_HOME="$PROJECT_ROOT/cache/xdg-data"
export TMPDIR="$PROJECT_ROOT/tmp"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT_ROOT/repos/openpi/third_party/libero:$PROJECT_ROOT/scripts"
export PYNPUT_BACKEND=dummy
export MPLCONFIGDIR="$PROJECT_ROOT/cache/matplotlib"
export NUMBA_CACHE_DIR="$PROJECT_ROOT/cache/numba"
export TORCH_HOME="$PROJECT_ROOT/cache/torch"
export LIBERO_CONFIG_PATH="$PROJECT_ROOT/configs/libero"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PATH="$PROJECT_ROOT/tools/bin:$PATH"
# pip-provided CUDA libraries must not be shadowed by another environment.
unset LD_LIBRARY_PATH
mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$UV_TOOL_DIR" "$UV_TOOL_BIN_DIR" \
  "$PIP_CACHE_DIR" "$HF_HOME" "$HF_LEROBOT_HOME" "$OPENPI_DATA_HOME" \
  "$JAX_COMPILATION_CACHE_DIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" \
  "$XDG_DATA_HOME" "$TMPDIR" "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR" \
  "$TORCH_HOME" "$LIBERO_CONFIG_PATH"
