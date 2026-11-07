#!/usr/bin/env bash
set -e
HERE="$(cd -- "$(dirname -- "$0")" && pwd)"
POLICY_PY="$HOME/xmh/openpi-agilex/.venv/bin/python"

with_ros() (
  export PATH="/usr/bin:/bin:/usr/sbin:/sbin"
  unset PYTHONPATH LD_LIBRARY_PATH
  source /opt/ros/noetic/setup.bash
  export ROS_MASTER_URI=http://127.0.0.1:11311 ROS_IP=127.0.0.1 ROS_HOSTNAME=localhost
  exec /usr/bin/python3 -B -u "$@"
)

with_dataset() (
  export PYTHONPATH="$HOME/xmh/openpi-agilex/src:$HOME/xmh/openpi-agilex/packages/openpi-client/src"
  unset LEROBOT_HOME
  export CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
  exec "$POLICY_PY" -B -u "$@"
)

doctor() {
  with_ros -c 'import rospy, rosgraph; from sensor_msgs.msg import Image; from PIL import features; assert features.check("jpg"), "Pillow lacks JPEG support"; print("OK 采集环境")'
  with_dataset -c 'from lerobot.common.datasets.lerobot_dataset import LeRobotDataset; from openpi.policies.aloha_policy import AlohaInputs; from openpi.transforms import RepackTransform, DeltaActions, make_bool_mask; print("OK LeRobot / OpenPI 环境")'
}

case "${1:-}" in
  sensors) exec /usr/bin/python3 -I -B -u "$HERE/site_sensors.py" ;;
  first)
    doctor
    receipt="$(mktemp)"
    trap 'rm -f -- "$receipt"' EXIT
    with_ros "$HERE/capture.py" --receipt "$receipt"
    IFS= read -r take < "$receipt"
    with_dataset "$HERE/dataset.py" review "$take"
    echo "回看页面 $take/review.html"
    if [[ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]] && command -v xdg-open >/dev/null; then
      xdg-open "$take/review.html" >/dev/null 2>&1 &
    fi
    echo '三路回看确认抓的是圆柱、松爪后留在盒内、无人手帮忙，再输入 yes 回车导出。只回车会保留原始数据。'
    IFS= read -r answer
    if [[ "$answer" == yes ]]; then
      with_dataset "$HERE/dataset.py" export --reviewed-success "$take"
    else
      echo "已保留本条 $take"
    fi ;;
  doctor) doctor ;;
  record) shift; with_ros "$HERE/capture.py" "$@" ;;
  review|export|verify) with_dataset "$HERE/dataset.py" "$@" ;;
  *) echo "用法: bash demo.sh sensors | first | doctor | record | review 示范目录 | export --reviewed-success 示范目录... | verify 导出目录" >&2; exit 2 ;;
esac
