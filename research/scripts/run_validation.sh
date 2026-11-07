#!/usr/bin/env bash
set -Eeuo pipefail
main() {
  source "$(dirname "$0")/project_env.sh"
  # All child processes inherit a bounded set of eight logical CPU cores.
  local cpu_list
  cpu_list=$(/usr/bin/python3 -B -c 'import os;print(",".join(map(str,sorted(os.sched_getaffinity(0))[:8])))')
  exec taskset -c "$cpu_list" "$PROJECT_ROOT/envs/openpi/bin/python" "$PROJECT_ROOT/scripts/run_validation.py" "$@"
}
main "$@"
