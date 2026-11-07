"""Reuse validated admission when executing from a frozen source directory."""
import os
import pathlib
import gpu_run

gpu_run.ROOT = pathlib.Path(os.environ['PROJECT_ROOT'])

if __name__ == '__main__':
    raise SystemExit(gpu_run.main())
