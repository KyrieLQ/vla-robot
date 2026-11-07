"""Fail-closed disk accounting with bounded retry for atomic JSON metadata races."""
import os
import pathlib
import re
import subprocess
import time

class StorageMeasurementError(RuntimeError):
    pass

_metadata_retries = 0

def _is_atomic_metadata_race(stderr, root):
    lines = stderr.strip().splitlines()
    if not lines:return False
    root = pathlib.Path(root).resolve()
    for line in lines:
        match = re.fullmatch(r"du: cannot access '([^']+)': No such file or directory", line)
        if not match:return False
        path = pathlib.Path(match.group(1))
        if not path.name.endswith('.json.tmp'):return False
        try:path.resolve().relative_to(root)
        except ValueError:return False
    return True

def du_bytes(path, attempts=5):
    """A failed partial measurement is never accepted; retry only known rename races."""
    global _metadata_retries
    path = pathlib.Path(path).resolve()
    env = dict(os.environ, LC_ALL='C')
    for attempt in range(attempts):
        result = subprocess.run(['du','-s','-B1',str(path)],capture_output=True,text=True,env=env,timeout=30)
        if result.returncode == 0:
            fields = result.stdout.split()
            if not fields or not fields[0].isdigit():raise StorageMeasurementError('Invalid successful du output')
            return int(fields[0])
        retryable = _is_atomic_metadata_race(result.stderr,path)
        if not retryable or attempt+1 == attempts:
            raise StorageMeasurementError('Disk measurement failed; no partial count accepted: '+result.stderr.strip()[:1200])
        _metadata_retries += 1
        print('Storage measurement observed an atomic JSON rename; bounded retry %d/%d' % (attempt+2,attempts),flush=True)
        time.sleep(0.02)
    raise AssertionError('Unreachable')

def check(reserve_bytes=0):
    root=pathlib.Path(os.environ['PROJECT_ROOT'])
    used=du_bytes(root);v=os.statvfs(root);free=v.f_bavail*v.f_frsize
    if used+reserve_bytes>120*2**30:raise RuntimeError('Project 120 GiB budget would be exceeded')
    if free-reserve_bytes<100*2**30:raise RuntimeError('Filesystem 100 GiB free-space reserve would be breached')
    return {'used_bytes':used,'available_bytes':free,'reserved_bytes':reserve_bytes,
            'atomic_metadata_measurement_retries':_metadata_retries}
