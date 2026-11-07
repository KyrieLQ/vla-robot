"""Count allocated phase bytes once across all attempts, including hard-link sharing."""
import os,pathlib,subprocess,time
from storage_guard_v2 import StorageMeasurementError,_is_atomic_metadata_race
_metadata_retries=0
def _within(path,root):
    try:path.relative_to(root);return True
    except ValueError:return False
def du_paths_bytes(paths,attempts=5):
    global _metadata_retries
    paths=list(dict.fromkeys(pathlib.Path(p).resolve() for p in paths))
    paths=[p for p in paths if not any(p!=other and _within(p,other) for other in paths)]
    assert paths
    for attempt in range(attempts):
        result=subprocess.run(['du','-s','-c','-B1',*[str(p) for p in paths]],capture_output=True,text=True,env=dict(os.environ,LC_ALL='C'),timeout=30)
        if result.returncode==0:
            fields=result.stdout.splitlines()[-1].split()
            if len(fields)!=2 or fields[1]!='total' or not fields[0].isdigit():raise StorageMeasurementError('Invalid aggregate du output')
            return int(fields[0])
        lines=result.stderr.strip().splitlines()
        retryable=bool(lines) and all(any(_is_atomic_metadata_race(line,p) for p in paths) for line in lines)
        if not retryable or attempt+1==attempts:raise StorageMeasurementError('Aggregate disk measurement failed: '+result.stderr.strip()[:1200])
        _metadata_retries+=1;time.sleep(0.02)
    raise AssertionError('Unreachable')
def du_bytes(path,attempts=5):return du_paths_bytes([path],attempts)
def check(reserve_bytes=0):
    root=pathlib.Path(os.environ['PROJECT_ROOT']);used=du_bytes(root)
    stat=os.statvfs(root);free=stat.f_bavail*stat.f_frsize
    if used+reserve_bytes>120*2**30:raise RuntimeError('Project 120 GiB budget would be exceeded')
    if free-reserve_bytes<100*2**30:raise RuntimeError('Filesystem 100 GiB free-space reserve would be breached')
    return {'used_bytes':used,'available_bytes':free,'reserved_bytes':reserve_bytes,'atomic_metadata_measurement_retries':_metadata_retries}
