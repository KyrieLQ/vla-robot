"""Account only for this project, including hard-link deduplication, before writes."""
import os, pathlib, subprocess

def check(reserve_bytes=0):
    root=pathlib.Path(os.environ['PROJECT_ROOT'])
    used=int(subprocess.check_output(['du','-s','-B1',str(root)],text=True).split()[0])
    v=os.statvfs(root);free=v.f_bavail*v.f_frsize
    if used+reserve_bytes>120*2**30:raise RuntimeError('Project 120 GiB budget would be exceeded')
    if free-reserve_bytes<100*2**30:raise RuntimeError('Filesystem 100 GiB free-space reserve would be breached')
    return {'used_bytes':used,'available_bytes':free,'reserved_bytes':reserve_bytes}
