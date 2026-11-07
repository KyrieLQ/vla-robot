"""Read-only RAM limits for the user-requested fixed-adapter Flow experiment."""
from pathlib import Path

def host_available():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):return int(line.split()[1])*1024
    raise RuntimeError('MemAvailable unavailable')

def cgroup_remaining():
    remaining=2**63-1
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        _,controllers,relative=line.split(':',2)
        if 'memory' in controllers.split(','):
            root=Path('/sys/fs/cgroup/memory');p=root/relative.lstrip('/')
            while True:
                limit=p/'memory.limit_in_bytes';usage=p/'memory.usage_in_bytes'
                if limit.exists() and usage.exists():
                    cap=int(limit.read_text())
                    if cap<2**60:remaining=min(remaining,cap-int(usage.read_text()))
                if p==root:break
                if root not in p.parents:break
                p=p.parent
    return remaining

def current_rss():
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):return int(line.split()[1])*1024
    return 0

def admission(resources):
    available=host_available();group=cgroup_remaining();need=resources['ram_cache_bytes']+resources['ram_overhead_bytes']
    ready=available>=need+resources['host_memory_reserve_bytes'] and group>=need
    return dict(status='RAM_READY' if ready else 'WAITING_FOR_HOST_MEMORY',host_available_bytes=available,cgroup_remaining_bytes=group,cache_bytes=resources['ram_cache_bytes'],host_reserve_bytes=resources['host_memory_reserve_bytes'],startup_required_host_bytes=need+resources['host_memory_reserve_bytes'])

def guard(resources):
    if host_available()<resources['host_memory_reserve_bytes']:raise RuntimeError('HOST_MEMORY_RESERVE_REACHED')
    if current_rss()>resources['max_process_rss_bytes']:raise RuntimeError('PROCESS_RSS_LIMIT_REACHED')
    if cgroup_remaining()<2*2**30:raise RuntimeError('CGROUP_MEMORY_HEADROOM_LOW')
