"""Aggregate only the owned probe process tree; do not act on other jobs."""
import subprocess
from flow_ram_resources_v1 import host_available

def check_ram(root_pid,resources):
    rows=[tuple(map(int,line.split())) for line in subprocess.check_output(['ps','-e','-o','pid=,ppid=,rss='],text=True).splitlines()]
    owned={root_pid}
    for _ in range(20):
        updated=owned|{pid for pid,parent,rss in rows if parent in owned}
        if updated==owned:break
        owned=updated
    rss=sum(rss*1024 for pid,parent,rss in rows if pid in owned);available=host_available()
    if rss>resources['max_process_rss_bytes']:raise RuntimeError('AGGREGATE_PROBE_RSS_LIMIT')
    if available<resources['host_memory_reserve_bytes']:raise RuntimeError('HOST_RAM_RESERVE_LIMIT')
    return dict(status='PASS',aggregate_owned_rss_bytes=rss,host_available_bytes=available,owned_process_count=len(owned))
