"""Respect kernel memory-node permissions when choosing the native JAX path."""
import json, os
from pathlib import Path

def parse_ranges(value):
    values=set()
    for item in value.strip().split(','):
        if '-' in item:
            low,high=map(int,item.split('-'));values.update(range(low,high+1))
        elif item:values.add(int(item))
    return values

def allowed_memory_nodes():
    line=next(x for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('Mems_allowed_list:'))
    return parse_ranges(line.split(':',1)[1])

def gpu_numa_node(gpu):
    domain,bus,device=gpu['pci_bus_id'].lower().split(':')
    path=Path('/sys/bus/pci/devices')/('%04x:%s:%s'%(int(domain,16),bus,device))/'numa_node'
    return int(path.read_text())

def audit_selected(selection):
    allowed=allowed_memory_nodes();rows=[]
    for gpu in selection:
        node=gpu_numa_node(gpu)
        rows.append(dict(index=gpu['index'],uuid=gpu['uuid'],numa_node=node,memory_node_permitted=node in allowed))
    assert len(rows)==4 and all(row['memory_node_permitted'] for row in rows),rows
    return dict(status='PASS',kernel_allowed_memory_nodes=sorted(allowed),selected=rows,permissions_unchanged=True)

def worker_main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--job',required=True);args,_=parser.parse_known_args()
    selection=json.loads(Path(os.environ['VLA_GPU_SELECTION']).read_text())['selected']
    from .pi05_formal_io_v1 import write
    write(Path(args.job).parent/'numa_admission.json',audit_selected(selection))
    # Backend initialization follows this check; it does not change cpusets.
    from .pi05_formal_worker_v2 import main
    main()

if __name__=='__main__':worker_main()
