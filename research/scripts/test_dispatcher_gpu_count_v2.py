"""CPU only: required multi-GPU groups must survive both idle samples."""
from pathlib import Path
import json,time
from unittest.mock import patch
import mini_vla_dispatcher_v2 as d
t=time.monotonic()
def gpu(i,busy=False):return dict(index=i,uuid='GPU-test-'+str(i),compute_pids=[123] if busy else [],used_mib=1000 if busy else 13,free_mib=23576 if busy else 24563,total_mib=24576,utilization=0)
def run(first,second,count=2,allowed=range(8)):
 with patch.object(d,'sample',side_effect=[first,second]),patch.object(d.time,'sleep'):
  return d.ready_device(dict(resources=dict(allowed_physical_gpus=list(allowed),simultaneous_gpus=count)))
assert run([gpu(6)],[gpu(6)])==[]
assert run([gpu(6),gpu(7)],[gpu(6),gpu(7)])==[6,7]
assert run([gpu(6),gpu(7)],[gpu(6),gpu(7,True)])==[]
assert run([gpu(6),gpu(7,True)],[gpu(6),gpu(7)])==[]
assert run([gpu(6),gpu(7)],[gpu(6),gpu(7)],allowed=[6])==[]
assert run([gpu(6)],[gpu(6)],count=1)==[6]
reserved=gpu(6);reserved.update(used_mib=1000,free_mib=23576)
assert run([reserved,gpu(7)],[reserved,gpu(7)])==[]
result=dict(status='PASS',single_idle_rejected_for_two_gpu_plan=True,two_sample_group_required=True,busy_memory_not_idle=True,allowed_device_scope_preserved=True,single_gpu_plan_still_supported=True,new_gpu_operations=0,new_simulation_episodes=0,elapsed_seconds=time.monotonic()-t)
Path('reports/dispatcher_gpu_count_v2_contract_20261021.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
