import copy,json,time
from pathlib import Path
from gpu_run_shared_v1 import eligible,candidates
R=Path(__file__).resolve().parents[1]
def main():
 start=time.monotonic();busy=dict(index=3,uuid='GPU-test3',free_mib=8192,total_mib=24576,used_mib=16384,utilization=100,compute_pids=[123])
 assert eligible(busy,4096)
 assert not eligible(dict(busy,free_mib=4095),4096)
 assert eligible(dict(busy,free_mib=4096),4096)
 assert candidates([busy],[busy],{3},4096)==[busy]
 assert candidates([busy],[busy],{0},4096)==[]
 assert candidates([busy],[dict(busy,free_mib=2048)],{3},4096)==[]
 assert candidates([dict(busy,free_mib=2048)],[busy],{3},4096)==[]
 richer=dict(busy,index=0,uuid='GPU-test0',free_mib=16000)
 assert candidates([busy,richer],[busy,richer],{0,3},4096)[0]['index']==0
 result=dict(status='PASS',checks=['existing_compute_and100percent_utilization_not_veto','memory_boundary4096MiB','both_samples_must_fit','allowed_device_filter','prefer_more_free_memory'],elapsed_seconds=time.monotonic()-start,new_gpu_compute=0,new_simulation_episodes=0)
 (R/'reports/gpu_shared_cpu_contract_20261029.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
