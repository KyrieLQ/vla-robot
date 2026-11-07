"""Bounded pipe, deterministic assignment and four-GPU admission contracts."""
import ast,hashlib,json,os,threading,time
from pathlib import Path
import numpy as np
from mini_vla.flow_feature_worker_v1 import send,bits_sha
from mini_vla.flow_parallel_extract_v1 import receive,partitions
from gpu_run_shared_v2 import requirements,candidates
R=Path(__file__).resolve().parents[1]
def main():
 started=time.monotonic();approval='configs/mini_vla_shared_gpu_20261103_APPROVED.json';a=json.loads((R/approval).read_text())
 p=dict(thread=a['thread'],resources=dict(gpu_sharing_approval=approval,gpu_sharing_approval_sha256=hashlib.sha256((R/approval).read_bytes()).hexdigest(),allowed_physical_gpus=list(range(8)),minimum_free_gpu_mib=6144,simultaneous_gpus=4,minimum_gpu_count=2,multi_gpu_assessment={'considered_counts':[1,2,3,4]}))
 allowed,minimum,count=requirements(p);assert count==4
 g=[dict(uuid=str(i),index=i,free_mib=8000+i,utilization=90) for i in range(8)];assert len(candidates(g,g,allowed,minimum))==8
 p['resources']['simultaneous_gpus']=5
 try:requirements(p)
 except AssertionError:pass
 else:raise AssertionError('More than4 admitted')
 entries=[dict(episode_id=i,start=0,end=10+i,tokens=150+i%5) for i in range(37)]
 for n in [2,3,4]:
  bins=partitions(entries,n);assert sorted(x for b in bins for x in b)==list(range(37));assert bins==partitions(entries,n)
 data=np.random.default_rng(42).integers(0,65536,size=(16,128,960),dtype=np.uint16)
 rd,wd=os.pipe();reader=os.fdopen(rd,'rb',buffering=0);writer=os.fdopen(wd,'wb',buffering=0)
 def produce():
  try:send(writer,dict(kind='episode',shape=list(data.shape),sha=bits_sha(data)),data)
  finally:writer.close()
 thread=threading.Thread(target=produce);thread.start();header,payload=receive(reader,max_bytes=data.nbytes);thread.join(timeout=10);assert not thread.is_alive();reader.close()
 actual=np.frombuffer(payload,np.uint16).reshape(header['shape']);np.testing.assert_array_equal(actual,data);assert bits_sha(actual)==header['sha']
 for f in ['mini_vla/flow_feature_worker_v1.py','mini_vla/flow_parallel_extract_v1.py','mini_vla/train_flow_tail_parallel_v1.py','scripts/gpu_run_shared_v2.py','scripts/mini_vla_dispatcher_v7.py']:ast.parse((R/f).read_text())
 result=dict(status='PASS',checks=['max4_gpu_policy','busy_gpu_with_memory_eligible','reject5_gpus','deterministic2_3_4_episode_partition_complete_nonoverlap','large_raw_uint16_pipe_bitwise_roundtrip','source_parse'],elapsed_seconds=time.monotonic()-started,new_gpu_compute=0,new_optimizer_updates=0)
 (R/'reports/flow_parallel_cpu_contract_20261103.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
