"""CPU-only test of H5 queues, terminal journals and refusing incomplete outcomes."""
from pathlib import Path
from types import SimpleNamespace
import tempfile,json,time
import numpy as np
from mini_vla.final_episode_v1 import run_episode,load_journal,validate_records
t=time.monotonic()
class Env:
 def __init__(self,success_at):self.n=0;self.success_at=success_at;self.sim=SimpleNamespace(data=SimpleNamespace(qpos=np.zeros(2),qvel=np.zeros(2)))
 def step(self,a):self.n+=1;self.sim.data.qpos[:]=self.n;return {},0,self.check_success(),{}
 def check_success(self):return self.n==self.success_at
def infer(o):
 assert set(o)=={'observation/image','observation/wrist_image','observation/state','prompt'}
 return dict(actions=np.repeat((np.arange(10)/10)[:,None],7,axis=1),server_timing=dict(infer_ms=1.))
def observation(_):return np.zeros((256,256,3),np.uint8),np.zeros((256,256,3),np.uint8),np.zeros(8),'pick bowl'
with tempfile.TemporaryDirectory() as d:
 for success_at,limit in [(7,12),(999,12)]:
  p=Path(d)/str(success_at)
  with p.open('x') as f:result=run_episode(Env(success_at),{},infer,observation,limit,f)
  records=load_journal(p);assert result==validate_records(records,limit)
  assert result['task_success']==(success_at==7)
  calls=[r['step'] for r in records if r['kind']=='call'];assert calls==([0,5] if success_at==7 else [0,5,10])
  steps=[r for r in records if r['kind']=='step'];np.testing.assert_allclose([r['action'][0] for r in steps],[(i%5)/10 for i in range(len(steps))])
  try:validate_records(records[:4],limit)
  except AssertionError:pass
  else:raise AssertionError('Incomplete trajectory accepted as a complete case')
  records[-1]['action'][0]=.99
  try:validate_records(records,limit)
  except AssertionError:pass
  else:raise AssertionError('Altered executed action accepted')
s=json.loads(Path('configs/mini_vla_final2000_scientific_20261021_v2.json').read_text());assert s['worker_suites']==[['libero_spatial','libero_10'],['libero_object','libero_goal']];assert s['initial_state_indices']==list(range(50)) and s['environment_seed']==7 and s['wait_steps']==10
out=dict(status='PASS',queue_K10_H5_verified=True,original_success_stop_and_limit=True,durable_terminal_journal=True,incomplete_outcomes_rejected=True,changed_actions_rejected=True,whole_suite_partition_verified=True,elapsed_seconds=time.monotonic()-t,gpu_operations=0,final_scenarios_accessed=False)
Path('reports/final_episode_cpu_contract_20261021.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
