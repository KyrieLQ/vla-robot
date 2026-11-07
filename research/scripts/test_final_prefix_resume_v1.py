"""CPU test: resume both a mid-block interruption and a logged-but-unexecuted RPC."""
from pathlib import Path
from types import SimpleNamespace
import json,tempfile,time
import numpy as np
from mini_vla.final_episode_v2 import run_episode,load_journal,validate_records,inspect_prefix
t=time.monotonic()
class Env:
 def __init__(self):self.n=0;self.sim=SimpleNamespace(data=SimpleNamespace(qpos=np.zeros(2),qvel=np.zeros(2)))
 def step(self,a):self.n+=1;self.sim.data.qpos[:]=self.n;return {},0,self.check_success(),{}
 def check_success(self):return self.n==13
def observation(_):return np.zeros((2,2,3),np.uint8),np.zeros((2,2,3),np.uint8),np.zeros(8),'test'
class Policy:
 def __init__(self,call=0):self.calls=call
 def __call__(self,o):
  self.calls+=1;return dict(actions=np.full((10,7),self.calls/10),server_timing=dict(infer_ms=1.))
with tempfile.TemporaryDirectory() as tmp:
 d=Path(tmp);full=d/'full';p=Policy()
 with full.open('x') as f:run_episode(Env(),{},p,observation,20,f)
 records=load_journal(full)
 for steps,include_next_call in [(6,False),(5,True)]:
  prefix=[]
  for r in records:
   if r['kind']=='step' and r['step']<steps:prefix.append(r)
   if r['kind']=='call' and (r['step']<steps or (include_next_call and r['step']==steps)):prefix.append(r)
  info=inspect_prefix(prefix,20);assert not info['complete'] and info['steps']==steps
  assert len(info['pending_actions'])==(4 if steps==6 else 5)
  env=Env()
  for r in prefix:
   if r['kind']=='step':env.step(r['action'])
  path=d/f'resume{steps}';path.write_text(''.join(json.dumps(x)+'\n' for x in prefix));policy=Policy(info['calls'])
  with path.open('a') as f:summary=run_episode(env,{},policy,observation,20,f,start_step=steps,pending_actions=info['pending_actions'],prior_calls=info['calls'])
  resumed=load_journal(path);assert summary==validate_records(records,20)==validate_records(resumed,20)
  np.testing.assert_array_equal([r['action'] for r in resumed if r['kind']=='step'],[r['action'] for r in records if r['kind']=='step']);assert policy.calls==p.calls==3
 # Real interrupted prefix: step50 was committed, four original commands remain.
 real=load_journal(Path('results/mini_vla_final2000_20261021_run1/worker1/libero_object/task1_init19/trajectory.jsonl'));r=inspect_prefix(real,280);assert r['steps']==51 and r['calls']==11 and len(r['pending_actions'])==4 and not r['complete']
result=dict(status='PASS',same_actions_and_outcome_after_midblock_resume=True,logged_RPC_reused_without_requery=True,real_partial_steps=51,real_pending_commands=4,new_simulation_episodes=0,gpu_operations=0,elapsed_seconds=time.monotonic()-t)
Path('reports/final_prefix_resume_cpu_contract_20261021.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
