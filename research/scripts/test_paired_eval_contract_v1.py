"""CPU tests for exported policy normalization, exact reset reuse and journal replay."""
import ast,hashlib,json,tempfile,time
from pathlib import Path
import numpy as np
import torch
from mini_vla.paired_eval_journal_v1 import append,load
from mini_vla.policy_m8_v1 import M8Policy
from mini_vla.vlm_action_heads_v1 import RegressionHead,HeadConfig
R=Path(__file__).resolve().parents[1]
def main():
 started=time.monotonic();torch.set_num_threads(2);checks={}
 for f in ['mini_vla/policy_paired_v1.py','mini_vla/libero_eval_paired_v1.py','mini_vla/paired_eval_journal_v1.py','mini_vla/pipeline_paired_dev50_v1.py','mini_vla/analyze_paired_dev50_v1.py','scripts/mini_vla_paired_eval_supervisor_v1.py']:ast.parse((R/f).read_text())
 old=(R/'mini_vla/libero_eval_m8_v1.py').read_text();new=(R/'mini_vla/libero_eval_paired_v1.py').read_text()
 assert old[old.index('    def reset('):old.index("    if a.variant=='preflight':")]==new[new.index('    def reset('):new.index("    if a.variant=='preflight':")]
 checks['canonical_reset_exact_source_reuse']=True
 with tempfile.TemporaryDirectory(dir=R/'tmp') as tmp:
  f=Path(tmp)/'journal.jsonl';chunks={}
  for step in range(51):
   if step%5==0:
    a=np.full((10,7),step/1000,np.float32);chunks[step]=a;append(f,dict(kind='call',step=step,actions=a.tolist(),normalized_actions=a.tolist()))
   a=chunks[step//5*5][step%5];append(f,dict(kind='step',step=step,action=a.tolist(),done=False))
  calls,steps=load(f);assert len(calls)==11 and len(steps)==51
  remaining=np.asarray(calls[50]['actions'],np.float32)[1:5];np.testing.assert_array_equal(remaining,chunks[50][1:5]);assert len(remaining)==4
  append(f,dict(kind='step',step=51,action=chunks[50][1].tolist(),done=True));calls,steps=load(f);assert len(steps)==52 and steps[-1]['done']
  append(f,dict(kind='step',step=52,action=chunks[50][2].tolist(),done=False))
  try:load(f);raise RuntimeError('post-terminal step accepted')
  except AssertionError:pass
 checks['51step_prefix_retains_pending4_actions_and_terminal_guard']=True
 ck=torch.load(R/'results/mini_vla_m6_paired_20261028_run1/frozen/endpoint.pt',map_location='cpu',weights_only=False)
 head=RegressionHead(HeadConfig(**ck['model_config']));head.load_state_dict(ck['state_dict']);h=torch.randn(12,960).to(torch.bfloat16);bits=h.view(torch.uint16).numpy()
 class Front:
  def encode(self,records):return [dict(bits=bits,input_sha256='test_input')]
 policy=M8Policy(Front(),head,ck['preprocessing']['normalization']);state=np.array([.4,.1,.9,0,0,1.57,.02,-.02],np.float32);im=np.zeros((256,256,3),np.uint8)
 r=policy.infer({'observation/image':im,'observation/wrist_image':im,'observation/state':state,'prompt':'fixed instruction'})
 stats=policy.normalization['actions'];expected=np.clip(r['normalized_actions']*np.asarray(stats['scale'],np.float32)+np.asarray(stats['mean'],np.float32),-1,1);np.testing.assert_allclose(r['actions'],expected,atol=1e-6,rtol=0)
 assert r['actions'].shape==(10,7) and np.isfinite(r['actions']).all() and r['hidden_sha256']==hashlib.sha256(bits.tobytes()).hexdigest();checks['real_exported_head_state_and_action_adapter_CPU']=True
 result=dict(status='PASS',checks=checks,elapsed_seconds=time.monotonic()-started,new_gpu_compute=0,new_simulation_episodes=0,limitation='RealGPU currentfeature parity and positive physical replay execute before any new policy rollout')
 (R/'reports/paired_eval_cpu_contract_20261029.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
