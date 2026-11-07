"""Regression test of real policy metadata construction and evaluator acceptance."""
import ast,copy,hashlib,json
from pathlib import Path
import torch
from mini_vla.flow_depth2_metadata_v1 import validate_metadata
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=json.loads((R/'configs/mini_vla_m7_flow_depth2_dev50_20261104_v1.json').read_text())
 ck=torch.load(R/p['checkpoint'],map_location='cpu',weights_only=False)
 policy=R/'mini_vla/policy_flow_depth2_v1.py';tree=ast.parse(policy.read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
 node=next(n for n in main.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='metadata' for t in n.targets))
 scope=dict(ck=ck,p=p,sha=sha,Path=Path,__file__=str(policy));exec(compile(ast.Module(body=[node],type_ignores=[]),str(policy),'exec'),scope)
 metadata=scope['metadata'];validate_metadata(metadata,p)
 assert metadata['updates']==79758 and metadata['epoch']==42
 rejected=[]
 for k in ['model','arm','epoch','updates','checkpoint_sha256','backbone_checkpoint_sha256','action_horizon','euler_steps','noise_seed_rule']:
  broken=dict(metadata);broken[k]='invalid'
  try:validate_metadata(broken,p)
  except AssertionError:rejected.append(k)
  else:raise AssertionError('Mismatch silently accepted: '+k)
 original=(R/'mini_vla/libero_eval_flow_tail_parallel_v1.py').read_text();new=(R/'mini_vla/libero_eval_flow_depth2_parallel_v1.py').read_text()
 assert original[original.index('    def reset('):original.index("    if a.variant=='preflight':")]==new[new.index('    def reset('):new.index("    if a.variant=='preflight':")]
 assert original[original.index('    rows=[];started='):]==new[new.index('    rows=[];started='):]
 oldline=next(x.strip() for x in original.splitlines() if x.strip().startswith("assert metadata['checkpoint_sha256']"))
 from types import SimpleNamespace
 try:exec(oldline,dict(metadata=metadata,selected=p['arms']['flow'],plan=p,a=SimpleNamespace(arm='flow')))
 except AssertionError:pass
 else:raise AssertionError('Old bug not reproduced')
 for f in ['mini_vla/libero_eval_flow_depth2_parallel_v1.py','mini_vla/parallel_flow_depth2_eval_worker_v2.py','mini_vla/pipeline_flow_depth2_parallel_dev50_v2.py','mini_vla/check_flow_depth2_metadata_v1.py','scripts/mini_vla_flow_depth2_eval_supervisor_v2.py']:ast.parse((R/f).read_text())
 result=dict(status='PASS',actual_policy_metadata=metadata,checkpoint_sha256=sha(R/p['checkpoint']),old_model_name_bug_reproduced=True,wrong_fields_rejected=rejected,reset_and_rollout_source_unchanged=True,new_training_updates=0,new_policy_calls=0,new_simulation_episodes=0)
 (R/'reports/flow_depth2_metadata_repair_20261104.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
