"""Actual checkpoint generation, server/client metadata and unchanged reset/queue contract."""
import argparse,ast,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from mini_vla.flow_time_film_v1 import TimeFiLMFlowHead
from mini_vla.vlm_action_heads_v1 import HeadConfig
from mini_vla.policy_m9_v1 import M9Policy
from mini_vla.flow_time_film_metadata_v1 import validate_metadata
from mini_vla.pipeline_flow_time_film_parallel_dev50_v1 import partition_tasks
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 start=time.monotonic();torch.set_num_threads(2);run=R/'results/mini_vla_m7_flow_time_film_train_20261104_run1';ck=torch.load(run/'training/best.pt',map_location='cpu',weights_only=False)
 assert ck['epoch']==50 and ck['updates']==94950 and ck['time_conditioning']=='post_block_affine_zero_init_v1'
 p=dict(checkpoint=str((run/'training/best.pt').relative_to(R)),checkpoint_sha256=sha(run/'training/best.pt'),selected_epoch=50,selected_updates=94950,decoder_layers=2,head_parameters=763335,time_conditioning=ck['time_conditioning'],backbone_checkpoint_sha256=ck['frontend_checkpoint_sha256']);p['arms']={'flow':{'checkpoint_sha256':p['checkpoint_sha256']}}
 policy_source=R/'mini_vla/policy_flow_time_film_v1.py';main_node=next(n for n in ast.parse(policy_source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='main');assignment=next(n for n in main_node.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='metadata' for t in n.targets));scope=dict(p=p,ck=ck,Path=Path,sha=sha,__file__=str(policy_source));exec(compile(ast.Module(body=[assignment],type_ignores=[]),str(policy_source),'exec'),scope);metadata=scope['metadata'];validate_metadata(metadata,p)
 fields=['model','time_conditioning','head_parameters','decoder_layers','arm','epoch','updates','checkpoint_sha256','backbone_checkpoint_sha256','action_horizon','euler_steps','noise_seed_rule']
 for key in fields:
  bad=dict(metadata);bad[key]='wrong'
  try:validate_metadata(bad,p)
  except AssertionError:pass
  else:raise AssertionError(key)
 head=TimeFiLMFlowHead(HeadConfig(**ck['model_config']));head.load_state_dict(ck['state_dict'],strict=True);head.eval();assert sum(v.numel() for v in head.parameters())==763335
 bits=np.load(run/'gate/input_000.npy')
 class Front:
  def encode(self,records):return [dict(bits=bits,input_sha256='cpu_contract')]
 policy=M9Policy(Front(),head,ck['preprocessing']['normalization']);obs={'observation/image':np.zeros((256,256,3),np.uint8),'observation/wrist_image':np.zeros((256,256,3),np.uint8),'observation/state':np.array([.4,.1,.9,0,0,1.57,.02,-.02],np.float32),'prompt':'cpu contract','sampling/task_id':2,'sampling/initial_state_index':42,'sampling/call_index':3};first=policy.infer(obs);assert np.isfinite(first['actions']).all()
 for count in [2,3,4]:
  parts=partition_tasks(count);pairs=[(t,i) for part in parts for t in part for i in [2,12,22,32,42]];assert len(pairs)==len(set(pairs))==50
  torch.randn(71*count);again=policy.infer(obs);np.testing.assert_array_equal(again['actions'],first['actions']);assert again['noise_seed']==first['noise_seed']
 original=(R/'mini_vla/libero_eval_flow_depth2_parallel_v1.py').read_text();current=(R/'mini_vla/libero_eval_flow_time_film_parallel_v1.py').read_text();assert current==original.replace('flow_depth2_metadata_v1','flow_time_film_metadata_v1')
 for f in ['mini_vla/parallel_flow_time_film_eval_worker_v1.py','mini_vla/pipeline_flow_time_film_parallel_dev50_v1.py','mini_vla/check_flow_time_film_online_v1.py','scripts/mini_vla_flow_time_film_eval_supervisor_v1.py','scripts/review_flow_time_film_dev50_v1.py']:ast.parse((R/f).read_text())
 result=dict(status='PASS',actual_policy_metadata=metadata,test_plan_identity=p,rejected_wrong_fields=fields,actual_checkpoint_actions_finite=True,partition2to4_exact50=True,private_rng_worker_independent=True,reset_queue_rollout_source_unchanged=True,new_optimizer_updates=0,new_policy_episodes=0,elapsed_seconds=time.monotonic()-start)
 (R/'reports/flow_time_film_eval_cpu_contract_20261104.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
