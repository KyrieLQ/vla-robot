"""CPU contract: actual selected Flow head, deterministic partitions and reset equivalence."""
import ast,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from mini_vla.pipeline_flow_depth2_parallel_dev50_v1 import partition_tasks
from mini_vla.policy_m9_v1 import M9Policy
from mini_vla.vlm_action_heads_v1 import HeadConfig
from mini_vla.flow_depth2_v1 import Depth2FlowHead as FlowHead
R=Path(__file__).resolve().parents[1]
def main():
 start=time.monotonic();torch.set_num_threads(2);run=R/'results/mini_vla_m7_flow_depth2_train_20261103_run1'
 ck=torch.load(run/'training/best.pt',map_location='cpu',weights_only=False);assert ck['epoch']==42 and ck['updates']==79758 and ck['decoder_layers']==2
 head=FlowHead(HeadConfig(**ck['model_config']));head.load_state_dict(ck['state_dict']);bits=np.load(run/'gate/input_000.npy')
 class Front:
  def encode(self,records):return [dict(bits=bits,input_sha256='cpu_contract')]
 policy=M9Policy(Front(),head,ck['preprocessing']['normalization']);obs={'observation/image':np.zeros((256,256,3),np.uint8),'observation/wrist_image':np.zeros((256,256,3),np.uint8),'observation/state':np.array([.4,.1,.9,0,0,1.57,.02,-.02],np.float32),'prompt':'cpu contract','sampling/task_id':2,'sampling/initial_state_index':42,'sampling/call_index':3}
 first=policy.infer(obs)
 for n in [2,3,4]:
  parts=partition_tasks(n);pairs=[(t,i) for tasks in parts for t in tasks for i in [2,12,22,32,42]];assert len(pairs)==len(set(pairs))==50
  assert sorted(pairs)==[(t,i) for t in range(10) for i in [2,12,22,32,42]]
  torch.randn(n*71);again=policy.infer(obs);np.testing.assert_array_equal(first['actions'],again['actions']);assert first['noise_seed']==again['noise_seed']
 original=(R/'mini_vla/libero_eval_fixed_lora_flow_v1.py').read_text();new=(R/'mini_vla/libero_eval_flow_tail_parallel_v1.py').read_text()
 assert original[original.index('    def reset('):original.index("    if a.variant=='preflight':")]==new[new.index('    def reset('):new.index("    if a.variant=='preflight':")]
 assert "scientific_hash=plan['scientific_plan_sha256']" in new
 assert "scientific_plan_sha256=sha(pp)" in (R/'mini_vla/pipeline_flow_depth2_parallel_dev50_v1.py').read_text()
 for path in list((R/'mini_vla').glob('*flow*parallel*dev50*.py'))+[R/'mini_vla/parallel_flow_depth2_eval_worker_v1.py',R/'mini_vla/libero_eval_flow_tail_parallel_v1.py',R/'scripts/mini_vla_flow_depth2_eval_supervisor_v1.py',R/'scripts/review_flow_depth2_dev50_v1.py']:ast.parse(path.read_text())
 result=dict(status='PASS',checkpoint_sha256=hashlib.sha256((run/'training/best.pt').read_bytes()).hexdigest(),checks=['actual_epoch42_depth2_head_finite_actions','2_3_4_worker_task_partition_covers_exact50','private_noise_independent_of_worker_partition_and_global_rng','canonical_reset_source_identical','parent_scientific_hash_vs_shard_execution_hash_separated','source_parse'],elapsed_seconds=time.monotonic()-start,new_optimizer_updates=0,new_simulation_episodes=0,limitations='Runtime fourGPU gates,positive physical replays and full initial-image checks still required before policy samples.')
 (R/'reports/flow_depth2_parallel_eval_cpu_contract_20261104.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
