import ast,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from mini_vla.policy_m9_v1 import M9Policy
from mini_vla import policy_fixed_lora_flow_v1,check_fixed_lora_flow_online_v1,pipeline_fixed_lora_flow_dev50_v1
assert policy_fixed_lora_flow_v1.M9Policy is M9Policy
from mini_vla.vlm_action_heads_v1 import FlowHead,HeadConfig
R=Path(__file__).resolve().parents[1]
def main():
 start=time.monotonic();torch.set_num_threads(2);run=R/'results/mini_vla_m7_fixed_lora_flow_20261029_run1';ck=torch.load(run/'training/best.pt',map_location='cpu',weights_only=False);head=FlowHead(HeadConfig(**ck['model_config']));head.load_state_dict(ck['state_dict']);bits=np.load(run/'gate/input_000.npy')
 class Front:
  def encode(self,records):return [dict(bits=bits,input_sha256='fixedtest')]
 policy=M9Policy(Front(),head,ck['preprocessing']['normalization']);obs={'observation/image':np.zeros((256,256,3),np.uint8),'observation/wrist_image':np.zeros((256,256,3),np.uint8),'observation/state':np.array([.4,.1,.9,0,0,1.57,.02,-.02],np.float32),'prompt':'fixed instruction','sampling/task_id':2,'sampling/initial_state_index':42,'sampling/call_index':3}
 a=policy.infer(obs);torch.randn(123);b=policy.infer(obs);np.testing.assert_array_equal(a['actions'],b['actions']);assert a['noise_seed']==b['noise_seed']
 seed=int.from_bytes(hashlib.sha256(b'20261020:2:42:3').digest()[:8],'little')%(2**63-1);noise=torch.randn((10,7),generator=torch.Generator().manual_seed(seed),dtype=torch.float32);assert a['noise_sha256']==hashlib.sha256(noise.numpy().tobytes()).hexdigest();assert a['actions'].shape==(10,7) and np.isfinite(a['actions']).all()
 old=(R/'mini_vla/libero_eval_paired_v1.py').read_text();new=(R/'mini_vla/libero_eval_fixed_lora_flow_v1.py').read_text();assert old[old.index('    def reset('):old.index("    if a.variant=='preflight':")]==new[new.index('    def reset('):new.index("    if a.variant=='preflight':")]
 for f in ['mini_vla/policy_fixed_lora_flow_v1.py','mini_vla/check_fixed_lora_flow_online_v1.py','mini_vla/libero_eval_fixed_lora_flow_v1.py','mini_vla/pipeline_fixed_lora_flow_dev50_v1.py','mini_vla/analyze_fixed_lora_flow_dev50_v1.py','scripts/mini_vla_fixed_lora_flow_eval_supervisor_v1.py']:ast.parse((R/f).read_text())
 result=dict(status='PASS',selected_epoch=ck['epoch'],checks=['actual_selected_head_on_saved_adapted_gate_feature','privateCPUnoise_independent_of_globalRNG','seed_and_noisehash_exact','K10D7_finite_commands','unchanged_canonical_reset_source','new_sources_parse'],elapsed_seconds=time.monotonic()-start,new_gpu_compute=0,new_simulation_episodes=0,limitation='LiveVLM/RAMgate identity and positive physical replay execute after GPU admission')
 (R/'reports/fixed_lora_flow_eval_cpu_contract_20261030.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
