"""Same actual initialization, predictions, noise RNG and nonzero modulation gradients."""
import json,time
from pathlib import Path
import numpy as np
import torch
from mini_vla.flow_depth2_v1 import Depth2FlowHead
from mini_vla.flow_time_film_v1 import TimeFiLMFlowHead,verify_common_initial
from mini_vla.vlm_action_heads_v1 import HeadConfig
from mini_vla.flow_tail_consistent_v1 import flow_loss_tail_consistent
from mini_vla.vlm_frontend_v1 import bits_bf16
R=Path(__file__).resolve().parents[1]
def main():
 start=time.monotonic();torch.set_num_threads(2)
 run=R/'results/mini_vla_m7_flow_depth2_train_20261103_run1'
 ck=torch.load(run/'training/initial.pt',map_location='cpu',weights_only=False);config=HeadConfig(**ck['model_config'])
 torch.manual_seed(42);base=Depth2FlowHead(config);base_rng=torch.get_rng_state().clone()
 torch.manual_seed(42);model=TimeFiLMFlowHead(config);assert torch.equal(base_rng,torch.get_rng_state());verify_common_initial(model,ck['state_dict'])
 assert sum(p.numel() for p in model.parameters())==763335
 bits=np.load(run/'gate/input_000.npy');h=bits_bf16(bits)[None].repeat(3,1,1).float();obs=dict(hidden=h,hidden_mask=torch.ones(h.shape[:2],dtype=torch.bool),state=torch.randn(3,8));noise=torch.randn(3,10,7);target=torch.randn_like(noise);target[1,6:]=target[1,5];mask=torch.ones(3,10,dtype=torch.bool);mask[1,6:]=False;tau=torch.tensor([0.,.35,.9])
 base.eval();model.eval()
 with torch.no_grad():
  a=base(**obs,noisy_action=noise,time=tau);b=model(**obs,noisy_action=noise,time=tau);torch.testing.assert_close(a,b,atol=0,rtol=0)
  a=base.sample(**obs,noise=noise);b=model.sample(**obs,noise=noise);torch.testing.assert_close(a,b,atol=0,rtol=0)
 loss=flow_loss_tail_consistent(model,obs,target,mask,noise,tau);loss.backward();grads=[]
 for proj in model.time_modulation:
  n=float(sum(p.grad.square().sum() for p in proj.parameters()).sqrt());assert n>0 and np.isfinite(n);grads.append(n)
 # A deterministic temporary nonzero scale proves the branch affects predictions.
 with torch.no_grad():
  model.time_modulation[0].weight[0,0]=.1
  changed=model.sample(**obs,noise=noise);assert not torch.equal(changed,b)
  model.time_modulation[0].weight.zero_()
 roundtrip=TimeFiLMFlowHead(config);roundtrip.load_state_dict(model.state_dict(),strict=True);roundtrip.eval()
 with torch.no_grad():torch.testing.assert_close(roundtrip.sample(**obs,noise=noise),b,atol=0,rtol=0)
 report=dict(status='PASS',head_parameters=763335,control_parameters=697287,additional_parameters=66048,common_initial_parameters_exact=True,construction_rng_identical=True,initial_velocity_and_Euler10_generation_bitwise_equal=True,time_modulation_gradient_norms=grads,nonzero_modulation_changes_output=True,strict_roundtrip_exact=True,new_optimizer_updates=0,new_simulation_episodes=0,elapsed_seconds=time.monotonic()-start)
 (R/'reports/flow_time_film_cpu_contract_20261104.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
