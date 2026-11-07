"""Actual old initialization and independent new-block gradient contract, no optimizer step."""
import json,time
from pathlib import Path
import torch
from mini_vla.flow_depth2_v1 import Depth2FlowHead,verify_common_initial
from mini_vla.vlm_action_heads_v1 import HeadConfig
from mini_vla.flow_tail_consistent_v1 import flow_loss_tail_consistent
R=Path(__file__).resolve().parents[1]
def main():
 started=time.monotonic();torch.set_num_threads(2);torch.manual_seed(42)
 old=torch.load(R/'results/mini_vla_m9_flow_train_20261021_run1/training/initial.pt',map_location='cpu',weights_only=False)
 model=Depth2FlowHead(HeadConfig(**old['model_config']));verify_common_initial(model,old['state_dict']);parameters=sum(p.numel() for p in model.parameters());assert parameters==697287
 obs=dict(hidden=torch.randn(3,17,960),hidden_mask=torch.ones(3,17,dtype=torch.bool),state=torch.randn(3,8));target=torch.randn(3,10,7);mask=torch.ones(3,10,dtype=torch.bool);mask[1,6:]=False;target[1,6:]=target[1,5]
 noise=torch.randn_like(target);tau=torch.tensor([0.,.35,.9]);loss=flow_loss_tail_consistent(model,obs,target,mask,noise,tau);loss.backward()
 norms=[]
 for layer in model.decoder.layers:
  grads=[p.grad for p in layer.parameters()];assert all(g is not None and torch.isfinite(g).all() for g in grads);norms.append(float(sum(g.square().sum() for g in grads).sqrt()));assert norms[-1]>0
 model.eval();sample=model.sample(**obs,noise=noise);assert sample.shape==(3,10,7) and torch.isfinite(sample).all()
 roundtrip=Depth2FlowHead(model.config);roundtrip.load_state_dict(model.state_dict());roundtrip.eval();torch.testing.assert_close(sample,roundtrip.sample(**obs,noise=noise),atol=0,rtol=0)
 report=dict(status='PASS',parameters=parameters,decoder_layers=2,common_parameters_exact_original=True,additional_block='Independent deep copy of original initial decoder block; equal initial values, no parameter tying or extra random draws',layer_gradient_norms=norms,checks=['finite_velocity_loss','both_blocks_receive_gradients','K10D7_generation','state_dict_strict_roundtrip_exact'],new_optimizer_updates=0,new_simulation_episodes=0,elapsed_seconds=time.monotonic()-started)
 (R/'reports/flow_depth2_cpu_contract_20261103.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
