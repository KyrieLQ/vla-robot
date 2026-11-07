"""Flow objective, Euler integration, padding isolation, deterministic noise and RNG resume."""
import copy,json,tempfile,time,types
from pathlib import Path
import torch
from mini_vla.vlm_action_heads_v1 import RegressionHead,FlowHead,flow_loss
from mini_vla.train_vlm_flow_v1 import fixed_noise,training_loss,save
R=Path(__file__).resolve().parents[1]
def main():
 start=time.monotonic();torch.set_num_threads(2);torch.manual_seed(42);regression=RegressionHead();torch.manual_seed(42);model=FlowHead()
 for k,v in regression.state_dict().items():torch.testing.assert_close(v,model.state_dict()[k],atol=0,rtol=0)
 assert sum(p.numel() for p in model.parameters())==432711
 obs=dict(hidden=torch.randn(2,12,960).bfloat16(),hidden_mask=torch.ones(2,12,dtype=torch.bool),state=torch.randn(2,8));obs['hidden_mask'][1,8:]=False
 target=torch.randn(2,10,7);mask=torch.ones(2,10,dtype=torch.bool);mask[1,3:]=False;noise=torch.randn_like(target);t=torch.tensor([.2,.8])
 model.eval();base=flow_loss(model,obs,target,mask,noise,t);dirty=target.clone();dirty[~mask]=float('nan');hidden=dict(obs);hidden['hidden']=obs['hidden'].clone();hidden['hidden'][~obs['hidden_mask']]=float('nan')
 torch.testing.assert_close(base,flow_loss(model,hidden,dirty,mask,noise,t),atol=0,rtol=0)
 # Known constant velocity integrates exactly to its endpoint in ten Euler steps.
 analytic=copy.deepcopy(model);velocity=target-noise;calls=[]
 def constant(self,**kw):calls.append(float(kw['time'][0]));return velocity
 analytic.velocity=types.MethodType(constant,analytic)
 with torch.no_grad():prediction=analytic.sample(**obs,noise=noise)
 torch.testing.assert_close(prediction,target,atol=1e-6,rtol=0);assert len(calls)==10
 torch.testing.assert_close(torch.tensor(calls),torch.arange(10)/10,atol=1e-7,rtol=0)
 # A zero-error analytic velocity must give exactly zero masked loss.
 class Exact:
  def __call__(self,**kw):return torch.where(mask[...,None],target-noise,0.)
 assert float(flow_loss(Exact(),obs,target,mask,noise,t))==0
 ids=dict(episode_id=torch.tensor([8,181]),frame_index=torch.tensor([0,130]));before=torch.get_rng_state().clone();a=fixed_noise(ids);assert torch.equal(before,torch.get_rng_state())
 rev={k:v.flip(0) for k,v in ids.items()};torch.testing.assert_close(a,fixed_noise(rev).flip(0),atol=0,rtol=0)
 assert not torch.equal(a[0],a[1])
 model.train();opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
 def step(m,o):
  o.zero_grad(set_to_none=True);loss=training_loss(m,obs,target,mask);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);o.step();return float(loss)
 initial=step(model,opt)
 with tempfile.TemporaryDirectory() as tmp:
  p=Path(tmp)/'resume.pt';save(p,dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=torch.get_rng_state()));step(model,opt);expected=step(model,opt)
  ck=torch.load(p,weights_only=False);restored=FlowHead();restored.load_state_dict(ck['model']);restored.train();o=torch.optim.AdamW(restored.parameters(),lr=3e-4,weight_decay=1e-4);o.load_state_dict(ck['optimizer']);torch.set_rng_state(ck['rng']);step(restored,o);actual=step(restored,o)
  assert actual==expected
  for k,v in model.state_dict().items():torch.testing.assert_close(v,restored.state_dict()[k],atol=0,rtol=0)
 result=dict(status='PASS',head_parameters=432711,shared_M8_seed42_initialization_exact=True,padded_targets_and_hidden_nan_isolated=True,known_velocity_zero_loss=True,ten_Euler_steps_reach_known_endpoint=True,validation_noise_batch_order_independent=True,validation_noise_preserves_global_rng=True,stochastic_optimizer_resume_exact=True,disposable_CPU_synthetic_updates=5,weights_discarded=True,simulation_episodes=0,gpu_operations=0,elapsed_seconds=time.monotonic()-start)
 (R/'reports/m9_flow_training_contract_20261021.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
