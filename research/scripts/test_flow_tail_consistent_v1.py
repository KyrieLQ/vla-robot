"""CPU contract for the next single-variable training arm, no optimization."""
import json,time
from pathlib import Path
import torch
from mini_vla.vlm_action_heads_v1 import FlowHead,HeadConfig,flow_loss
from mini_vla.flow_tail_consistent_v1 import flow_loss_tail_consistent
R=Path(__file__).resolve().parents[1]
def main():
    start=time.monotonic();torch.set_num_threads(2);torch.manual_seed(42)
    model=FlowHead(HeadConfig()).eval()
    obs=dict(hidden=torch.randn(4,9,960),hidden_mask=torch.ones(4,9,dtype=torch.bool),state=torch.randn(4,8))
    target=torch.randn(4,10,7);noise=torch.randn_like(target);time_=torch.tensor([0.,.2,.8,.99]);allvalid=torch.ones(4,10,dtype=torch.bool)
    old=flow_loss(model,obs,target,allvalid,noise,time_);g1=torch.autograd.grad(old,tuple(model.parameters()))
    new=flow_loss_tail_consistent(model,obs,target,allvalid,noise,time_);g2=torch.autograd.grad(new,tuple(model.parameters()))
    torch.testing.assert_close(old,new,atol=1e-6,rtol=1e-6)
    maxgrad=0.
    for a,b in zip(g1,g2):
        maxgrad=max(maxgrad,float((a-b).abs().max()));torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
    mask=allvalid.clone();mask[1,7:]=False;mask[2,1:]=False
    target[1,7:]=target[1,6];target[2,1:]=target[2,0]
    class Spy:
        def __call__(self,**kwargs):
            assert 'action_mask' not in kwargs
            self.x=kwargs['noisy_action'];self.out=torch.zeros_like(target,requires_grad=True)
            return self.out
    spy=Spy();loss=flow_loss_tail_consistent(spy,obs,target,mask,noise,time_);loss.backward()
    assert torch.count_nonzero(spy.out.grad[~mask])==0
    torch.testing.assert_close(spy.x,(1-time_[:,None,None])*noise+time_[:,None,None]*target,atol=0,rtol=0)
    assert torch.count_nonzero(spy.x[~mask])>0
    tail_loss=flow_loss_tail_consistent(model,obs,target,mask,noise,time_)
    grads=torch.autograd.grad(tail_loss,tuple(model.parameters()));assert all(torch.isfinite(g).all() for g in grads)
    expected=((target-noise)[mask]**2).mean();torch.testing.assert_close(loss,expected)
    result=dict(status='PASS',full_window_loss_difference=float((old-new).abs()),full_window_gradient_max_difference=maxgrad,
                checks=['full_window_loss_and_gradient_match_original','future_mask_not_passed_to_model','all10_noisy_tokens_retained','invalid_outputs_zero_loss_gradient','valid_only_loss_denominator','finite_tail_gradients'],
                new_optimizer_updates=0,new_simulation_episodes=0,elapsed_seconds=time.monotonic()-start,
                limitations='Synthetic CPU contract only. Full-data training, sampled current features, initialization and throughput checks are still required before launching this next arm.')
    (R/'reports/flow_tail_consistency_cpu_contract_20261103.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
