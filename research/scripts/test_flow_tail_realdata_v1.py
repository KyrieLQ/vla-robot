"""Read existing adapted features and actual numeric windows; zero training updates."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from mini_vla.vlm_action_heads_v1 import FlowHead,HeadConfig,flow_loss
from mini_vla.flow_tail_consistent_v1 import flow_loss_tail_consistent
R=Path(__file__).resolve().parents[1]
def main():
 start=time.monotonic();torch.set_num_threads(2);torch.manual_seed(42)
 run=R/'results/mini_vla_m7_fixed_lora_flow_20261029_run1';p=json.loads((run/'plan.json').read_text())
 head=FlowHead(HeadConfig(**p['head_config']));old=torch.load(R/p['old_flow_initial'],map_location='cpu',weights_only=False)
 assert all(torch.equal(v,old['state_dict'][k]) for k,v in head.state_dict().items())
 head.eval();samples=json.loads((R/p['sample_manifest']).read_text())['samples'];gate=json.loads((run/'frontend_gate.json').read_text())['rows']
 index={e['episode_id']:e for e in json.loads((R/p['cache']/'train/index.json').read_text())};actions=np.load(R/p['numeric_cache']/'train/actions.npy',mmap_mode='r');states=np.load(R/p['numeric_cache']/'train/state.npy',mmap_mode='r')
 L=max(index[s['episode_id']]['tokens'] for s in samples);hidden=torch.zeros(120,L,960);hm=torch.zeros(120,L,dtype=torch.bool);ys=[];ss=[];masks=[]
 for i,(s,g) in enumerate(zip(samples,gate)):
  f=run/'gate'/g['file'];assert hashlib.sha256(f.read_bytes()).hexdigest()==g['file_sha256'];a=torch.from_numpy(np.load(f)).view(torch.bfloat16).float();hidden[i,:len(a)]=a;hm[i,:len(a)]=True
  e=index[s['episode_id']];ix=e['start']+s['frame_index'];asked=ix+np.arange(10);mask=asked<e['end'];y=np.array(actions[np.minimum(asked,e['end']-1),0]);assert mask[0]
  if not mask.all():np.testing.assert_array_equal(y[~mask],np.repeat(y[int(mask.sum())-1][None],int((~mask).sum()),axis=0))
  ys.append(y);ss.append(states[ix]);masks.append(mask)
 target=torch.tensor(np.array(ys));mask=torch.tensor(np.array(masks));state=torch.tensor(np.array(ss));full=mask.all(1)
 obs=dict(hidden=hidden[full],hidden_mask=hm[full],state=state[full]);noise=torch.randn_like(target);tau=torch.rand(120)
 oldloss=flow_loss(head,obs,target[full],mask[full],noise[full],tau[full]);g1=torch.autograd.grad(oldloss,tuple(head.parameters()))
 newloss=flow_loss_tail_consistent(head,obs,target[full],mask[full],noise[full],tau[full]);g2=torch.autograd.grad(newloss,tuple(head.parameters()))
 torch.testing.assert_close(oldloss,newloss,atol=1e-6,rtol=1e-6)
 maximum=0.
 for a,b in zip(g1,g2):
  maximum=max(maximum,float((a-b).abs().max()));torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
 loss=flow_loss_tail_consistent(head,dict(hidden=hidden,hidden_mask=hm,state=state),target,mask,noise,tau)
 grads=torch.autograd.grad(loss,tuple(head.parameters()));assert all(torch.isfinite(x).all() for x in grads)
 counts={}
 for part in ['train','validation']:
  entries=json.loads((R/p['cache']/part/'index.json').read_text());frames=sum(e['end']-e['start'] for e in entries);valid=sum(sum(min(10,e['end']-i) for i in range(e['start'],e['end'])) for e in entries);counts[part]=dict(episodes=len(entries),frames=frames,valid_targets=valid)
 assert counts['train']==dict(episodes=1506,frames=243000,valid_targets=2362230)
 assert counts['validation']==dict(episodes=187,frames=30465,valid_targets=296235)
 result=dict(status='PASS',existing_train_inputs=120,full_windows=int(full.sum()),tail_windows=int((~full).sum()),initial_weights_exact_original=True,repeat_last_targets_verified=True,full_window_loss_difference=float((oldloss-newloss).abs()),full_window_max_gradient_difference=maximum,all_tail_gradients_finite=True,counts=counts,new_optimizer_updates=0,new_simulation_episodes=0,elapsed_seconds=time.monotonic()-start)
 (R/'reports/flow_tail_realdata_contract_20261103.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
