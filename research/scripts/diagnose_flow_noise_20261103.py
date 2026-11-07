"""Followup CPU-only fixed-head sampling variance diagnosis, no learning."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from mini_vla.vlm_action_heads_v1 import FlowHead,HeadConfig
R=Path(__file__).resolve().parents[1]
O=R/'reports/flow_diagnosis_20261103'
RUN=R/'results/mini_vla_m7_fixed_lora_flow_20261029_run1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

@torch.inference_mode()
def main():
 t0=time.monotonic();torch.set_num_threads(2);torch.set_num_interop_threads(1)
 plan=json.loads((O/'followup_plan.json').read_text());p=json.loads((RUN/'plan.json').read_text())
 c=torch.load(RUN/'training/best.pt',map_location='cpu',weights_only=False)
 f=FlowHead(HeadConfig(**c['model_config'])).eval().requires_grad_(False);f.load_state_dict(c['state_dict'])
 samples=json.loads((R/p['sample_manifest']).read_text())['samples'];gate=json.loads((RUN/'frontend_gate.json').read_text())['rows']
 ix={e['episode_id']:e for e in json.loads((R/p['cache']/'train/index.json').read_text())}
 nums={k:np.load(R/p['numeric_cache']/'train'/(k+'.npy'),mmap_mode='r') for k in ['actions','state']}
 L=max(ix[s['episode_id']]['tokens'] for s in samples)
 hidden=torch.zeros(120,L,960);hm=torch.zeros(120,L,dtype=torch.bool);ys=[];masks=[];ss=[]
 for i,(s,g) in enumerate(zip(samples,gate)):
  assert (s['episode_id'],s['frame_index'])==(g['episode_id'],g['frame_index'])
  path=RUN/'gate'/g['file'];assert sha(path)==g['file_sha256']
  a=torch.from_numpy(np.load(path)).view(torch.bfloat16).float();hidden[i,:len(a)]=a;hm[i,:len(a)]=True
  e=ix[s['episode_id']];idx=e['start']+s['frame_index'];ids=idx+np.arange(10)
  ys.append(nums['actions'][np.minimum(ids,e['end']-1),0]);ss.append(nums['state'][idx]);masks.append(ids<e['end'])
 y=torch.tensor(np.array(ys));mask=torch.tensor(np.array(masks));state=torch.tensor(np.array(ss))
 perm=torch.randperm(120,generator=torch.Generator().manual_seed(123))
 memory={'base':f.memory(hidden=hidden,hidden_mask=hm,state=state),
         'visual':f.memory(hidden=hidden[perm],hidden_mask=hm[perm],state=state),
         'state':f.memory(hidden=hidden,hidden_mask=hm,state=state[perm])}
 def generate(z,steps,mm,pp,fixed_time=False):
  x=z.clone()
  for i in range(steps):
   x=x+f.velocity(memory=mm,memory_padding=pp,noisy_action=x,time=torch.full((120,),0. if fixed_time else i/steps))/steps
  return x
 result={};arrays={}
 for variant in plan['variants']:
  print(variant,flush=True)
  steps=40 if variant=='Euler40' else 5 if variant=='Euler5' else 10
  mm,pp=memory['visual' if variant.startswith('shuffle_visual') else 'state' if variant.startswith('shuffle_state') else 'base']
  hook=f.time_mlp.register_forward_hook(lambda mod,args,out:torch.zeros_like(out)) if variant=='zero_time_Euler10' else None
  predictions=[]
  for seed in plan['seeds']:
   z=torch.randn((120,10,7),generator=torch.Generator().manual_seed(seed))
   predictions.append(generate(z,steps,mm,pp,variant=='constant_t0_Euler10'))
  if hook:hook.remove()
  pred=torch.stack(predictions);arrays[variant]=pred.numpy()
  stats={}
  for name,sel in [('all',torch.ones(120,dtype=torch.bool)),('full_window',mask.all(1)),('tail',~mask.all(1))]:
   x=pred[:,sel,0];target=y[sel,0];mean=x.mean(0)
   stats[name]=dict(n=int(sel.sum()),per_seed_mse=(x-target).square().mean((1,2)).tolist(),sample_mse=float((x-target).square().mean()),mean_mse=float((mean-target).square().mean()),variance=float(x.var(0,unbiased=False).mean()),mean_per_dim=(mean-target).square().mean(0).tolist(),variance_per_dim=x.var(0,unbiased=False).mean(0).tolist())
   assert abs(stats[name]['sample_mse']-stats[name]['mean_mse']-stats[name]['variance'])<1e-6
  norm=c['preprocessing']['normalization']['actions']
  grip=pred[:,:,0,6]*norm['scale'][6]+norm['mean'][6]
  truth=y[:,0,6]*norm['scale'][6]+norm['mean'][6]
  stats['gripper']=dict(any_seed_sign_disagreement=float(((grip>=0).any(0)&(grip<0).any(0)).float().mean()),average_sign_accuracy=float(((grip>=0)==(truth>=0)).float().mean()))
  result[variant]=stats
 np.savez_compressed(O/'probe_predictions.npz',target=y.numpy(),mask=mask.numpy(),**arrays)
 report=dict(scope='120 TRAIN gate inputs; 40 tasks,40 tails; not a validation score or deployment selection',
             plan_sha256=sha(O/'followup_plan.json'),script_sha256=sha(Path(__file__)),flow_sha256=sha(RUN/'training/best.pt'),
             variants=result,elapsed_seconds=time.monotonic()-t0,new_optimizer_updates=0,new_simulation_episodes=0)
 (O/'noise_results.json').write_text(json.dumps(report,indent=2)+'\n')
 print('COMPLETE',report['elapsed_seconds'],flush=True)
if __name__=='__main__':main()
