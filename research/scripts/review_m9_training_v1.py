"""Independent complete M9 checkpoint and all30465 validation frame audit."""
import bisect,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from mini_vla.vlm_action_heads_v1 import FlowHead,HeadConfig
R=Path(__file__).resolve().parents[1];RUN=R/'results/mini_vla_m9_flow_train_20261021_run1'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
 return h.hexdigest()
def main():
 start=time.monotonic();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;assert torch.cuda.device_count()==1
 out=RUN/'review';out.mkdir(exist_ok=True);assert not (out/'independent_acceptance.json').exists();pp=R/'configs/mini_vla_m9_flow_train_20261021_v1.json';p=json.loads(pp.read_text());train=RUN/'training';rep=json.loads((train/'report.json').read_text());history=json.loads((train/'history.json').read_text())
 for f,h in json.loads((RUN/'source_manifest.json').read_text()).items():assert sha(R/f)==h,f
 for f,h in p['input_sha256'].items():assert sha(R/f)==h,f
 assert len(history)==50 and rep['epochs']==50 and rep['updates']==94950
 for epoch,row in enumerate(history,1):
  assert row['epoch']==epoch and row['updates']==epoch*1899 and row['train_frames']==243000 and row['valid_target_steps']==2362230
  assert row['validation']['frames']==30465 and row['validation']['valid_target_steps']==296235 and set(row['validation']['per_task'])=={str(i) for i in range(40)}
 selected=min(history,key=lambda r:r['validation']['normalized_mse']);assert selected['epoch']==rep['best_epoch']==43
 pre=json.loads((R/p['preprocessing']).read_text());initial=best=None;checkpoints={}
 for name in ['initial','best','latest']:
  file=train/(name+'.pt');assert sha(file)==rep['checkpoint_sha256'][name];ck=torch.load(file,map_location='cpu',weights_only=False)
  assert ck['scientific_plan_sha256']==sha(pp) and ck['cache_manifest_sha256']==rep['cache_manifest_sha256'] and ck['preprocessing']==pre and ck['vlm_frozen'] and ck['seed']==42
  expected=0 if name=='initial' else 43 if name=='best' else 50;assert ck['epoch']==expected and ck['updates']==expected*1899 and len(ck['history'])==expected
  assert ck['cpu_rng'].dtype==torch.uint8 and len(ck['cuda_rng'])==1 and ck['cuda_rng'][0].dtype==torch.uint8
  if expected:assert {int(s['step']) for s in ck['optimizer']['state'].values()}=={ck['updates']}
  checkpoints[name]=dict(epoch=expected,updates=ck['updates'],sha256=sha(file))
  if name=='initial':initial=ck['state_dict']
  if name=='best':best=ck
 torch.manual_seed(42);reference=FlowHead(HeadConfig(**best['model_config']));assert sum(x.numel() for x in reference.parameters())==432711
 for k,v in reference.state_dict().items():torch.testing.assert_close(v,initial[k],rtol=0,atol=0)
 model=FlowHead(HeadConfig(**best['model_config']));model.load_state_dict(best['state_dict']);model.cuda().eval()
 for prefix in ['observation_projection.','state_encoder.','action_queries','decoder.','output.','noisy_action_projection.','time_mlp.']:assert any(not torch.equal(initial[k],v) for k,v in best['state_dict'].items() if k.startswith(prefix))
 cache=R/p['cache'];numeric=R/p['numeric_cache']/'validation';arrays={k:np.load(numeric/(k+'.npy'),mmap_mode='r') for k in ['state','actions','episode_id','task_id']};n=len(arrays['state']);assert n==30465
 index=json.loads((cache/'validation/index.json').read_text());ends=[e['end'] for e in index];loaded={};end_by_id={int(e):int(np.flatnonzero(arrays['episode_id']==e)[-1]) for e in np.unique(arrays['episode_id'])}
 first_sum=chunk_sum=constant_sum=0.;valid_count=0;tasks={i:[0.,0] for i in range(40)}
 with torch.no_grad():
  for startidx in range(0,n,128):
   stop=min(n,startidx+128);rows=[];targets=[];masks=[];noises=[]
   for i in range(startidx,stop):
    entry=index[bisect.bisect_right(ends,i)];eid=int(arrays['episode_id'][i]);assert eid==entry['episode_id']
    if eid not in loaded:loaded={eid:np.load(cache/entry['directory']/'hidden.npy',mmap_mode='r')}
    raw=f'20261020:{eid}:{i-entry["start"]}'.encode();ns=int.from_bytes(hashlib.sha256(raw).digest()[:8],'little')%(2**63-1);noises.append(torch.randn((10,7),generator=torch.Generator().manual_seed(ns),dtype=torch.float32))
    rows.append(np.array(loaded[eid][i-entry['start']],copy=True));end=end_by_id[eid];indices=np.minimum(np.arange(i,i+10),end);assert np.all(arrays['episode_id'][indices]==eid)
    targets.append(np.array(arrays['actions'][indices,0],dtype=np.float64));masks.append(np.arange(i,i+10)<=end)
   width=max(len(x) for x in rows);bits=np.zeros((len(rows),width,960),np.uint16);mask=np.zeros((len(rows),width),bool)
   for i,row in enumerate(rows):bits[i,:len(row)]=row;mask[i,:len(row)]=True
   memory,padding=model.memory(hidden=torch.from_numpy(bits).view(torch.bfloat16).cuda(),hidden_mask=torch.from_numpy(mask).cuda(),state=torch.from_numpy(np.array(arrays['state'][startidx:stop])).cuda())
   generated=torch.stack(noises).cuda()
   for step in range(10):
    tau=torch.full((len(rows),),step/10,dtype=torch.float32,device='cuda')
    velocity=model.velocity(memory=memory,memory_padding=padding,noisy_action=generated,time=tau)
    generated=generated+velocity/10
   pred=generated.cpu().numpy().astype(np.float64)
   target=np.stack(targets);valid=np.stack(masks);assert np.isfinite(pred).all();error=(pred-target)**2;first_sum+=error[:,0].sum();chunk_sum+=(error*valid[:,:,None]).sum();constant_sum+=(target[:,0]**2).sum();valid_count+=int(valid.sum())
   for task,e in zip(arrays['task_id'][startidx:stop],error[:,0].mean(1)):tasks[int(task)][0]+=float(e);tasks[int(task)][1]+=1
   if startidx%3200==0:print('independent validation',stop,'/30465',flush=True)
 assert valid_count==296235
 values=dict(normalized_mse=float(first_sum/(n*7)),all_valid_chunk_mse=float(chunk_sum/(valid_count*7)),mean_action_normalized_mse=float(constant_sum/(n*7)))
 for k,v in values.items():assert abs(v-rep['selected_validation'][k])<2e-5,(k,v,rep['selected_validation'][k])
 task_errors={str(t):abs(s/count-rep['selected_validation']['per_task'][str(t)]['mse']) for t,(s,count) in tasks.items()}
 for t,(s,count) in tasks.items():assert count==rep['selected_validation']['per_task'][str(t)]['frames'] and task_errors[str(t)]<2e-5
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,ax=plt.subplots(1,2,figsize=(10,4));x=[r['epoch'] for r in history];ax[0].plot(x,[r['train_velocity_mse'] for r in history],label='train velocity MSE');ax[0].legend();ax[1].plot(x,[r['validation']['normalized_mse'] for r in history],label='val first-actionMSE');ax[1].axvline(43,color='gray',linestyle='--');ax[1].legend();fig.tight_layout();fig.savefig(RUN/'training_review.png',dpi=140);plt.close(fig)
 result=dict(status='PASS',epochs=50,updates=94950,train_frames_per_epoch=243000,validation_frames=n,valid_K10_validation_targets=valid_count,selected_epoch=43,selected_updates=81657,head_parameters=432711,frozen_VLM_parameters=507482304,trainable_VLM_parameters=0,checkpoints=checkpoints,independent_validation=values,max_per_task_numerical_error=max(task_errors.values()),all40_tasks_matched=True,seed42_initialization_exact=True,all_head_module_groups_updated=True,elapsed_seconds=time.monotonic()-start,new_training_updates=0,new_simulation_episodes=0,confirmation_accessed=False,independent_noise_seed_and_Euler_loop=True,validation_target_not_input=True,claim_boundary='Single fixed-noise generated-action MSE;not directly a point-regression optimum or a newclosedloop success score.')
 (out/'independent_acceptance.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
