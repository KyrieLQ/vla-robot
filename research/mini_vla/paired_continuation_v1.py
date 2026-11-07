"""One complete paired continuation epoch, microbatch1, commit every optimizer step.
No final benchmark access. Raw observations, not frozen features, feed the LoRA arm.
"""
import argparse,bisect,hashlib,json,math,os,signal,time
from collections import OrderedDict
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch
from .smolvlm_lora_v1 import DifferentiableFrontend,adapter_parameters
from .vlm_frontend_v1 import decode_record
from .vlm_cached_data_v2 import VLMCacheData
from .vlm_action_heads_v1 import RegressionHead,HeadConfig
from .train_vlm_head_v1 import sha,write,save
R=Path(__file__).resolve().parents[1]
STOP=False

def request_stop(*_):
 global STOP
 STOP=True

def lr_factor(update,total=1899,warmup=100):
 if not 0<=update<total:raise ValueError(update)
 if update<warmup:return (update+1)/warmup
 phase=(update-warmup)/(total-1-warmup)
 return .1+.9*.5*(1+math.cos(math.pi*phase))

def sample_order(count,seed=42):
 return torch.randperm(count,generator=torch.Generator().manual_seed(seed)).numpy()

def loss_numerator(pred,target,mask):
 return torch.where(mask[...,None],(pred-target).square(),0.).sum()

class RawData:
 """Bounded RAM parquet LRU; never loads any hidden.npy or retired image array."""
 def __init__(self,plan,part):
  self.index=json.loads((R/plan['cache']/part/'index.json').read_text());self.ends=[e['end'] for e in self.index]
  self.meta={e['episode_id']:e for e in json.loads((R/plan['split']).read_text())[part]}
  self.numeric={k:np.load(R/plan['numeric_cache']/part/(k+'.npy'),mmap_mode='r') for k in ['state','actions','episode_id','task_id']}
  self.root=R/plan['dataset_root'];self.tables=OrderedDict();self.table_bytes=0
  assert len(self.numeric['state'])==self.ends[-1]
  for e in self.index:
   assert e['end']-e['start']==self.meta[e['episode_id']]['length']
   assert np.all(self.numeric['episode_id'][e['start']:e['end']]==e['episode_id'])
 def __len__(self):return self.ends[-1]
 def episode(self,i):
  if not 0<=i<len(self):raise IndexError(i)
  return self.index[bisect.bisect_right(self.ends,i)]
 def valid_count(self,i):return min(10,self.episode(i)['end']-i)
 def __getitem__(self,i):
  e=self.episode(i);eid=e['episode_id'];f=i-e['start']
  if eid not in self.tables:
   table=pq.read_table(self.root/f'data/chunk-{eid//1000:03d}/episode_{eid:06d}.parquet',columns=['image','wrist_image','episode_index','frame_index'],use_threads=False)
   self.tables[eid]=table;self.table_bytes+=table.nbytes
   while len(self.tables)>1 and (len(self.tables)>12 or self.table_bytes>256*2**20):
    _,old=self.tables.popitem(last=False);self.table_bytes-=old.nbytes
  self.tables.move_to_end(eid);row=self.tables[eid].slice(f,1).to_pylist()[0]
  assert (row['episode_index'],row['frame_index'])==(eid,f)
  requested=np.arange(i,i+10);valid=requested<e['end'];indices=np.minimum(requested,e['end']-1)
  return dict(record=decode_record(row,self.meta[eid]['instruction']),state=torch.from_numpy(np.array(self.numeric['state'][i])),actions=torch.from_numpy(np.array(self.numeric['actions'][indices,0])),action_mask=torch.from_numpy(valid),episode_id=eid,frame_index=f,task_id=int(self.numeric['task_id'][i]))

def observations(sample,front):
 if front is None:
  hidden=sample['hidden'][None].cuda();mask=sample['hidden_mask'][None].cuda()
 else:hidden,mask=front(front.inputs([sample['record']]))
 return dict(hidden=hidden,hidden_mask=mask,state=sample['state'][None].cuda())

@torch.no_grad()
def evaluate(head,front,data,out,phase,progress):
 head.eval()
 if front is not None:front.eval()
 # Persist every prediction to allow full independent CPU recomputation, not just means.
 preds=np.empty((len(data),10,7),np.float32);first=chunk=0.;targets=0;began=time.monotonic()
 for i in range(len(data)):
  s=data[i];p=head(**observations(s,front))[0]
  assert torch.isfinite(p).all();preds[i]=p.cpu().numpy();err=(preds[i].astype(np.float64)-s['actions'].numpy())**2
  first+=float(err[0].sum());chunk+=float(err[s['action_mask'].numpy()].sum());targets+=int(s['action_mask'].sum())
  if i%250==0:write(progress,dict(stage=phase,frames=i+1,total_frames=len(data),elapsed_seconds=time.monotonic()-began))
  if STOP:raise InterruptedError('Validation interrupted; no model selection committed')
 assert len(data)==30465 and targets==296235
 tmp=out.with_suffix('.npy.tmp')
 with tmp.open('wb') as f:np.save(f,preds);f.flush();os.fsync(f.fileno())
 tmp.replace(out)
 return dict(frames=len(data),valid_target_steps=targets,normalized_mse=first/(7*len(data)),all_valid_chunk_mse=chunk/(7*targets),predictions_sha256=sha(out),elapsed_seconds=time.monotonic()-began)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--run',required=True);args=ap.parse_args()
 pp=Path(args.plan).resolve();p=json.loads(pp.read_text());run=Path(args.run).resolve();run.mkdir(exist_ok=True)
 assert p['authorization']['time_overrun_allowed'] and p['training']['additional_full_epochs']==1
 for f,h in p['input_sha256'].items():assert sha(R/f)==h,f
 downloads=json.loads((R/p['model_manifest']).read_text())
 for name,meta in downloads['files'].items():
  assert sha(R/p['model_directory']/name)==meta['sha256'],name
 assert torch.cuda.device_count()==1
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
 initial=torch.load(R/p['head_checkpoint'],map_location='cpu',weights_only=False)
 assert initial['epoch']==12 and initial['architecture']=='frozen_vlm_regression'
 order=sample_order(243000);order_sha=hashlib.sha256(order.astype('<i8').tobytes()).hexdigest();assert len(np.unique(order))==243000
 write(run/'sample_order.json',dict(seed=42,frames=243000,sha256=order_sha,updates=1899,last_batch=56))
 for arm in ['frozen','lora']:
  out=run/arm;out.mkdir(exist_ok=True);progress=run/'progress.json'
  if (out/'complete.json').exists():
   committed=json.loads((out/'complete.json').read_text());assert committed['plan_sha256']==sha(pp)
   for name,h in committed['checkpoints'].items():assert sha(out/name)==h
   continue
  torch.manual_seed(42);front=None
  if arm=='lora':front=DifferentiableFrontend(str(R/p['model_directory']),rank=8,alpha=8,checkpointing=True)
  head=RegressionHead(HeadConfig(**initial['model_config'])).cuda();head.load_state_dict(initial['state_dict'])
  ads={} if front is None else adapter_parameters(front.model)
  groups=[dict(params=list(head.parameters()),lr=3e-5,base_lr=3e-5)]
  if ads:groups.append(dict(params=list(ads.values()),lr=1e-5,base_lr=1e-5))
  opt=torch.optim.AdamW(groups,weight_decay=1e-4)
  train=VLMCacheData(R/p['cache'],R/p['numeric_cache'],'train',max_open_episodes=12) if front is None else RawData(p,'train')
  val=VLMCacheData(R/p['cache'],R/p['numeric_cache'],'validation',max_open_episodes=2) if front is None else RawData(p,'validation')
  seen=updates=target_count=0;train_sum=0.;history=[];began=time.monotonic()
  def snapshot():
   return dict(architecture='smolvlm_paired_regression',arm=arm,model_config=initial['model_config'],state_dict={k:v.detach().cpu() for k,v in head.state_dict().items()},adapters={k:v.detach().cpu() for k,v in ads.items()},optimizer=opt.state_dict(),updates=updates,seen=seen,valid_target_steps=target_count,train_sum=train_sum,preprocessing=initial['preprocessing'],initial_checkpoint_sha256=p['input_sha256'][p['head_checkpoint']],plan_sha256=sha(pp),sample_order_sha256=order_sha,cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),history=history,vlm_frozen=not bool(ads))
  if (out/'latest.pt').exists():
   c=torch.load(out/'latest.pt',map_location='cpu',weights_only=False);assert c['plan_sha256']==sha(pp) and c['sample_order_sha256']==order_sha and c['arm']==arm
   head.load_state_dict(c['state_dict']);assert set(c['adapters'])==set(ads)
   for n,v in ads.items():v.data.copy_(c['adapters'][n])
   opt.load_state_dict(c['optimizer']);seen=c['seen'];updates=c['updates'];target_count=c['valid_target_steps'];train_sum=c['train_sum'];history=c['history']
   torch.set_rng_state(c['cpu_rng']);torch.cuda.set_rng_state_all(c['cuda_rng'])
  if not (out/'initial_validation.json').exists():
   assert updates==0
   metric=evaluate(head,front,val,out/'initial_predictions.npy',arm+'_initial_validation',progress)
   save(out/'initial.pt',snapshot());write(out/'initial_validation.json',metric)
  initial_metric=json.loads((out/'initial_validation.json').read_text());assert sha(out/'initial_predictions.npy')==initial_metric['predictions_sha256']
  head.train()
  if front is not None:front.train();front.enable_adapter_training(True)
  params=[x for g in opt.param_groups for x in g['params']]
  for start in range(seen,len(train),128):
   if STOP:raise InterruptedError('Stopped at committed optimizer boundary')
   tic=time.monotonic();ids=order[start:start+128].tolist()
   denom=7*sum(min(10,train.index[bisect.bisect_right(train.ends,i)]['end']-i) for i in ids)
   opt.zero_grad(set_to_none=True)
   for g in opt.param_groups:g['lr']=g['base_lr']*lr_factor(updates)
   batch_sum=0.
   for i in ids:
    s=train[i];pred=head(**observations(s,front));num=loss_numerator(pred,s['actions'][None].cuda(),s['action_mask'][None].cuda())
    assert torch.isfinite(num);(num/denom).backward();batch_sum+=float(num.detach())
   if front is not None:
    assert all(x.grad is None for n,x in front.model.named_parameters() if n not in ads)
   params=[x for g in opt.param_groups for x in g['params']]
   norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step()
   seen+=len(ids);updates+=1;target_count+=denom//7;train_sum+=batch_sum
   # Every completed update is durable. Only overwritten latest checkpoint is superseded.
   save(out/'latest.pt',snapshot())
   row=dict(arm=arm,updates=updates,seen=seen,valid_target_steps=target_count,loss=batch_sum/denom,lr=[g['lr'] for g in opt.param_groups],gradient_norm=float(norm),seconds=time.monotonic()-tic)
   with (out/'updates.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush()
   write(progress,dict(stage=arm+'_training',updates=updates,total_updates=1899,frames=seen,total_frames=243000,last_step_seconds=row['seconds'],estimated_training_seconds_left=(1899-updates)*row['seconds']))
   if updates%20==0:print(json.dumps(row),flush=True)
  assert updates==1899 and seen==243000 and target_count==2362230
  end=evaluate(head,front,val,out/'endpoint_predictions.npy',arm+'_endpoint_validation',progress);write(out/'endpoint_validation.json',end)
  save(out/'endpoint.pt',snapshot());selected='endpoint.pt' if end['normalized_mse']<initial_metric['normalized_mse'] else 'initial.pt'
  report=dict(status='COMPLETE_REVIEW_REQUIRED',arm=arm,initial_validation=initial_metric,endpoint_validation=end,selected=selected,selection='complete validation first-action MSE; initial wins ties',updates=updates,frames=seen,plan_sha256=sha(pp),sample_order_sha256=order_sha,checkpoints={n:sha(out/n) for n in ['initial.pt','latest.pt','endpoint.pt']},elapsed_this_process_seconds=time.monotonic()-began,trainable_adapters=sum(x.numel() for x in ads.values()),trainable_head=sum(x.numel() for x in head.parameters()),new_simulation_episodes=0)
  write(out/'complete.json',report)
  del front,head,opt,params,groups,ads,train,val;torch.cuda.empty_cache()
 write(run/'training_complete.json',dict(status='COMPLETE_REVIEW_REQUIRED',arms=['frozen','lora'],updates_per_arm=1899,closedloop_started=False,final2000_accessed=False))
 return 0
if __name__=='__main__':raise SystemExit(main())
