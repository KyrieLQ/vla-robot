"""M9 frozen-VLM conditional flow training: epoch-safe checkpoints and full validation."""
import argparse,hashlib,json,math,os,random,shutil,time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from .vlm_action_heads_v1 import FlowHead,HeadConfig,flow_loss,example_seed
from .vlm_cached_data_v2 import VLMCacheData,collate_cached
from .model import masked_mse
R=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
 return h.hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def save(p,x):
 t=p.with_suffix('.pt.tmp');torch.save(x,t)
 with t.open('rb') as f:os.fsync(f.fileno())
 t.replace(p)
def seed(value):
 random.seed(value);np.random.seed(value);torch.manual_seed(value);torch.cuda.manual_seed_all(value)
def worker_init(_):torch.set_num_threads(1)
def loader(data,shuffle=False,epoch=0,workers=2):
 args=dict(batch_size=128,shuffle=shuffle,drop_last=False,num_workers=workers,collate_fn=collate_cached,pin_memory=True,generator=torch.Generator().manual_seed(42+epoch))
 if workers:args.update(worker_init_fn=worker_init,multiprocessing_context='spawn',prefetch_factor=2,persistent_workers=False)
 return DataLoader(data,**args)
def move(batch):
 obs={k:batch[k].cuda(non_blocking=True) for k in ['hidden','hidden_mask','state']}
 return obs,batch['actions'].cuda(non_blocking=True),batch['action_mask'].cuda(non_blocking=True)
def optimizer(model):return torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
def fixed_noise(batch):
 # Private CPU generators: per-frame noise is independent of batching, order and train RNG.
 return torch.stack([torch.randn((10,7),generator=torch.Generator().manual_seed(example_seed(int(e),int(f))),dtype=torch.float32) for e,f in zip(batch['episode_id'],batch['frame_index'])])
def training_loss(model,obs,target,mask):
 noise=torch.randn_like(target);time=torch.rand(len(target),device=target.device)
 return flow_loss(model,obs,target,mask,noise,time)
@torch.no_grad()
def evaluate(model,data_loader):
 model.eval();first=chunk=constant=0.;frames=targets=0;per={i:[0.,0] for i in range(40)}
 for batch in data_loader:
  obs,target,mask=move(batch);pred=model.sample(**obs,noise=fixed_noise(batch).cuda());assert pred.shape==target.shape and torch.isfinite(pred).all()
  error=(pred-target).square();first+=float(error[:,0].double().sum());chunk+=float(torch.where(mask[...,None],error,0.).double().sum());constant+=float(target[:,0].double().square().sum());frames+=len(pred);targets+=int(mask.sum())
  values=error[:,0].double().mean(-1).cpu().numpy()
  for task,err in zip(batch['task_id'].tolist(),values):per[task][0]+=float(err);per[task][1]+=1
 return dict(frames=frames,valid_target_steps=targets,normalized_mse=first/(frames*7),all_valid_chunk_mse=chunk/(targets*7),mean_action_normalized_mse=constant/(frames*7),per_task={str(k):dict(frames=n,mse=s/n) for k,(s,n) in per.items() if n})
def budget_remaining(run):
 from datetime import datetime
 state=json.loads((run/'status.json').read_text())
 return max(0.,state['timeout_seconds']-(time.time()-datetime.fromisoformat(state['started_at']).timestamp()))
def profile(plan,run,train,val):
 seed(42);model=FlowHead(HeadConfig(**plan['head_config'])).cuda();opt=optimizer(model)
 train_ids=np.linspace(0,len(train)-1,20*128,dtype=np.int64).tolist();val_ids=np.linspace(0,len(val)-1,10*128,dtype=np.int64).tolist()
 np.random.default_rng(20261021).shuffle(train_ids)
 assert set(int(train.numeric['task_id'][i]) for i in train_ids)==set(range(40))
 write(run/'profile_samples.json',dict(train_indices=train_ids,validation_indices=val_ids,selection='uniformly-spaced train frames shuffled with seed20261021 to represent random batch I/O; evenly-spaced validation frames',train_tasks=40))
 train_loader=loader(Subset(train,train_ids),workers=plan['workers']);val_loader=loader(Subset(val,val_ids),workers=plan['workers']);steps=[];forward=[];warm_start=time.monotonic();last=time.perf_counter();torch.cuda.reset_peak_memory_stats();model.train()
 for batch in train_loader:
  obs,target,mask=move(batch);opt.zero_grad(set_to_none=True);loss=training_loss(model,obs,target,mask);assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);assert torch.isfinite(norm);opt.step();torch.cuda.synchronize();t=time.perf_counter();steps.append(t-last);last=t
 model.eval();last=time.perf_counter()
 with torch.no_grad():
  for batch in val_loader:
   obs,target,mask=move(batch);out=model.sample(**obs,noise=fixed_noise(batch).cuda());assert torch.isfinite(out).all();torch.cuda.synchronize();t=time.perf_counter();forward.append(t-last);last=t
 assert len(steps)==20 and len(forward)==10
 train_time=float(np.mean(steps[5:]));val_time=float(np.mean(forward[2:]));estimate=1.15*(94950*train_time+239*51*val_time+600)
 result=dict(status='PASS' if estimate<budget_remaining(run)-120 else 'BUDGET_INSUFFICIENT',train_batch_seconds=train_time,validation_batch_seconds=val_time,estimated_full_training_validation_audit_seconds=estimate,available_after_profile_seconds=budget_remaining(run),discarded_profile_optimizer_updates=20,profile_samples=2560,parameter_count=sum(p.numel() for p in model.parameters()),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,raw_train_batch_seconds=steps,raw_validation_batch_seconds=forward,elapsed_seconds=time.monotonic()-warm_start)
 write(run/'throughput_gate.json',result);print(json.dumps(result),flush=True);del model,opt;torch.cuda.empty_cache()
 return result['status']=='PASS'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--run',required=True);args=ap.parse_args();pp=Path(args.plan);p=json.loads(pp.read_text());run=Path(args.run);out=run/'training';out.mkdir(exist_ok=True);started=time.monotonic()
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 assert torch.cuda.device_count()==1
 for f,h in p['input_sha256'].items():assert sha(R/f)==h,f
 acceptance=json.loads((R/p['cache_acceptance']).read_text());assert acceptance['status']=='PASS' and acceptance['frames']==273465
 assert sha(R/p['cache']/'complete.json')==acceptance['cache_manifest_sha256']
 train=VLMCacheData(R/p['cache'],R/p['numeric_cache'],'train');val=VLMCacheData(R/p['cache'],R/p['numeric_cache'],'validation');assert len(train)==243000 and len(val)==30465
 if not profile(p,run,train,val):return 78
 seed(42);model=FlowHead(HeadConfig(**p['head_config'])).cuda();opt=optimizer(model);assert sum(x.numel() for x in model.parameters())==p['expected_parameters']
 val_loader=loader(val,workers=p['workers']);history=[];updates=0;best_value=float('inf');best_epoch=None;scientific_sha=sha(R/p.get('scientific_plan',str(pp.relative_to(R) if pp.is_absolute() else pp)));pre=json.loads((R/p['preprocessing']).read_text())
 def checkpoint(epoch,validation):
  return dict(architecture='frozen_vlm_flow',model_config=asdict(model.config),state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=opt.state_dict(),epoch=epoch,updates=updates,seed=42,scientific_plan_sha256=scientific_sha,execution_plan_sha256=sha(pp),cache_manifest_sha256=acceptance['cache_manifest_sha256'],cache_binding=json.loads((R/p['cache']/'complete.json').read_text())['binding'],preprocessing=pre,vlm_frozen=True,history=history,best_epoch=best_epoch,best_value=best_value,validation=validation,cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
 initial_validation=evaluate(model,val_loader);assert initial_validation['frames']==30465 and initial_validation['valid_target_steps']==296235;write(out/'initial_validation.json',initial_validation);save(out/'initial.pt',checkpoint(0,initial_validation));start_epoch=1
 if p.get('resume_checkpoint'):
  ck=torch.load(R/p['resume_checkpoint'],map_location='cpu',weights_only=False);assert ck['scientific_plan_sha256']==scientific_sha and ck['cache_manifest_sha256']==acceptance['cache_manifest_sha256'];model.load_state_dict(ck['state_dict']);opt.load_state_dict(ck['optimizer']);history=ck['history'];updates=ck['updates'];best_value=ck['best_value'];best_epoch=ck['best_epoch'];start_epoch=ck['epoch']+1;torch.set_rng_state(ck['cpu_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng']);shutil.copy2(R/p['resume_best_checkpoint'],out/'best.pt')
 torch.cuda.reset_peak_memory_stats()
 for epoch in range(start_epoch,51):
  epoch_start=time.monotonic();model.train();train_sum=0.;seen=targets=0
  for batch in loader(train,shuffle=True,epoch=epoch,workers=p['workers']):
   obs,target,mask=move(batch);opt.zero_grad(set_to_none=True);loss=training_loss(model,obs,target,mask);assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);assert torch.isfinite(norm);opt.step();count=int(mask.sum());train_sum+=float(loss.detach())*count*7;targets+=count;seen+=len(target);updates+=1
   if updates%100==0:
    elapsed=time.monotonic()-started;write(run/'progress.json',dict(stage='training',epoch=epoch,epochs=50,updates=updates,total_updates=94950,frames_in_epoch=seen,training_velocity_mse_running=train_sum/(targets*7),elapsed_seconds=elapsed,estimated_remaining_seconds=(94950-updates)*(time.monotonic()-epoch_start)/max((seen+127)//128,1)));print(f'epoch{epoch} update{updates}/94950 trainVelocityMSE{train_sum/(targets*7):.6f}',flush=True)
  assert seen==243000 and targets==2362230 and updates==epoch*1899
  metrics=evaluate(model,val_loader);assert metrics['frames']==30465 and metrics['valid_target_steps']==296235 and len(metrics['per_task'])==40
  history.append(dict(epoch=epoch,updates=updates,train_frames=seen,valid_target_steps=targets,train_velocity_mse=train_sum/(targets*7),validation=metrics,elapsed_seconds=time.monotonic()-epoch_start))
  if metrics['normalized_mse']<best_value:best_value=metrics['normalized_mse'];best_epoch=epoch;save(out/'best.pt',checkpoint(epoch,metrics))
  save(out/'latest.pt',checkpoint(epoch,metrics));write(out/'history.json',history);print(f'completed epoch{epoch}: validationFirstMSE={metrics["normalized_mse"]:.6f} selectedEpoch={best_epoch}',flush=True)
 best=torch.load(out/'best.pt',map_location='cpu',weights_only=False);assert len(history)==50 and updates==94950
 report=dict(flow_steps=10,noise_seed_rule='SHA256(20261020,episode,frame);privateCPUfloat32generator',status='COMPLETE_REVIEW_REQUIRED',architecture='frozenVLM+conditionalflowhead',epochs=50,updates=94950,best_epoch=best_epoch,selected_validation=best['validation'],training_seed=42,parameters=sum(x.numel() for x in model.parameters()),vlm_trainable_parameters=0,vlm_total_parameters=507482304,checkpoint_sha256={n:sha(out/(n+'.pt')) for n in ['initial','best','latest']},cache_manifest_sha256=acceptance['cache_manifest_sha256'],training_elapsed_seconds=time.monotonic()-started,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,discarded_profile_optimizer_updates=20,new_simulation_episodes=0,confirmation_accessed=False)
 write(out/'report.json',report);write(run/'progress.json',dict(stage='complete',epochs=50,updates=94950,best_epoch=best_epoch));print(json.dumps(report),flush=True);return 0
if __name__=='__main__':raise SystemExit(main())
