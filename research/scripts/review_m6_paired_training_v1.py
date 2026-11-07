"""Independent full-run source, optimizer, parameter, data and validation audit."""
import hashlib,json,math,time
from pathlib import Path
import numpy as np
import torch
from mini_vla.paired_continuation_v1 import sample_order,lr_factor
from mini_vla.train_vlm_head_v1 import sha,write
R=Path(__file__).resolve().parents[1]
def main():
 started=time.monotonic();torch.set_num_threads(2);run=R/'results/mini_vla_m6_paired_20261028_run1';p=json.loads((run/'plan.json').read_text());ph=sha(run/'plan.json')
 for f,h in p['input_sha256'].items():
  assert sha(R/f)==h,f
  frozen=run/'frozen_source'/f
  if frozen.exists():assert sha(frozen)==h
 split=json.loads((R/p['split']).read_text());train_ids={x['episode_id'] for x in split['train']};val_ids={x['episode_id'] for x in split['validation']};assert len(train_ids)==1506 and len(val_ids)==187 and not train_ids&val_ids
 order=sample_order(243000);oh=hashlib.sha256(order.astype('<i8').tobytes()).hexdigest();index=json.loads((R/p['cache']/'train/index.json').read_text());tails=np.empty(243000,dtype=np.int64)
 for e in index:tails[e['start']:e['end']]=np.minimum(10,np.arange(e['end']-e['start'],0,-1))
 counts=np.cumsum(tails[order]);old=torch.load(R/p['head_checkpoint'],map_location='cpu',weights_only=False)
 metrics=json.loads((run/'review/full_validation_recomputed.json').read_text());assert metrics['status']=='PASS'
 summary={};initial_predictions=[]
 for arm in ['frozen','lora']:
  out=run/arm;report=json.loads((out/'complete.json').read_text());assert report['updates']==1899 and report['frames']==243000 and report['plan_sha256']==ph and report['sample_order_sha256']==oh
  for f,h in report['checkpoints'].items():assert sha(out/f)==h
  initial=torch.load(out/'initial.pt',map_location='cpu',weights_only=False);end=torch.load(out/'endpoint.pt',map_location='cpu',weights_only=False);latest=torch.load(out/'latest.pt',map_location='cpu',weights_only=False)
  assert initial['updates']==0 and not initial['optimizer']['state']
  assert all(torch.equal(v,old['state_dict'][k]) for k,v in initial['state_dict'].items())
  for c in [end,latest]:
   assert c['updates']==1899 and c['seen']==243000 and c['valid_target_steps']==2362230 and c['plan_sha256']==ph and c['sample_order_sha256']==oh
   assert 'cpu_rng' in c and 'cuda_rng' in c and len(c['optimizer']['state'])>0
  assert all(torch.equal(v,latest['state_dict'][k]) for k,v in end['state_dict'].items())
  assert all(torch.equal(v,latest['adapters'][k]) for k,v in end['adapters'].items())
  changed_head=[k for k,v in end['state_dict'].items() if not torch.equal(v,initial['state_dict'][k])];assert changed_head
  changes={};delta_sq={}
  for k,v in end['adapters'].items():
   assert torch.isfinite(v).all() and k.endswith(('.lora_a','.lora_b'))
   group='vision' if 'vision_model' in k else 'text';assert 'vision_model' in k or 'text_model' in k
   changes[group]=changes.get(group,0)+int(not torch.equal(v,initial['adapters'][k]));delta_sq[group]=delta_sq.get(group,0.)+float((v.double()-initial['adapters'][k].double()).square().sum())
   if k.endswith('.lora_b'):assert initial['adapters'][k].count_nonzero()==0
  if arm=='lora':assert set(changes)=={'vision','text'} and min(changes.values())>0 and sum(v.numel() for v in end['adapters'].values())==2228224
  else:assert not end['adapters']
  rows=[json.loads(x) for x in (out/'updates.jsonl').read_text().splitlines()];assert len(rows)==1899
  for j,row in enumerate(rows):
   seen=min((j+1)*128,243000);assert row['updates']==j+1 and row['seen']==seen and row['valid_target_steps']==int(counts[seen-1]);assert math.isfinite(row['loss']) and math.isfinite(row['gradient_norm'])
   expected=[3e-5*lr_factor(j)]+([1e-5*lr_factor(j)] if arm=='lora' else [])
   np.testing.assert_allclose(row['lr'],expected,atol=1e-15,rtol=0)
  assert abs(sum(row['loss']*(int(counts[min((j+1)*128,243000)-1])-(int(counts[j*128-1]) if j else 0))*7 for j,row in enumerate(rows))-end['train_sum'])<.01
  first=metrics['arms'][arm]['initial']['first_mse'];last=metrics['arms'][arm]['endpoint']['first_mse'];assert report['selected']==('endpoint.pt' if last<first else 'initial.pt')
  summary[arm]=dict(updates=1899,frames=243000,validation_frames=30465,initial_mse=first,endpoint_mse=last,selected=report['selected'],checkpoint=str((out/report['selected']).relative_to(R)),checkpoint_sha256=sha(out/report['selected']),head_tensors_updated=len(changed_head),adapter_tensors_updated=changes,adapter_delta_l2={k:math.sqrt(v) for k,v in delta_sq.items()},train_step_total_seconds=sum(row['seconds'] for row in rows),wall_seconds=report['elapsed_this_process_seconds'])
  initial_predictions.append(np.load(out/'initial_predictions.npy'))
 assert np.array_equal(*initial_predictions)
 status=json.loads((run/'status.json').read_text());assert status['status']=='COMPLETE'
 ledger=json.loads((R/p['resources']['ledger']).read_text());entry=next(x for x in ledger['attempts'] if x['run']==str(run));assert entry['status']=='COMPLETE' and abs(entry['elapsed_seconds']-status['elapsed_seconds'])<1
 result=dict(status='PASS',plan_sha256=ph,arms=summary,all_initial_validation_predictions_bitwise_equal=True,source_bindings_verified=len(p['input_sha256']),train_validation_episode_disjoint=True,supervised_elapsed_seconds=status['elapsed_seconds'],lora_minus_frozen_mse=summary['lora']['endpoint_mse']-summary['frozen']['endpoint_mse'],new_gpu_compute=0,new_simulation_episodes=0,elapsed_seconds=time.monotonic()-started,limitations=['Base weights were not saved in adapted checkpoints. Frozen base proof uses source/optimizer exclusion and run-time no-gradient assertions; no full before/after live-base hash exists.', 'One continuation epoch and one training seed; offline MSE is not closed-loop success.'])
 write(run/'review/independent_acceptance.json',result);print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
