"""Independent complete50epoch/metadata/full saved prediction audit; no RAM recreation."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from mini_vla.vlm_action_heads_v1 import HeadConfig
from mini_vla.flow_time_film_v1 import TimeFiLMFlowHead as FlowHead,verify_common_initial
from mini_vla.train_vlm_head_v1 import sha,write
R=Path(__file__).resolve().parents[1]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args();RUN=Path(a.run)
 started=time.monotonic();torch.set_num_threads(2);p=json.loads((RUN/'plan.json').read_text());report=json.loads((RUN/'training/report.json').read_text());out=RUN/'review';out.mkdir(exist_ok=True)
 for f,h in p['input_sha256'].items():
  assert sha(R/f)==h,f
  frozen=RUN/'frozen_source'/f
  if frozen.exists():assert sha(frozen)==h
 split=json.loads((R/p['split']).read_text());train={e['episode_id']:e for e in split['train']};val={e['episode_id']:e for e in split['validation']};assert len(train)==1506 and len(val)==187 and not set(train)&set(val)
 receipts=[json.loads(x) for x in (RUN/'cache_receipts.jsonl').read_text().splitlines()];assert len(receipts)==1693 and len({e['episode_id'] for e in receipts})==1693
 cache=json.loads((RUN/'ram_cache_acceptance.json').read_text());assert cache['status']=='PASS' and cache['receipt_sha256']==sha(RUN/'cache_receipts.jsonl') and cache['allRAM_receipts_second_pass_verified'] and report['ram_cache_immutable_after_training']
 assert cache['frontend_checkpoint_sha256']==p['backbone_checkpoint_sha256']
 assert cache['all1693_features_equal_original_adapted_receipts'] and 2<=cache['parallel_workers']<=4
 assert sha(RUN/'cache_receipts.jsonl')==sha(R/p['reference_adapted_receipts'])
 for i in range(1,cache['parallel_workers']):
  worker=json.loads((RUN/'workers'/str(i)/'gate.json').read_text());assert worker['status']=='PASS' and worker['samples']==120 and worker['all_saved_adapted_gate_features_equal']
 assert json.loads((RUN/'throughput_admission.json').read_text())['status']=='PASS'
 assert json.loads((RUN/'initial_time_contract.json').read_text())['initial_generation_bitwise_equal']
 count=bytes_=0;metas={**train,**val};layouts={}
 for part in ['train','validation']:
  layouts.update({e['episode_id']:e for e in json.loads((R/p['cache']/part/'index.json').read_text())})
 for e in receipts:
  eid=e['episode_id'];meta=metas[eid];layout=layouts[eid];assert e['part']==('train' if eid in train else 'validation') and e['frames']==meta['length'] and e['shape']==[meta['length'],layout['tokens'],960]
  source=R/p['dataset_root']/f'data/chunk-{eid//1000:03d}/episode_{eid:06d}.parquet';assert sha(source)==e['source_sha256']
  hashes=json.loads((R/p['cache']/layout['directory']/'input_hashes.json').read_text());h=hashlib.sha256()
  for item in hashes:h.update((item['input_sha256'] if isinstance(item,dict) else item).encode())
  assert h.hexdigest()==e['input_sequence_sha256'];count+=e['frames'];bytes_+=int(np.prod(e['shape']))*2
 assert count==273465 and bytes_==84665278080
 gate=json.loads((RUN/'frontend_gate.json').read_text());samples=json.loads((R/p['sample_manifest']).read_text())['samples'];assert len(gate['rows'])==len(samples)==120 and gate['status']=='PASS'
 for row,s in zip(gate['rows'],samples):
  assert (row['episode_id'],row['frame_index'])==(s['episode_id'],s['frame_index']) and row['episode_id'] in train
  f=RUN/'gate'/row['file'];assert sha(f)==row['file_sha256'];a=np.load(f);assert a.dtype==np.uint16 and a.shape==(layouts[s['episode_id']]['tokens'],960) and hashlib.sha256(a.tobytes()).hexdigest()==row['feature_sha256']
 loading=json.loads((RUN/'loading.json').read_text());assert loading['frontend_trainable_parameters']==0 and loading['head_state_from_regression_loaded'] is False and all(not x for x in loading['loading'].values())
 history=json.loads((RUN/'training/history.json').read_text());assert len(history)==50
 for i,row in enumerate(history,1):assert row['epoch']==i and row['updates']==i*1899 and row['train_frames']==243000 and row['valid_target_steps']==2362230 and row['validation']['frames']==30465 and row['validation']['valid_target_steps']==296235
 best_epoch=min(history,key=lambda x:(x['validation']['normalized_mse'],x['epoch']))['epoch'];assert best_epoch==report['best_epoch']
 assert report['tail_convention']==p['tail_convention']
 ck={}
 for name in ['initial','best','latest']:
  file=RUN/'training'/(name+'.pt');assert sha(file)==report['checkpoints'][name];ck[name]=torch.load(file,map_location='cpu',weights_only=False);c=ck[name]
  assert c['tail_convention']==p['tail_convention']=='all10_repeat_last_inputs_valid_only_loss_no_decoder_endmask'
  assert c['architecture']=='fixed_adapted_vlm_flow_time_film' and c['decoder_layers']==2 and c['frontend_checkpoint_sha256']==p['backbone_checkpoint_sha256'] and c['vlm_and_adapters_frozen'] and c['ram_cache_receipts_sha256']==cache['receipt_sha256']
  assert c['plan_sha256']==sha(RUN/'plan.json') and 'cpu_rng' in c and 'cuda_rng' in c and len(c['history'])==c['epoch']
  assert c['updates']==c['epoch']*1899 and (bool(c['optimizer']['state'])==(c['epoch']>0))
  model=FlowHead(HeadConfig(**c['model_config']));model.load_state_dict(c['state_dict']);assert sum(x.numel() for x in model.parameters())==763335
 assert ck['initial']['epoch']==0 and ck['latest']['epoch']==50 and ck['best']['epoch']==best_epoch
 old=torch.load(R/p['old_flow_initial'],map_location='cpu',weights_only=False);initial_model=FlowHead(HeadConfig(**ck['initial']['model_config']));initial_model.load_state_dict(ck['initial']['state_dict']);verify_common_initial(initial_model,old['state_dict'])
 assert any(not torch.equal(v,ck['initial']['state_dict'][k]) for k,v in ck['best']['state_dict'].items())
 for i in range(2):assert any(torch.count_nonzero(ck['latest']['state_dict'][f'time_modulation.{i}.{name}'])>0 for name in ['weight','bias'])
 control_predictions=np.load(R/p['control_training_run']/'training/initial_predictions.npy');new_predictions=np.load(RUN/'training/initial_predictions.npy');np.testing.assert_array_equal(new_predictions,control_predictions)
 numeric=R/p['numeric_cache']/'validation';target=np.load(numeric/'actions.npy',mmap_mode='r')[:,0];task_id=np.load(numeric/'task_id.npy',mmap_mode='r');index=json.loads((R/p['cache']/'validation/index.json').read_text());recomputed={}
 for name in ['initial','best','latest']:
  file=RUN/'training'/(name+'_predictions.npy');assert sha(file)==report['prediction_hashes'][name];pred=np.load(file,mmap_mode='r');assert pred.shape==(30465,10,7) and np.isfinite(pred).all()
  first=chunk=0.;valid=0;per={}
  for e in index:
   for i in range(e['start'],e['end']):
    n=min(10,e['end']-i);err=(pred[i,:n].astype(np.float64)-target[i:i+n])**2;first+=float(err[0].sum());chunk+=float(err.sum());valid+=n
    s,c=per.get(int(task_id[i]),(0.,0));per[int(task_id[i])]=(s+float(err[0].mean()),c+1)
  assert valid==296235 and len(per)==40
  v=dict(frames=30465,valid_target_steps=valid,first_mse=first/(30465*7),chunk_mse=chunk/(valid*7),per_task={str(k):dict(frames=n,mse=s/n) for k,(s,n) in per.items()})
  assert abs(v['first_mse']-ck[name]['validation']['normalized_mse'])<1e-6 and abs(v['chunk_mse']-ck[name]['validation']['all_valid_chunk_mse'])<1e-6
  recomputed[name]=v
 result=dict(status='PASS',epochs=50,updates=94950,best_epoch=best_epoch,checkpoint=str((RUN/'training/best.pt').relative_to(R)),checkpoint_sha256=sha(RUN/'training/best.pt'),backbone_checkpoint=p['backbone_checkpoint'],backbone_sha256=p['backbone_checkpoint_sha256'],head_parameters=763335,frontend_trainable_parameters=0,frames=273465,episodes=1693,source_files_independently_rehashed=1693,gate120_persisted_features_verified=True,all_epoch_history_verified=True,validation_predictions=recomputed,ram_feature_bytes=bytes_,cache_seconds=cache['elapsed_seconds'],training_epochs_seconds=sum(x['elapsed_seconds'] for x in history),supervised_seconds=json.loads((RUN/'status.json').read_text())['elapsed_seconds'],elapsed_seconds=time.monotonic()-started,new_gpu_compute=0,new_training_updates=0,new_simulation_episodes=0,limitations=['Full adapted RAM features were released with the process; receipt consistency and executed full-hash assertions are verified, not an independent reread of all79GiB today.', 'Independent saved-prediction rescoring does not rerun the full neural inference;120 persisted adapted gate features support the next online identity check.', 'Fixed-backbone system comparison with46/50 regression; head training histories differ. No closed-loop result yet.'])
 write(out/'independent_acceptance.json',result);print(json.dumps({k:v for k,v in result.items() if k!='validation_predictions'},ensure_ascii=False))
if __name__=='__main__':main()
