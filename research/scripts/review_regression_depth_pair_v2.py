"""Independent complete50epoch/metadata/full saved prediction audit; no RAM recreation."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from mini_vla.vlm_action_heads_v1 import HeadConfig
from mini_vla.regression_depth_pair_v1 import RegressionDepthHead,verify_initial
from mini_vla.train_vlm_head_v1 import sha,write
R=Path(__file__).resolve().parents[1]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args();RUN=Path(a.run)
 started=time.monotonic();torch.set_num_threads(2);p=json.loads((RUN/'plan.json').read_text());report=json.loads((RUN/'report.json').read_text());out=RUN/'review';out.mkdir(exist_ok=True)
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
 loading=json.loads((RUN/'loading.json').read_text());assert loading['frontend_trainable_parameters']==0 and loading['head_from_regression_endpoint'] is False and all(not x for x in loading['loading'].values())
 assert report['epochs_per_arm']==50 and report['updates_per_arm']==94950
 probe=json.loads((RUN/'head_probe.json').read_text());assert probe['status']=='PASS'
 if probe['selected_mode']=='dual_gpu':assert probe['dual_gpu_numerically_admitted'] and probe['parameter_max_error']<=1e-6
 else:assert 'singleGPU' in probe['reason'] or 'oneGPU' in probe['reason']
 bundle=torch.load(RUN/'paired_latest.pt',map_location='cpu',weights_only=False);assert sha(RUN/'paired_latest.pt')==report['paired_latest_sha256'] and bundle['epoch']==50 and bundle['updates']==94950 and 'cpu_rng' in bundle and 'cuda_rng' in bundle
 assert bundle['receipt_sha256']==cache['receipt_sha256'] and bundle['frontend_checkpoint_sha256']==p['backbone_checkpoint_sha256']
 numeric=R/p['numeric_cache']/'validation';target=np.load(numeric/'actions.npy',mmap_mode='r')[:,0];index=json.loads((R/p['cache']/'validation/index.json').read_text());results={};initials={}
 for name,layers,count in [('depth1',1,398663),('depth2',2,663239)]:
  arm=RUN/name;history=json.loads((arm/'history.json').read_text());assert history==bundle['history'][name] and len(history)==50
  for i,row in enumerate(history,1):
   assert row['epoch']==i and row['updates']==i*1899 and row['train_frames']==243000 and row['valid_target_steps']==2362230 and row['validation']['frames']==30465
   assert row['sample_order_sha256']==bundle['history']['depth1'][i-1]['sample_order_sha256'] and len(row['sample_order_sha256'])==64
  best_epoch=min(history,key=lambda x:(x['validation']['normalized_mse'],x['epoch']))['epoch'];assert best_epoch==report['arms'][name]['best_epoch']==bundle['best'][name]['epoch']
  states=bundle['optimizers'][name]['state'];assert states and all(int(v['step'])==94950 for v in states.values())
  rec={}
  for point in ['initial','best','latest']:
   file=arm/(point+'.pt');assert sha(file)==report['arms'][name]['checkpoints'][point];ck=torch.load(file,map_location='cpu',weights_only=False)
   assert ck['architecture']=='fixed_adapted_vlm_regression_depth' and ck['decoder_layers']==layers and ck['frontend_checkpoint_sha256']==p['backbone_checkpoint_sha256'] and ck['vlm_and_adapters_frozen'] and ck['ram_cache_receipts_sha256']==cache['receipt_sha256']
   expected_epoch=0 if point=='initial' else 50 if point=='latest' else best_epoch;assert ck['epoch']==expected_epoch and ck['updates']==expected_epoch*1899
   model=RegressionDepthHead(HeadConfig(**ck['model_config']),layers);model.load_state_dict(ck['state_dict'],strict=True);assert sum(v.numel() for v in model.parameters())==count
   if point=='initial':initials[name]=model
   else:
    source=bundle['states'][name] if point=='latest' else bundle['best'][name]['state'];assert all(torch.equal(v,source[k]) for k,v in ck['state_dict'].items())
   predfile=arm/(point+'_predictions.npy');assert sha(predfile)==report['arms'][name]['prediction_hashes'][point];pred=np.load(predfile,mmap_mode='r');assert pred.shape==(30465,10,7) and np.isfinite(pred).all()
   first=chunk=0.;valid=0
   for e in index:
    for i in range(e['start'],e['end']):
     n=min(10,e['end']-i);err=(pred[i,:n].astype(np.float64)-target[i:i+n])**2;first+=float(err[0].sum());chunk+=float(err.sum());valid+=n
   assert valid==296235;v=dict(frames=30465,valid_target_steps=valid,first_mse=first/(30465*7),chunk_mse=chunk/(valid*7));assert abs(v['first_mse']-ck['validation']['normalized_mse'])<1e-6 and abs(v['chunk_mse']-ck['validation']['all_valid_chunk_mse'])<1e-6;rec[point]=v
  results[name]=dict(parameters=count,best_epoch=best_epoch,checkpoint=str((arm/'best.pt').relative_to(R)),checkpoint_sha256=sha(arm/'best.pt'),validation_predictions=rec)
 old=torch.load(R/p['regression_initial'],map_location='cpu',weights_only=False);verify_initial(initials['depth1'],initials['depth2'],old['state_dict'])
 result=dict(status='PASS',arms=results,epochs_per_arm=50,updates_per_arm=94950,shared_initial_parameters_exact=True,same_complete_sample_order_per_epoch=True,paired_optimizer_RNG_checkpoint_verified=True,full_source_files_rehashed=1693,feature_receipts_sha256=cache['receipt_sha256'],ram_cache_bytes=84665278080,frontend_trainable_parameters=0,selected_training_mode=report['selected_mode'],cache_seconds=cache['elapsed_seconds'],supervised_seconds=json.loads((RUN/'status.json').read_text())['elapsed_seconds'],elapsed_seconds=time.monotonic()-started,new_training_updates=0,new_simulation_episodes=0,limitations=['Full79GiB features were RAM-only and released;receipts and executed doublehash assertions verified,not freshly reextracted.', 'Full saved-prediction rescoring does not rerun all neural forwards;online120 parity remains required.', 'No new closed-loop success obtained by training audit;old46/50 only historical reference.'])
 write(out/'independent_acceptance.json',result);print(json.dumps({k:v for k,v in result.items() if k!='arms'}))
if __name__=='__main__':main()
