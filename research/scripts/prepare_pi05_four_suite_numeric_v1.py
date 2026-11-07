"""Raw training-only numeric stats and split manifests; no images/model/GPU loaded."""
import hashlib,json,time
from collections import Counter
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 start=time.monotonic();split_path=R/'configs/mini_vla_four_suite_demo_split_20261019_v1.json';split=json.loads(split_path.read_text());source=R/'datasets/lerobot/physical-intelligence/libero';out=R/'reports/pi05_four_suite_prepare_20261104';out.mkdir(exist_ok=True);train=split['train'];val=split['validation'];tr={e['episode_id'] for e in train};va={e['episode_id'] for e in val};assert len(tr)==1506 and len(va)==187 and not tr&va
 sums={};receipts=[];arrays={k:[] for k in ['state','actions']};frames=0
 for n,e in enumerate(train,1):
  eid=e['episode_id'];path=source/f'data/chunk-{eid//1000:03d}/episode_{eid:06d}.parquet';table=pq.read_table(path,columns=['state','actions','episode_index','frame_index'],use_threads=False);a=table.to_pydict();assert len(a['state'])==e['length'] and set(a['episode_index'])=={eid};np.testing.assert_array_equal(a['frame_index'],np.arange(e['length']))
  record=dict(episode_id=eid,frames=e['length'],suite=e['suite'],task_id=e['task_id'],numeric_sha256={})
  for k,dim in [('state',8),('actions',7)]:
   x=np.asarray(a[k],np.float32);assert x.shape==(e['length'],dim) and np.isfinite(x).all();arrays[k].append(x);record['numeric_sha256'][k]=hashlib.sha256(x.astype('<f4').tobytes()).hexdigest()
  receipts.append(record);frames+=e['length']
  if n%100==0:write(out/'progress.json',dict(episodes=n,total=1506,frames=frames,elapsed_seconds=time.monotonic()-start))
 assert frames==243000;norm={}
 for k,chunks in arrays.items():
  x=np.concatenate(chunks).astype(np.float64);norm[k]=dict(mean=x.mean(0).tolist(),std=x.std(0).tolist(),q01=np.quantile(x,.01,axis=0).tolist(),q99=np.quantile(x,.99,axis=0).tolist());assert np.isfinite(np.array(list(norm[k].values()))).all()
 write(out/'norm_stats.json',dict(norm_stats=norm));write(out/'training_numeric_receipts.json',dict(episodes=1506,frames=243000,receipts=receipts))
 for part,entries in [('train',train),('validation',val)]:
  manifest=dict(suite='four_suite_joint',source_repo='physical-intelligence/libero',source_revision='a4336d589d589045d1c56423ffdf3b88a0e19b1f',episode_ids=[e['episode_id'] for e in entries],frames=sum(e['length'] for e in entries),episodes=[dict(episode_index=e['episode_id'],length=e['length'],tasks=[e['instruction']]) for e in entries],task_count=40,split_sha256=sha(split_path),partition=part)
  write(out/(part+'_manifest.json'),manifest)
 old=json.loads((R/'reports/spatial_full_manifest.json').read_text());overlap=sorted(set(old['episode_ids'])&va);assert len(overlap)==48
 result=dict(status='DATA_READY_TRAINING_NOT_FROZEN',train_episodes=1506,validation_episodes=187,train_frames=243000,validation_frames=sum(e['length'] for e in val),train_tasks=len({(e['suite'],e['task_id']) for e in train}),train_suites=dict(Counter(e['suite'] for e in train)),validation_suites=dict(Counter(e['suite'] for e in val)),train_validation_disjoint=True,stats_from_raw_train_numeric_only=True,stats_frames=frames,norm_stats_sha256=sha(out/'norm_stats.json'),numeric_receipts_sha256=sha(out/'training_numeric_receipts.json'),old_pi05_training_overlap_with_new_validation=overlap,required_initialization='official pi05_base,not old30k',pending=['normalization through actualpi05 transforms','strict compact trainstate save/reload incloptimizer/RNG','multiGPU memory/gradient/throughput preflight','freeze formal training coverage and resources'],new_optimizer_updates=0,new_simulation_episodes=0,new_gpu_compute=0,elapsed_seconds=time.monotonic()-start)
 write(out/'acceptance.json',result);print(json.dumps(result))
if __name__=='__main__':main()
