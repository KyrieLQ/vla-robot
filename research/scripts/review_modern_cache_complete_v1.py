"""Independent complete cache/source/numeric audit; CPU-only, no VLM forward."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib,io,json,time
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from mini_vla.preflight_m8_v2 import make_processor
from mini_vla.vlm_frontend_v1 import prompt_for
R=Path(__file__).resolve().parents[1];RUN=R/'results/mini_vla_modern_cache_20261021_run3'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
 return h.hexdigest()
def main():
 start=time.monotonic();out=RUN/'review';out.mkdir(exist_ok=True);assert not (out/'independent_acceptance.json').exists()
 planpath=R/'configs/mini_vla_modern_cache_resume_20261021_v1.json';p=json.loads(planpath.read_text());c=R/p['cache'];complete=json.loads((c/'complete.json').read_text());split=json.loads((R/p['split']).read_text());pre=json.loads((R/p['preprocessing']).read_text())
 assert complete['frames']==273465 and complete['episodes']==1693 and complete['binding']==p['cache_binding']
 for f,h in json.loads((RUN/'source_manifest.json').read_text()).items():assert sha(R/f)==h,f
 for f,h in p['input_sha256'].items():assert sha(R/f)==h,f
 source={e['path']:e for e in json.loads((R/p['source_integrity']).read_text())['files']}
 trainids={e['episode_id'] for e in split['train']};validids={e['episode_id'] for e in split['validation']};assert len(trainids)==1506 and len(validids)==187 and not trainids&validids
 processor=make_processor(R/p['model_directory']);instructions={e['instruction'] for e in split['train']};assert len(instructions)==40
 prompts={s:prompt_for(processor,s) for s in instructions};tokens={}
 for instruction,prompt in prompts.items():
  z=processor(text=[prompt],images=[[Image.new('RGB',(256,256)),Image.new('RGB',(256,256))]],return_tensors='np',padding=True)
  assert int((z['input_ids']==49190).sum())==128;tokens[instruction]=z['input_ids'][0][z['attention_mask'][0].astype(bool)].astype(np.int32)
 samples=json.loads((R/p['sample_manifest']).read_text())['samples'];gate_by_eid={s['episode_id']:(i,s['frame_index']) for i,s in enumerate(samples)}
 previous=json.loads((R/'results/mini_vla_m6_prepare_20261019_run3/review/independent_acceptance.json').read_text());arrays={};indexes={};jobs=[]
 for part in ['train','validation']:
  assert sha(c/part/'index.json')==complete['parts'][part]['index_sha256'];idx=json.loads((c/part/'index.json').read_text());assert len(idx)==len(split[part]);indexes[part]=idx
  arrays[part]={k:np.load(R/p['numeric_cache']/part/(k+'.npy'),mmap_mode='r') for k in ['state','actions','episode_id','task_id']}
  for k in arrays[part]:
   rel=p['numeric_cache']+'/'+part+'/'+k+'.npy';assert sha(R/rel)==previous['cache_all_files_sha256'][rel]
  offset=0
  for e,entry in zip(split[part],idx):
   assert entry['episode_id']==e['episode_id'] and entry['suite']==e['suite'] and entry['task_id']==e['task_id']
   assert entry['start']==offset and entry['end']==offset+e['length'];offset+=e['length'];jobs.append((part,e,entry))
  assert offset==(243000 if part=='train' else 30465)
 def audit(job):
  part,e,entry=job;eid=e['episode_id'];directory=c/entry['directory'];receipt=json.loads((directory/'complete.json').read_text());assert sha(directory/'complete.json')==entry['receipt_sha256']
  assert receipt['status']=='PASS' and receipt['binding']==p['cache_binding'] and receipt['episode']==e
  for name,h in receipt['hashes'].items():assert sha(directory/name)==h,str(directory/name)
  hidden=np.load(directory/'hidden.npy',mmap_mode='r');ids=np.load(directory/'token_ids.npy');assert hidden.dtype==np.uint16 and hidden.shape==(e['length'],len(tokens[e['instruction']]),960);np.testing.assert_array_equal(ids,tokens[e['instruction']])
  for first in range(0,len(hidden),16):assert not np.any((hidden[first:first+16]&0x7f80)==0x7f80),'NaN/Inf BF16 bits'
  source_name=f'data/chunk-{eid//1000:03d}/episode_{eid:06d}.parquet';raw=R/p['dataset_root']/source_name;assert raw.stat().st_size==source[source_name]['bytes'];assert sha(raw)==receipt['source_parquet_sha256']==source[source_name]['sha256']
  table=pq.read_table(raw,columns=['episode_index','frame_index','state','actions','image','wrist_image'],use_threads=False);assert table.num_rows==e['length'];assert set(table['episode_index'].to_pylist())=={eid};assert table['frame_index'].to_pylist()==list(range(e['length']))
  a=arrays[part];first,last=entry['start'],entry['end'];assert np.all(a['episode_id'][first:last]==eid)
  gid=['libero_spatial','libero_object','libero_goal','libero_10'].index(e['suite'])*10+e['task_id'];assert np.all(a['task_id'][first:last]==gid)
  for name in ['state','actions']:
   rawnum=np.asarray(table[name].to_pylist(),dtype=np.float32);stat=pre['normalization'][name];normalized=(rawnum-np.asarray(stat['mean'],np.float32))/np.asarray(stat['scale'],np.float32);actual=a[name][first:last] if name=='state' else a[name][first:last,0];np.testing.assert_array_equal(normalized,actual)
  inputs=json.loads((directory/'input_hashes.json').read_text());assert [x['frame_index'] for x in inputs]==list(range(e['length']))
  selected={0,e['length']//2,e['length']-1}
  if eid in gate_by_eid:selected.add(gate_by_eid[eid][1])
  for i in selected:
   row=table.slice(i,1).to_pylist()[0];pieces=[]
   for name in ['image','wrist_image']:
    with Image.open(io.BytesIO(row[name]['bytes'])) as im:
     rgb=im.convert('RGB');assert rgb.size==(256,256);pieces.append(rgb.tobytes())
   actual=hashlib.sha256(b''.join(pieces)+prompts[e['instruction']].encode()).hexdigest();assert actual==inputs[i]['input_sha256']
  gatechecked=0
  if eid in gate_by_eid:
   gate_index,frame=gate_by_eid[eid];reference=np.load(c/'gate'/f'input_{gate_index:03d}.npy');np.testing.assert_array_equal(hidden[frame],reference);gatechecked=1
  steps=sum(min(10,e['length']-i) for i in range(e['length']))
  return dict(part=part,episode_id=eid,frames=e['length'],valid_K10_steps=steps,payload_bytes=hidden.nbytes,raw_input_checks=len(selected),production_to_gate_exact=gatechecked,receipt_sha256=entry['receipt_sha256'])
 rows=[]
 with ThreadPoolExecutor(max_workers=2) as pool:
  for row in pool.map(audit,jobs):
   rows.append(row)
   if len(rows)%100==0:
    progress=dict(checked_episodes=len(rows),total_episodes=1693,checked_frames=sum(x['frames'] for x in rows),elapsed_seconds=time.monotonic()-start);(out/'progress.json').write_text(json.dumps(progress)+'\n');print(json.dumps(progress),flush=True)
 gate=json.loads((c/'gate/complete.json').read_text());assert gate['status']=='PASS' and gate['inputs']==120 and gate['max_relative_L2']==0
 for name,h in gate['hashes'].items():assert sha(c/'gate'/name)==h
 assert len(rows)==1693 and sum(x['frames'] for x in rows)==273465 and sum(x['payload_bytes'] for x in rows)==84665278080 and sum(x['production_to_gate_exact'] for x in rows)==120
 parts={part:dict(episodes=sum(x['part']==part for x in rows),frames=sum(x['frames'] for x in rows if x['part']==part),valid_K10_steps=sum(x['valid_K10_steps'] for x in rows if x['part']==part)) for part in ['train','validation']};assert parts['train']['valid_K10_steps']==2362230 and parts['validation']['valid_K10_steps']==296235
 result=dict(status='PASS',episodes=1693,frames=273465,parts=parts,payload_bytes=84665278080,all_source_and_output_hashes_verified=True,all_BF16_values_finite=True,all_numeric_arrays_bitwise_equal_to_raw_training_transform=True,train_validation_disjoint=True,input_pixel_hash_scope='first/middle/last everyepisode plus originalgateframes;allsourceparquets hash verified,not fullVLMrecompute',input_pixel_hash_checks=sum(x['raw_input_checks'] for x in rows),production_cache_exact_vs_original120_gate=True,cache_manifest_sha256=sha(c/'complete.json'),gate_sha256=sha(c/'gate/complete.json'),verified_receipts=rows,elapsed_seconds=time.monotonic()-start,new_training_updates=0,new_simulation_episodes=0,confirmation_accessed=False)
 (out/'independent_acceptance.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='verified_receipts'},indent=2))
if __name__=='__main__':main()
