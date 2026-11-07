"""Audit all committed cache shards after storage guard stop; no model or simulation."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,time
R=Path(__file__).resolve().parents[1]
RUN=R/'results/mini_vla_modern_cache_20261020_run2'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*2**20),b''):h.update(block)
 return h.hexdigest()
def main():
 start=time.monotonic();out=RUN/'review';out.mkdir(exist_ok=True)
 assert not (out/'partial_integrity.json').exists()
 p=json.loads((R/'configs/mini_vla_modern_cache_20261020_v2.json').read_text());split=json.loads((R/p['split']).read_text());cache=R/p['cache']
 sources={x['path']:x['sha256'] for x in json.loads((R/p['source_integrity']).read_text())['files']}
 rows=[];missing=[]
 for part in ['train','validation']:
  for e in split[part]:
   d=cache/part/f"episode_{e['episode_id']:06d}";f=d/'complete.json'
   if not f.exists():missing.append(dict(part=part,episode=e,partial_files=[dict(name=x.name,bytes=x.stat().st_size) for x in d.glob('*') if x.is_file()] if d.exists() else []));continue
   receipt=json.loads(f.read_text());assert receipt['status']=='PASS' and receipt['binding']==p['cache_binding'] and receipt['episode']==e and receipt['part']==part
   eid=e['episode_id'];source=f'data/chunk-{eid//1000:03d}/episode_{eid:06d}.parquet';assert receipt['source_parquet_sha256']==sources[source]
   assert receipt['shape'][0]==e['length'] and receipt['shape'][2]==960 and receipt['dtype']=='uint16_raw_bfloat16'
   rows.append((part,e,d,receipt))
 def check(row):
  part,e,d,receipt=row
  for name,h in receipt['hashes'].items():assert sha(d/name)==h,str(d/name)
  inputs=json.loads((d/'input_hashes.json').read_text());assert [x['frame_index'] for x in inputs]==list(range(e['length']))
  assert len(inputs)==e['length'] and all(len(x['input_sha256'])==64 for x in inputs)
  return dict(part=part,episode_id=e['episode_id'],frames=e['length'],receipt_sha256=sha(d/'complete.json'),hashes=receipt['hashes'])
 verified=[]
 with ThreadPoolExecutor(max_workers=2) as pool:
  for item in pool.map(check,rows):
   verified.append(item)
   if len(verified)%100==0:
    progress=dict(checked_episodes=len(verified),committed_episodes=len(rows),elapsed_seconds=time.monotonic()-start)
    (out/'partial_audit_progress.json').write_text(json.dumps(progress)+'\n');print(json.dumps(progress),flush=True)
 totals={part:dict(episodes=sum(x['part']==part for x in verified),frames=sum(x['frames'] for x in verified if x['part']==part)) for part in ['train','validation']}
 gate=json.loads((cache/'gate/complete.json').read_text());assert gate['status']=='PASS' and gate['inputs']==120 and gate['max_relative_L2']==0
 for f,h in gate['hashes'].items():assert sha(cache/'gate'/f)==h
 frames=sum(x['frames'] for x in verified);remaining=sum(x['episode']['length'] for x in missing);assert frames+remaining==273465 and len(rows)+len(missing)==1693
 result=dict(status='COMMITTED_SHARDS_PASS_COLLECTION_INCOMPLETE',verified_episodes=len(verified),expected_episodes=1693,verified_frames=frames,expected_frames=273465,missing_frames=remaining,missing_episodes=missing,by_part=totals,verified_receipts=verified,gate120_pass=True,elapsed_seconds=time.monotonic()-start,source_hash_scope='receipt versus frozen source manifest; raw parquet previously checked during extraction, not reread in this audit',new_training_updates=0,new_simulation_episodes=0)
 (out/'partial_integrity.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['verified_receipts','missing_episodes']},indent=2))
if __name__=='__main__':main()
