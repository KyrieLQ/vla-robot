"""Audit every committed case and the exact interrupted prefix; no partial final score."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
from mini_vla.final_episode_v2 import load_journal,validate_records,inspect_prefix
R=Path(__file__).resolve().parents[1];RUN=R/'results/mini_vla_final2000_20261021_run1'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def main():
 start=time.monotonic();out=RUN/'review';out.mkdir(exist_ok=True);plan=json.loads((R/'configs/mini_vla_final2000_20261021_v1.json').read_text());science=json.loads((R/plan['scientific_plan']).read_text());science_sha=sha(R/plan['scientific_plan'])
 for f,h in json.loads((RUN/'source_manifest.json').read_text()).items():assert sha(R/f)==h,f
 cases=[];counts={};videos=0;decoded=0;partial=[];empty=[]
 manifest={(c['suite'],c['task_id'],c['initial_state_index']):c for c in json.loads((R/science['case_manifest']).read_text())['cases']}
 for wi,suites in enumerate(plan['worker_suites']):
  for suite in suites:
   folder=RUN/f'worker{wi}'/suite;limit=next(s['max_action_steps'] for s in science['suites'] if s['name']==suite);complete_ids=[];counts[suite]=0
   for d in sorted(folder.glob('task*_init*')):
    if not (d/'trajectory.jsonl').exists():
     assert not (d/'episode.json').exists();empty.append(str(d.relative_to(RUN)));continue
    raw=(d/'trajectory.jsonl').read_bytes();assert raw.endswith(b'\n')
    records=load_journal(d/'trajectory.jsonl');prefix=inspect_prefix(records,limit);identity=json.loads((d/'start.json').read_text());key=(suite,identity['task_id'],identity['initial_state_index']);assert key in manifest
    assert identity['checkpoint_sha256']==plan['checkpoint_sha256'] and identity['scientific_plan_sha256']==science_sha and identity['partition']==manifest[key]['partition']
    for r in records:
     if r['kind']=='call':assert len(r['input_sha256'])==64 and r['latency']>=0
    complete=(d/'episode.json').exists()
    if complete:
     row=json.loads((d/'episode.json').read_text());assert prefix['complete'];summary=validate_records(records,limit)
     for k,v in summary.items():assert row[k]==v
     for k,v in identity.items():assert row[k]==v
     assert row['trace_sha256']==sha(d/'trajectory.jsonl');assert row['environment_seed']==7 and row['action_horizon']==10 and row['execute_steps']==5
     complete_ids.append(key[1]*50+key[2]);counts[suite]+=1
     if key[2] in [0,25]:
      assert sha(d/'external.mp4')==row['video_sha256'];n=0
      with imageio.get_reader(str(d/'external.mp4'),ffmpeg_params=['-threads','1']) as reader:
       for frame in reader:assert frame.shape==(256,256,3);n+=1
      assert n==row['steps']+1;decoded+=n;videos+=1
     else:assert row['video_sha256'] is None
    else:
     assert not prefix['complete'];partial.append(dict(case=str(d.relative_to(RUN)),steps=prefix['steps'],calls=prefix['calls'],pending_commands=len(prefix['pending_actions'])))
    files={str(f.relative_to(RUN)):sha(f) for f in d.iterdir() if f.is_file()};cases.append(dict(case=str(d.relative_to(RUN)),status='COMPLETE' if complete else 'PARTIAL',files=files))
   assert sorted(complete_ids)==list(range(counts[suite])),'Noncontiguous committed final cases'
 assert sum(counts.values())==161 and len(partial)==1 and partial[0]['steps']==51 and partial[0]['pending_commands']==4
 result=dict(status='AUDITED_PARTIAL_PRESERVE_OUTCOMES',source_run=str(RUN.relative_to(R)),checkpoint_sha256=plan['checkpoint_sha256'],scientific_plan_sha256=science_sha,complete_cases=161,partial_cases=1,complete_by_suite=counts,partial=partial,empty_pre_policy_directories=empty,verified_videos=videos,decoded_video_frames=decoded,cases=cases,new_policy_calls=0,new_simulation_episodes=0,elapsed_seconds=time.monotonic()-start,claim_boundary='Coverage and immutable recovery evidence only; no complete2000 or finalsuccess score.')
 p=R/'configs/mini_vla_final_resume_manifest_20261021_v1.json';assert not p.exists();p.write_text(json.dumps(result,indent=2)+'\n');(out/'partial_acceptance.json').write_text(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2)+'\n');print({k:v for k,v in result.items() if k!='cases'})
if __name__=='__main__':main()
