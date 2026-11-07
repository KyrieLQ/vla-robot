"""Require all2000 final cases, original predicates and complete journals before publishing."""
import argparse,json
from pathlib import Path
import numpy as np
from .final_episode_v1 import load_journal,validate_records
from .final_suite_client_v1 import sha,write,R
def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--run',required=True);a=p.parse_args();plan=json.loads(Path(a.plan).read_text());run=Path(a.run);science=json.loads((R/plan['scientific_plan']).read_text());all_rows=[];suites={};video_count=0;hashes={}
 manifest=json.loads((R/science['case_manifest']).read_text())['cases'];expected={(c['suite'],c['task_id'],c['initial_state_index']):c for c in manifest};assert len(expected)==2000
 for worker,names in enumerate(plan['worker_suites']):
  for suite in names:
   directory=run/f'worker{worker}'/suite;result=json.loads((directory/'result.json').read_text());rows=result['episodes'];assert len(rows)==500
   limit=next(x['max_action_steps'] for x in science['suites'] if x['name']==suite);ids=set();latency=[];partitions={};tasks=[]
   for row in rows:
    key=(suite,row['task_id'],row['initial_state_index']);assert key in expected and key not in ids;ids.add(key);d=directory/f"task{key[1]}_init{key[2]}"
    assert json.loads((d/'episode.json').read_text())==row and sha(d/'trajectory.jsonl')==row['trace_sha256']
    assert row['checkpoint_sha256']==plan['checkpoint_sha256'] and row['scientific_plan_sha256']==sha(R/plan['scientific_plan']) and row['partition']==expected[key]['partition']
    records=load_journal(d/'trajectory.jsonl');summary=validate_records(records,limit)
    for k,v in summary.items():assert row[k]==v
    calls=[r for r in records if r['kind']=='call'];latency.extend(r['latency'] for r in calls[1:])
    if key[2] in [0,25]:assert sha(d/'external.mp4')==row['video_sha256'];video_count+=1
    else:assert row['video_sha256'] is None
    part=partitions.setdefault(row['partition'],dict(successes=0,episodes=0));part['episodes']+=1;part['successes']+=int(row['task_success']);hashes[str(d/'episode.json')]=sha(d/'episode.json')
   for task in range(10):tasks.append(dict(task_id=task,successes=sum(int(r['task_success']) for r in rows if r['task_id']==task),episodes=50))
   suites[suite]=dict(successes=sum(int(r['task_success']) for r in rows),episodes=500,per_task=tasks,partitions=partitions,rpc_median_ms=float(np.median(latency)*1000),rpc_p95_ms=float(np.quantile(latency,.95)*1000),elapsed_seconds=result['elapsed_seconds']);all_rows.extend(rows)
 assert len(all_rows)==2000 and video_count==80
 assert sum(r['partition']=='development' for r in all_rows)==800 and sum(r['partition']=='confirmation' for r in all_rows)==1200
 rng=np.random.default_rng(20261021);boot=[]
 for suite in sorted(suites):
  means=np.array([t['successes']/50 for t in suites[suite]['per_task']]);boot.append(means[rng.integers(0,10,(10000,10))].mean(1))
 score=sum(r['task_success'] for r in all_rows)/2000;ci=np.quantile(np.mean(boot,axis=0),[.025,.975]).tolist()
 write(run/'analysis.json',dict(status='COMPLETE_REVIEW_REQUIRED',selected_model='M6',episodes=2000,successes=sum(r['task_success'] for r in all_rows),macro_suite_success_rate=score,task_cluster_descriptive_interval=ci,suites=suites,videos=80,artifacts_sha256=hashes,claim_boundary='Selected model absolute performance on fourknown suites;includes800development/1200confirmation,not unseen-task claim and no matched finalbaseline improvement claim.'))
if __name__=='__main__':main()
