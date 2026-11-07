"""Full queue/initial-state/action/journal checks and descriptive paired statistics."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from .policy_m8_v1 import sha,R
from .paired_eval_journal_v1 import load as load_journal
from .train_vlm_head_v1 import write

def summarize_difference(rows,left,right):
 pairs=[dict(task_id=k[0],initial_state_index=k[1],left=rows[left][k]['task_success'],right=rows[right][k]['task_success'],difference=int(rows[right][k]['task_success'])-int(rows[left][k]['task_success'])) for k in sorted(rows[left])]
 bytask=[np.mean([p['difference'] for p in pairs if p['task_id']==t]) for t in range(10)]
 samples=np.random.default_rng(20261019).choice(bytask,(10000,10),replace=True).mean(1);d=[p['difference'] for p in pairs]
 return dict(left=left,right=right,wins=d.count(1),losses=d.count(-1),difference=float(np.mean(d)),task_cluster_descriptive_95_interval=np.quantile(samples,[.025,.975]).tolist(),pairs=pairs)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--run',required=True);a=ap.parse_args();pp=Path(a.plan);p=json.loads(pp.read_text());run=Path(a.run);expected={(t,i) for t in range(10) for i in [2,12,22,32,42]};rows={};traces={};metrics={}
 assert json.loads((run/'simulation_preflight/result.json').read_text())['status']=='PASS'
 for arm,directory in [('original',R/p['original_evaluation']),('frozen',run/'frozen/h5'),('lora',run/'lora/h5')]:
  result=json.loads((directory/'result.json').read_text());assert result['status']=='PASS' and len(result['episodes'])==50
  rows[arm]={};traces[arm]={};latencies=[]
  if arm!='original':assert json.loads((run/arm/'online_contract.json').read_text())['status']=='PASS'
  for e in result['episodes']:
   key=e['task_id'],e['initial_state_index'];assert key in expected and key not in rows[arm]
   folder=directory/e['pair_id'];assert sha(folder/'trace.npz')==e['trace_sha256'] and sha(folder/'external.mp4')==e['video_sha256'];assert json.loads((folder/'episode.json').read_text())==e
   with np.load(folder/'trace.npz') as z:t={k:z[k] for k in z.files}
   steps=e['steps'];assert len(t['actions'])==len(t['done'])==steps
   assert bool(t['done'].any())==e['task_success'] and not t['done'][:-1].any();assert e['task_success'] or steps==220
   assert e['policy_calls']==len(t['call_steps']) and np.isfinite(t['actions']).all()
   np.testing.assert_array_equal(t['call_steps'],np.arange(0,steps,5));np.testing.assert_array_equal(t['action_chunks'][:,:5].reshape(-1,7)[:steps],t['actions'])
   if arm!='original':
    assert e['checkpoint_sha256']==p['arms'][arm]['checkpoint_sha256'] and e['plan_sha256']==sha(pp) and e['arm']==arm
    assert sha(folder/'trajectory.jsonl')==e['journal_sha256'];calls,physical=load_journal(folder/'trajectory.jsonl');assert len(calls)==len(t['call_steps']) and len(physical)==steps
    for i,s in enumerate(t['call_steps']):
     c=calls[int(s)];np.testing.assert_array_equal(c['actions'],t['action_chunks'][i]);np.testing.assert_array_equal(c['normalized_actions'],t['normalized_action_chunks'][i]);assert c['observation_sha256']==t['observation_sha256'][i] and c['vlm_input_sha256']==t['vlm_input_sha256'][i]
    for i,step in enumerate(physical):np.testing.assert_array_equal(step['action'],t['actions'][i]);assert step['done']==bool(t['done'][i])
    stats=json.loads((R/p['preprocessing']).read_text())['normalization']['actions'];recovered=np.clip(t['normalized_action_chunks']*np.asarray(stats['scale'],np.float32)+np.asarray(stats['mean'],np.float32),-1,1);np.testing.assert_allclose(t['action_chunks'],recovered,atol=1e-6,rtol=0)
   rows[arm][key]=e;traces[arm][key]=t;latencies.extend(t['latency'][1:])
  assert set(rows[arm])==expected
  metrics[arm]=dict(successes=sum(e['task_success'] for e in rows[arm].values()),episodes=50,steps=sum(e['steps'] for e in rows[arm].values()),calls=sum(e['policy_calls'] for e in rows[arm].values()),rpc_median_ms=float(np.median(latencies)*1000),elapsed_seconds=result['elapsed_seconds'])
 assert metrics['original']['successes']==39
 for key in sorted(expected):
  for arm in ['frozen','lora']:
   assert rows['original'][key]['initial_image_sha256']==rows[arm][key]['initial_image_sha256']
   assert rows['original'][key]['language']==rows[arm][key]['language']
   for field in ['initial_qpos','initial_qvel']:np.testing.assert_allclose(traces['original'][key][field],traces[arm][key][field],atol=1e-7,rtol=0)
 report=dict(status='COMPLETE_REVIEW_REQUIRED',metrics=metrics,paired=summarize_difference(rows,'frozen','lora'),versus_original={a:summarize_difference(rows,'original',a) for a in ['frozen','lora']},all100_queues_and_journals_verified=True,all_initial_states_and_images_matched=True,new_optimizer_updates=0,final2000_accessed=False,video_review='All100 file hashes checked; independent decoding and qualitative review follow terminal event',claim_boundary='One seed one complete continuation epoch; adaptive originalSpatial50 development, not finalLIBERO or unseen-task generalization')
 write(run/'analysis.json',report);print(json.dumps(report['metrics']),flush=True)
if __name__=='__main__':main()
