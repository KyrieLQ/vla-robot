"""Full queue/initial-state/action/journal checks and descriptive paired statistics."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
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
 for arm,directory in [('regression',R/p['baseline_evaluation']),('control_flow',R/p['control_flow_evaluation']),('flow',run/'flow/h5')]:
  result=json.loads((directory/'result.json').read_text());assert result['status']=='PASS' and len(result['episodes'])==50
  rows[arm]={};traces[arm]={};latencies=[]
  if arm=='flow':assert json.loads((run/arm/'online_contract.json').read_text())['status']=='PASS'
  for e in result['episodes']:
   key=e['task_id'],e['initial_state_index'];assert key in expected and key not in rows[arm]
   folder=directory/e['pair_id'];assert sha(folder/'trace.npz')==e['trace_sha256'] and sha(folder/'external.mp4')==e['video_sha256'];assert json.loads((folder/'episode.json').read_text())==e
   with np.load(folder/'trace.npz') as z:t={k:z[k] for k in z.files}
   steps=e['steps'];assert len(t['actions'])==len(t['done'])==steps
   assert bool(t['done'].any())==e['task_success'] and not t['done'][:-1].any();assert e['task_success'] or steps==220
   assert e['policy_calls']==len(t['call_steps']) and np.isfinite(t['actions']).all()
   np.testing.assert_array_equal(t['call_steps'],np.arange(0,steps,5));np.testing.assert_array_equal(t['action_chunks'][:,:5].reshape(-1,7)[:steps],t['actions'])
   if arm=='flow':
    assert e['checkpoint_sha256']==p['arms'][arm]['checkpoint_sha256'] and e['plan_sha256']==sha(pp) and e['arm']==arm
    assert sha(folder/'trajectory.jsonl')==e['journal_sha256'];calls,physical=load_journal(folder/'trajectory.jsonl');assert len(calls)==len(t['call_steps']) and len(physical)==steps
    for i,s in enumerate(t['call_steps']):
     c=calls[int(s)];np.testing.assert_array_equal(c['actions'],t['action_chunks'][i]);np.testing.assert_array_equal(c['normalized_actions'],t['normalized_action_chunks'][i]);assert c['observation_sha256']==t['observation_sha256'][i] and c['vlm_input_sha256']==t['vlm_input_sha256'][i]
    for i,step in enumerate(physical):np.testing.assert_array_equal(step['action'],t['actions'][i]);assert step['done']==bool(t['done'][i])
    stats=json.loads((R/p['preprocessing']).read_text())['normalization']['actions'];recovered=np.clip(t['normalized_action_chunks']*np.asarray(stats['scale'],np.float32)+np.asarray(stats['mean'],np.float32),-1,1);np.testing.assert_allclose(t['action_chunks'],recovered,atol=1e-6,rtol=0)
   if arm in ['control_flow','flow']:
    assert len(t['noise_seed'])==len(t['noise_sha256'])==len(t['call_steps'])
    for call,(saved_seed,saved_hash) in enumerate(zip(t['noise_seed'],t['noise_sha256'])):
     seed=int.from_bytes(hashlib.sha256(f'20261020:{key[0]}:{key[1]}:{call}'.encode()).digest()[:8],'little')%(2**63-1)
     assert int(saved_seed)==seed
     noise=torch.randn((10,7),generator=torch.Generator().manual_seed(seed),dtype=torch.float32)
     assert saved_hash==hashlib.sha256(noise.numpy().tobytes()).hexdigest()
     if arm=='flow':assert calls[int(t['call_steps'][call])]['noise_seed']==seed and calls[int(t['call_steps'][call])]['noise_sha256']==saved_hash
   rows[arm][key]=e;traces[arm][key]=t;latencies.extend(t['latency'][1:])
  assert set(rows[arm])==expected
  metrics[arm]=dict(successes=sum(e['task_success'] for e in rows[arm].values()),episodes=50,steps=sum(e['steps'] for e in rows[arm].values()),calls=sum(e['policy_calls'] for e in rows[arm].values()),rpc_median_ms=float(np.median(latencies)*1000),elapsed_seconds=result['elapsed_seconds'])
 assert metrics['regression']['successes']==46 and metrics['control_flow']['successes']==22
 for key in sorted(expected):
  for arm in ['control_flow','flow']:
   assert rows['regression'][key]['initial_image_sha256']==rows[arm][key]['initial_image_sha256']
   assert rows['regression'][key]['language']==rows[arm][key]['language']
   for field in ['initial_qpos','initial_qvel']:np.testing.assert_allclose(traces['regression'][key][field],traces[arm][key][field],atol=1e-7,rtol=0)
 report=dict(status='COMPLETE_REVIEW_REQUIRED',metrics=metrics,paired_vs_regression=summarize_difference(rows,'regression','flow'),paired_vs_control_flow=summarize_difference(rows,'control_flow','flow'),all50_queues_journals_noise_verified=True,all_initial_states_and_images_matched=True,new_optimizer_updates=0,final2000_accessed=False,video_review='All50newvideo hashes checked; independent complete decoding follows terminal event',claim_boundary='One seed,originalSpatial50 development. Two vs one Decoder layers share fixed backbone,data,tail convention and50epoch schedule; extra independent block increases parameter count; regression is a contextual reference. Parallel evaluation preserves the original per-task reset and per-call noise rules.')
 write(run/'analysis.json',report);print(json.dumps(report['metrics']),flush=True)
if __name__=='__main__':main()
