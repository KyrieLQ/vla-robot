"""Fresh NumPy audit and task-level seam summaries for the completed noise control."""
import csv,hashlib,json,time
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1]
RUN=R/'results/mini_vla_m7_flow_noise_dev50_20261103_run1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 started=time.monotonic();p=json.loads((RUN/'plan.json').read_text());a=json.loads((RUN/'review/independent_acceptance.json').read_text());seams=json.loads((RUN/'noise_seam_analysis.json').read_text());assert a['status']=='PASS'
 rows=[];recomputed_seams=[]
 for task in range(10):
  for init in [2,12,22,32,42]:
   pair=f'task{task}_init{init}';new=RUN/'flow/h5'/pair;old=R/p['fresh_flow_evaluation']/pair
   record=[];traces=[]
   for folder in [old,new]:
    e=json.loads((folder/'episode.json').read_text());assert (e['task_id'],e['initial_state_index'])==(task,init)
    assert sha(folder/'trace.npz')==e['trace_sha256'] and sha(folder/'external.mp4')==e['video_sha256']
    z=np.load(folder/'trace.npz');assert e['task_success']==bool(z['done'].any())
    assert not z['done'][:-1].any() and len(z['done'])==e['steps'];record.append(e);traces.append(z)
   assert record[0]['initial_image_sha256']==record[1]['initial_image_sha256']
   np.testing.assert_allclose(traces[0]['normalized_action_chunks'][0],traces[1]['normalized_action_chunks'][0],atol=1e-6,rtol=0)
   calls=[json.loads(x) for x in (new/'trajectory.jsonl').read_text().splitlines() if json.loads(x)['kind']=='call']
   assert len({c['noise_seed'] for c in calls})==len({c['noise_sha256'] for c in calls})==1
   locked=np.array([x['normalized_actions'] for x in calls]);fresh=np.array([x['shadow_fresh_normalized_actions'] for x in calls])
   measure=dict(task_id=task,init=init,first_call_matches_baseline=True,
    same_observation_noise_mse=float(np.mean((locked[1:,:5]-fresh[1:,:5])**2)),
    fixed_overlap_mse=float(np.mean((locked[:-1,5:]-locked[1:,:5])**2)),
    shadow_fresh_overlap_mse=float(np.mean((locked[:-1,5:]-fresh[1:,:5])**2)))
   recorded=next(x for x in seams['rows'] if (x['task_id'],x['init'])==(task,init))
   for k in ['same_observation_noise_mse','fixed_overlap_mse','shadow_fresh_overlap_mse']:assert abs(measure[k]-recorded[k])<1e-12
   recomputed_seams.append(measure)
   rows.append(dict(task_id=task,initial_state_index=init,pair=pair,fresh_success=int(record[0]['task_success']),fixed_success=int(record[1]['task_success']),fresh_steps=record[0]['steps'],fixed_steps=record[1]['steps']))
 d=np.array([x['fixed_success']-x['fresh_success'] for x in rows]);bytask=d.reshape(10,5).mean(1)
 ci=np.quantile(np.random.default_rng(20261019).choice(bytask,(10000,10),replace=True).mean(1),[.025,.975]).tolist()
 assert d.sum()==-14 and int((d==1).sum())==2 and int((d==-1).sum())==16
 assert sum(x['fixed_success'] for x in rows)==7 and sum(x['fresh_success'] for x in rows)==21
 summaries={}
 for key in ['baseline_executed_boundary','fixed_executed_boundary']:
  summaries[key]={metric:float(np.mean([x[key][metric] for x in seams['rows'] if x[key][metric] is not None])) for metric in ['normalized_mse','gripper_switch_rate']}
 result=dict(status='PASS',episodes=50,fresh=21,fixed=7,wins=2,losses=16,difference=float(d.mean()),task_cluster_descriptive_95_interval=ci,
  first_calls_and_initial_images_match=True,persisted_noise_constant=True,seams_independently_recomputed=True,
  executed_boundary_episode_means=summaries,elapsed_seconds=time.monotonic()-started,new_simulation_episodes=0,new_optimizer_updates=0,
  interpretation='This rejects this episode-fixed-noise intervention as an improvement. It does not prove all temporal correlation/warm starts fail or prove unique cause of original Flow gap.')
 (RUN/'review/followup_audit.json').write_text(json.dumps(result,indent=2)+'\n')
 with (RUN/'review/paired_cases.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(json.dumps(result),flush=True)
if __name__=='__main__':main()
