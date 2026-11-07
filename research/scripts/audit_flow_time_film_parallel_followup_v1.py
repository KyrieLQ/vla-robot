"""Independent task-partition/barrier/provenance and raw-outcome recomputation."""
import csv,hashlib,json,time
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1]
RUN=R/'results/mini_vla_m7_flow_time_film_dev50_20261104_run1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 start=time.monotonic();p=json.loads((RUN/'plan.json').read_text());parent_hash=sha(RUN/'plan.json');gate=json.loads((RUN/'parallel_gates.json').read_text());assert gate['status']=='PASS' and 2<=gate['workers']<=4
 release=(RUN/'release_evaluation.json').stat().st_mtime_ns;seen=set();rows=[]
 for i,tasks in enumerate(gate['task_partitions']):
  directory=RUN/'workers'/str(i);shard=json.loads((directory/'plan.json').read_text());assert shard['evaluation']['task_ids']==tasks and shard['scientific_plan_sha256']==parent_hash
  assert (directory/'ready.json').stat().st_mtime_ns<=release
  assert (directory/'server_metadata_gate.json').stat().st_mtime_ns<=(directory/'ready.json').stat().st_mtime_ns
  from mini_vla.flow_time_film_metadata_v1 import validate_metadata
  mg=json.loads((directory/'server_metadata_gate.json').read_text());assert mg['status']=='PASS' and mg['new_policy_calls']==0;validate_metadata(mg['metadata'],p)
  contract=json.loads((directory/'flow/online_contract.json').read_text());assert contract['status']=='PASS' and contract['maximum_action_error']==0
  result=json.loads((directory/'flow/h5/result.json').read_text());assert len(result['episodes'])==len(tasks)*5
  for e in result['episodes']:
   key=e['task_id'],e['initial_state_index'];assert key not in seen and e['task_id'] in tasks;seen.add(key)
   folder=directory/'flow/h5'/e['pair_id'];canonical=RUN/'flow/h5'/e['pair_id'];assert canonical.is_symlink() and canonical.resolve()==folder.resolve()
   assert (folder/'episode_start.json').stat().st_mtime_ns>=release
   assert e['plan_sha256']==parent_hash and sha(folder/'trace.npz')==e['trace_sha256']
   trace=np.load(folder/'trace.npz');assert bool(trace['done'][-1])==e['task_success'] and not trace['done'][:-1].any()
   control=R/p['control_flow_evaluation']/e['pair_id'];baseline=json.loads((control/'episode.json').read_text());other=np.load(control/'trace.npz');assert e['initial_image_sha256']==baseline['initial_image_sha256']
   for field in ['initial_qpos','initial_qvel']:np.testing.assert_allclose(trace[field],other[field],atol=1e-7,rtol=0)
   rows.append(dict(task_id=key[0],init=key[1],control_success=int(baseline['task_success']),time_film_success=int(e['task_success']),control_steps=baseline['steps'],time_film_steps=e['steps'],worker=i))
 assert seen=={(t,j) for t in range(10) for j in [2,12,22,32,42]};rows.sort(key=lambda x:(x['task_id'],x['init']))
 d=np.array([x['time_film_success']-x['control_success'] for x in rows]);assert sum(x['control_success'] for x in rows)==28
 bytask=d.reshape(10,5).mean(1);ci=np.quantile(np.random.default_rng(20261019).choice(bytask,(10000,10),replace=True).mean(1),[.025,.975]).tolist()
 assert int(d.sum())==sum(x['time_film_success'] for x in rows)-28
 result=dict(status='PASS',episodes=50,workers=gate['workers'],control_successes=28,time_film_successes=sum(x['time_film_success'] for x in rows),wins=int((d==1).sum()),losses=int((d==-1).sum()),task_cluster_descriptive_95_interval=ci,all_worker_gates_before_release=True,all_policy_samples_after_release=True,canonical_links_to_original_worker_records=True,all_initial_states_and_images_match=True,elapsed_seconds=time.monotonic()-start,new_simulation_episodes=0,new_training_updates=0)
 (RUN/'review/followup_audit.json').write_text(json.dumps(result,indent=2)+'\n')
 with (RUN/'review/paired_cases.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(json.dumps(result))
if __name__=='__main__':main()
