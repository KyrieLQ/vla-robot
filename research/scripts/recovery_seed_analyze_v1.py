"""All130 raw branches and fixed20x3 paired outcome matrix before any inference."""
import argparse,collections,datetime,json,os,pathlib,time
import numpy as np
from paired_live_audit_v1 import arrays,sha,check_saved_branch,verify_audit_bindings
ROOT=pathlib.Path(os.environ['PROJECT_ROOT'])
def statistics(a,r,b,task_ids):
    assert a.shape==r.shape==(20,3) and np.isin(a,[0,1]).all() and np.isin(r,[0,1]).all();d=r.astype(int)-a.astype(int);varying=np.ptp(d,axis=1)>0;ab=sum(int(a[i,0])!=int(value) for i,value in b.items());count=int(varying.sum())
    if ab:category='INCONCLUSIVE_IMPLEMENTATION_NOISE'
    elif count>=8:category='HIGH_OBSERVED_SEED_VARIATION'
    elif count<=2:category='LOW_OBSERVED_SEED_VARIATION'
    else:category='INCONCLUSIVE_INTERMEDIATE_VARIATION'
    task_means=np.array([d[np.asarray(task_ids)==t].mean() for t in range(10)]);rng=np.random.default_rng(2026101410);distribution=task_means[rng.integers(0,10,(10000,10))].mean(1)
    return dict(interpretation=category,roots_with_varying_gain=count,roots_total=20,replicates=3,a_b_success_discordance=ab,a_b_controls=len(b),continue_successes_by_seed=a.sum(0).tolist(),resample_successes_by_seed=r.sum(0).tolist(),observed_mean_paired_gain=float(d.mean()),task_cluster_descriptive_interval=np.quantile(distribution,[.025,.975]).tolist(),mean_within_root_sample_variance=float(d.var(axis=1,ddof=1).mean()),per_root_gains=d.tolist(),per_root_sample_variance=d.var(axis=1,ddof=1).tolist(),observed_all_three_wins=int(np.all(d==1,axis=1).sum()),observed_all_three_losses=int(np.all(d==-1,axis=1).sum()))
def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--run',required=True);a=p.parse_args();plan=json.loads(pathlib.Path(a.plan).read_text());run=pathlib.Path(a.run);start=time.monotonic();roots={r['root_id']:r for r in plan['policy_seed_roots']};expected={(j['root_id'],j['method']) for j in plan['jobs']};rows={};hashes=[];model_calls=0;physics_controls=0;maxmodel=0.;maxphysics=0.
    for worker in (0,1):
        result=json.loads((run/f'workers/worker{worker}/result.json').read_text());assert result['status']=='PASS' and result['completed_branches']==65
        for row in result['branches']:
            key=row['root_id'],row['method'];assert key in expected and key not in rows;root=roots[row['root_id']];directory=pathlib.Path(row['output']);assert root['worker']==worker and row['verification_status']=='PASS';tr,obs,oracle,meta=check_saved_branch(directory,row,root,plan);verify_audit_bindings(directory,row);assert bool(oracle['goal_predicate'][:,0].any())==row['task_success'];rows[key]=row
            audit=json.loads((directory/'model_replay.json').read_text());model_calls+=audit['calls_checked'];maxmodel=max(maxmodel,audit['max_action_chunk_error'])
            if row['method'] in ('continue_a','continue_b'):
                audit=json.loads((directory/'physics_replay.json').read_text());physics_controls+=1;maxphysics=max(maxphysics,max(audit['physical_max_errors'].values()))
            hashes.append(dict(root_id=key[0],method=key[1],sha256={q.name:sha(q) for q in directory.iterdir() if q.is_file() and not q.name.endswith('.tmp')}))
    assert set(rows)==expected and len(rows)==130 and physics_controls==70;lookup={v:i for i,v in enumerate(plan['selected_pair_ids'])};ca=np.zeros((20,3),dtype=int);res=ca.copy();cb={};taskids=[None]*20;trajectory=[]
    for rid,root in roots.items():
        i=lookup[root['pair_id']];j=root['replicate_index'];taskids[i]=root['task_id'];ca[i,j]=rows[rid,'continue_a']['task_success'];res[i,j]=rows[rid,'resample5']['task_success'];assert rows[rid,'continue_a']['root_image_sha256']==rows[rid,'resample5']['root_image_sha256']
        if (rid,'continue_b') in rows:
            assert j==0;cb[i]=rows[rid,'continue_b']['task_success'];a_trace=arrays(pathlib.Path(rows[rid,'continue_a']['output'])/'policy_trace.npz');b_trace=arrays(pathlib.Path(rows[rid,'continue_b']['output'])/'policy_trace.npz');same=a_trace['action'].shape==b_trace['action'].shape;delta=float(abs(a_trace['action']-b_trace['action']).max()) if same else None;trajectory.append(dict(root_id=rid,same_steps=same,max_action_difference_if_same_length=delta))
    summary=statistics(ca,res,cb,taskids);np.savez_compressed(run/'paired_outcomes.npz',continue_a=ca,resample5=res,continue_b_indices=np.array(sorted(cb)),continue_b=np.array([cb[i] for i in sorted(cb)]),task_id=np.array(taskids));(run/'artifact_audit_index.json').write_text(json.dumps(hashes,indent=2)+'\n')
    cost={method:dict(branches=sum(k[1]==method for k in rows),policy_calls=sum(v['policy_calls'] for k,v in rows.items() if k[1]==method),steps=sum(v['steps'] for k,v in rows.items() if k[1]==method),inference_seconds=sum(v['inference_seconds'] for k,v in rows.items() if k[1]==method)) for method in ('continue_a','resample5','continue_b')}
    report=dict(status='COMPLETE',branches=130,expected_branches=130,fit_episodes=20,task_clusters=10,seeded_roots=60,policy_replicate_seeds=plan['policy_replicate_seeds'],selected_pair_ids=plan['selected_pair_ids'],all_input_replay_calls=model_calls,physics_controls=physics_controls,model_replay_max_error=maxmodel,physics_replay_max_error=maxphysics,summary=summary,repeat_control_trajectory=trajectory,cost=cost,calibration_used=0,confirmation_used=0,training_updates=0,old_results_mixed=False,elapsed_seconds=time.monotonic()-start,scope='Fixed fit-only repeated-policy-seed study of resample5. K=3 cannot certify per-root success probability; no confirmation or full-benchmark claim.',next_action='Independent full-result review, scope-limited interpretation, records and budget accounting. No automatic larger sampling or deployment.')
    (run/'SEED_STABILITY_RESULTS.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
