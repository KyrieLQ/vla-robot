"""Independently rebuild the fixed paired matrix from raw terminal predicates."""
import argparse,collections,json,os,pathlib,time,datetime
import numpy as np
from paired_live_audit_v1 import arrays,sha,verify_audit_bindings,check_saved_branch
ROOT=pathlib.Path(os.environ['PROJECT_ROOT'])
def read(p):return json.loads(pathlib.Path(p).read_text())
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();run=pathlib.Path(a.run).resolve();start=time.monotonic();status=read(run/'status.json');assert status['status']=='COMPLETE';plan=read(status['plan']);expected={(j['root_id'],j['method']) for j in plan['jobs']};roots={r['root_id']:r for r in plan['policy_seed_roots']};result=read(run/'SEED_STABILITY_RESULTS.json');index=read(run/'artifact_audit_index.json');assert len(index)==130
    for path,h in read(run/'source_manifest.json')['sha256'].items():assert sha(run/'frozen_source'/path)==h
    assert sha(ROOT/plan['seed_approval_file'])==plan['seed_approval_sha256'];table={};model_calls=0;physics=0;live_calls=collections.Counter();live_steps=collections.Counter()
    records={}
    for worker in (0,1):
        w=read(run/f'workers/worker{worker}/result.json');assert w['status']=='PASS' and len(w['branches'])==65
        for row in w['branches']:
            key=(row['root_id'],row['method']);assert key not in records and roots[key[0]]['worker']==worker;records[key]=row
    assert set(records)==expected
    for binding in index:
        key=(binding['root_id'],binding['method']);row=records[key];directory=pathlib.Path(row['output']);root=roots[key[0]]
        for name,h in binding['sha256'].items():assert sha(directory/name)==h
        tr,obs,oracle,metadata=check_saved_branch(directory,row,root,plan);verify_audit_bindings(directory,row);terminal=bool(oracle['goal_predicate'][-1,0]);assert terminal==bool(tr['done'][-1])==row['task_success'];table[key]=int(terminal)
        model_calls+=read(directory/'model_replay.json')['calls_checked'];physics+=key[1] in ('continue_a','continue_b');live_calls[key[1]]+=len(obs['call_step']);live_steps[key[1]]+=len(tr['action'])
    observed=[];control_discord=0;bytask=collections.defaultdict(list);paired=arrays(run/'paired_outcomes.npz');source_pairs=plan['selected_pair_ids'];both={}
    for rid,root in roots.items():both[root['pair_id'],root['replicate_index']]=rid
    for i,pair in enumerate(source_pairs):
        gains=[]
        for j in range(3):
            rid=both[pair,j];aa=table[rid,'continue_a'];rr=table[rid,'resample5'];assert aa==paired['continue_a'][i,j] and rr==paired['resample5'][i,j];gains.append(rr-aa)
            if (rid,'continue_b') in table:control_discord+=table[rid,'continue_b']!=aa
        observed.append(gains);bytask[roots[both[pair,0]]['task_id']].append(float(np.mean(gains)))
    gain=np.asarray(observed);varying=int(np.sum(np.max(gain,axis=1)!=np.min(gain,axis=1)));variance=np.mean([sum((row-row.mean())**2)/(len(row)-1) for row in gain]);summary=result['summary'];assert varying==summary['roots_with_varying_gain'] and control_discord==summary['a_b_success_discordance'];np.testing.assert_allclose(variance,summary['mean_within_root_sample_variance'],atol=1e-12,rtol=0)
    category='INCONCLUSIVE_IMPLEMENTATION_NOISE' if control_discord else ('HIGH_OBSERVED_SEED_VARIATION' if varying>=8 else ('LOW_OBSERVED_SEED_VARIATION' if varying<=2 else 'INCONCLUSIVE_INTERMEDIATE_VARIATION'));assert category==summary['interpretation'];assert observed==summary['per_root_gains'] and len(bytask)==10 and all(len(v)==2 for v in bytask.values())
    rng=np.random.default_rng(2026101410);values=np.array([np.mean(bytask[i]) for i in range(10)]);boot=values[rng.integers(0,10,(10000,10))].mean(1);np.testing.assert_allclose(np.quantile(boot,[.025,.975]),summary['task_cluster_descriptive_interval'],atol=1e-12,rtol=0)
    assert model_calls==result['all_input_replay_calls'] and physics==70
    for method in live_calls:assert live_calls[method]==result['cost'][method]['policy_calls'] and live_steps[method]==result['cost'][method]['steps']
    preflight=read(run/'COLLECTION_GATE.json');assert preflight['workers_passed']==2 and preflight['preflight_existing_controls_passed']==4
    out=run/'review';out.mkdir(exist_ok=True);report=dict(status='PASS',reviewed_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),branches=130,unique_fit_episodes=20,policy_seeded_roots=60,model_replay_calls=model_calls,physics_controls=physics,raw_goal_matrix_verified=True,all_frozen_hashes_passed=True,interpretation=category,roots_with_varying_gain=varying,a_b_success_discordance=control_discord,old_outcomes_mixed=False,confirmation_used=0,elapsed_seconds=time.monotonic()-start,review_script_sha256=sha(pathlib.Path(__file__)));(out/'acceptance.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
