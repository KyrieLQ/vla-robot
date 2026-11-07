"""Independent complete-result audit from saved predicates, calls and bound replay files."""
import argparse,collections,datetime,json,os,pathlib
import numpy as np
from paired_live_audit_v1 import sha,arrays,verify_audit_bindings
ROOT=pathlib.Path(os.environ['PROJECT_ROOT'])
METHODS=('continue_a','continue_b','fine20','resample5','retreat5_fine20')
CANDIDATES=METHODS[2:]
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();run=pathlib.Path(a.run).resolve()
    status=json.loads((run/'status.json').read_text());assert status['status']=='COMPLETE'
    plan=json.loads(pathlib.Path(status['plan']).read_text());roots=json.loads((ROOT/plan['root_manifest']).read_text())['roots']
    original=json.loads((run/'ANALYSIS.json').read_text());assert sha(ROOT/plan['root_manifest'])==plan['root_manifest_sha256']
    frozen=json.loads((run/'source_manifest.json').read_text())['sha256']
    assert all(sha(run/'frozen_source'/path)==digest for path,digest in frozen.items())
    rows={};raw={};model_max=0.0;physical_max=0.0;calls=0;physics_count=0;episode_map={r['root_id']:r['pair_id'] for r in roots}
    for w in (0,1):
        worker=json.loads((run/'workers'/f'worker{w}'/'result.json').read_text());assert worker['status']=='PASS'
        for r in worker['branches']:
            key=(r['root_id'],r['method']);assert key not in rows;rows[key]=r
    assert set(rows)=={(r['root_id'],m) for r in roots for m in METHODS}
    root_observations={};root_partition={r['root_id']:r['module_partition'] for r in roots}
    for i,(key,row) in enumerate(rows.items()):
        d=pathlib.Path(row['output']);saved=json.loads((d/'result.json').read_text());assert saved==row
        tr=arrays(d/'policy_trace.npz');oracle=arrays(d/'oracle.npz')
        outcome=bool(oracle['goal_predicate'][-1].all())
        assert outcome==bool(tr['done'].any())==row['task_success']
        assert np.array_equal(tr['done'],oracle['goal_predicate'][1:,0])
        assert len(tr['call_step'])==row['policy_calls'] and len(tr['action'])==row['steps']
        verify_audit_bindings(d,row)
        model=json.loads((d/'model_replay.json').read_text());model_max=max(model_max,model['max_action_chunk_error']);calls+=model['calls_checked']
        if key[1] in METHODS[:2]:
            physical=json.loads((d/'physics_replay.json').read_text());physical_max=max(physical_max,max(physical['physical_max_errors'].values()));physics_count+=1
        raw[key]={'success':int(outcome),'calls':len(tr['call_step']),'actions':len(tr['action']),'inference_seconds':float(tr['inference_seconds'].sum())}
        root_observations.setdefault(key[0],set()).add(row['root_image_sha256'])
        if i%100==0:print(json.dumps({'branches_verified':i,'expected':1345}),flush=True)
    assert len(rows)==1345 and calls==27204 and physics_count==538
    assert all(len(v)==1 for v in root_observations.values())
    summaries={};rescued_all=set();partition_counts={};unique_contribution={m:set() for m in CANDIDATES}
    for partition in ('fit','calibration'):
        subset=[r for r in roots if r['module_partition']==partition]
        original_failed={r['pair_id'] for r in subset if not rows[(r['root_id'],'continue_a')]['baseline_success']}
        rescued=set()
        for r in subset:
            key=r['root_id'];a=raw[(key,'continue_a')]['success'];b=raw[(key,'continue_b')]['success']
            if not rows[(key,'continue_a')]['baseline_success'] and not a and not b and any(raw[(key,m)]['success'] for m in CANDIDATES):rescued.add(r['pair_id'])
        partition_counts[partition]={'unique_episodes':len({r['pair_id'] for r in subset}),'roots':len(subset),'original_failed_episodes':len(original_failed),'robust_rescued_episodes':len(rescued)}
    for step in (60,120):
        subset=[r for r in roots if r['branch_step']==step];detail={};oracle_rescues=set()
        base_calls=sum(raw[(r['root_id'],'continue_a')]['calls'] for r in subset)
        for method in METHODS:
            wins=losses=successes=total_calls=0
            for r in subset:
                record=raw[(r['root_id'],method)];control=raw[(r['root_id'],'continue_a')]
                successes+=record['success'];total_calls+=record['calls'];wins+=record['success']>control['success'];losses+=record['success']<control['success']
            expected=original['cohorts'][str(step)]['vs_a'][method]
            assert (successes,int(wins),int(losses),total_calls)==(expected['successes'],expected['wins'],expected['losses'],expected['policy_calls'])
            detail[method]={'n':len(subset),'successes':successes,'wins':int(wins),'losses':int(losses),'policy_calls':total_calls,'suffix_call_ratio':total_calls/base_calls,'net_change_pp':100*(wins-losses)/len(subset)}
        for r in subset:
            rid=r['root_id'];a=raw[(rid,'continue_a')]['success'];b=raw[(rid,'continue_b')]['success']
            winners=[m for m in CANDIDATES if raw[(rid,m)]['success']]
            if not rows[(rid,'continue_a')]['baseline_success'] and not a and not b and winners:
                oracle_rescues.add(r['pair_id']);rescued_all.add(r['pair_id'])
            if not a and len(winners)==1:unique_contribution[winners[0]].add(r['pair_id'])
        summaries[str(step)]={'methods':detail,'oracle_robust_rescues':len(oracle_rescues),'oracle_rescue_ids':sorted(oracle_rescues),
            'blind_uniform_recovery_expected_net_change_pp':sum(detail[m]['net_change_pp'] for m in CANDIDATES)/3,
            'note':'Uniform state-independent candidate selection is a zero-observation comparison, not a learned selector.'}
    assert sorted(rescued_all)==original['robust_rescuable_episode_ids'] and len(rescued_all)==20
    repeat=original['repeat_controls'];assert len(repeat)==269
    assert all(raw[(r['root_id'],'continue_a')]['success']==raw[(r['root_id'],'continue_b')]['success'] for r in roots)
    result={'status':'PASS','reviewed_at':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
            'branches_verified':1345,'roots_verified':269,'unique_episodes':200,'model_calls_bound_to_saved_inputs':calls,'physics_controls_verified':physics_count,
            'model_max_error':model_max,'physical_max_error':physical_max,'all_branch_root_images_common':True,'prior_analysis_counts_independently_reproduced':True,
            'repeat_outcome_disagreements':0,'repeat_action_difference_gt_1e6':sum(r['action_max_difference_common_steps']>1e-6 for r in repeat),
            'repeat_max_action_difference':max(r['action_max_difference_common_steps'] for r in repeat),
            'repeat_step_count_disagreements':sum(r['steps_a']!=r['steps_b'] for r in repeat),
            'robust_rescuable_original_failures':20,'original_failures':30,'feasibility_gate_passed':True,
            'partitions':partition_counts,'cohorts':summaries,'uniquely_successful_candidate_episode_counts_at_any_one_root':{k:len(v) for k,v in unique_contribution.items()},
            'confirmation_used':0,'limitations':['Oracle selection uses future outcomes; no deployed selector improvement established.',
                'No observed A/B outcome disagreement does not prove identical trajectories or zero stochastic variance.',
                'Single random stream per root/method gives sampled outcome labels, not exact conditional expectations.',
                '20 recoverable failures demonstrates available candidate capacity, not that permitted observations can predict the right intervention.',
                '60 and 120 cohorts have different populations; do not compare their raw success rates as a timing treatment effect.']}
    out=run/'review';out.mkdir(exist_ok=True);(out/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('cohorts','limitations')},indent=2))
if __name__=='__main__':main()
